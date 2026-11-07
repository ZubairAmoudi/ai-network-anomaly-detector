"""Data preprocessing pipeline.

Provides ``load_and_clean_data``, which loads a raw CSV file, normalises
column names, converts string placeholders and unparseable numeric entries to
missing values, removes infinite and missing values, and persists the cleaned
dataset to ``data/processed/``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

# Project root is the parent of the ``src`` directory containing this file.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Cell values (compared case-insensitively after stripping whitespace) that are
# treated as missing. An empty string also covers whitespace-only cells.
# "none" is deliberately NOT included: it can be a legitimate categorical value
# (e.g. TCP_Flags == "NONE" in NetFlow data).
DEFAULT_PLACEHOLDERS: frozenset[str] = frozenset(
    {"", "?", "n/a", "na", "#n/a", "nan", "null"}
)

# Minimum fraction of non-missing values in a text column that must parse as
# numbers for the column to be coerced to a numeric dtype.
DEFAULT_NUMERIC_THRESHOLD = 0.9

logger = logging.getLogger(__name__)


def _is_text_column(series: pd.Series) -> bool:
    """Return True for object or string-dtype columns."""
    return series.dtype == object or isinstance(series.dtype, pd.StringDtype)


def _replace_placeholders(df: pd.DataFrame, placeholders: frozenset[str]) -> int:
    """Strip text cells and replace placeholder strings with NaN in place.

    Returns the number of cells converted to NaN.
    """
    replaced = 0
    for col in df.columns:
        if not _is_text_column(df[col]):
            continue
        stripped = df[col].str.strip()
        mask = stripped.str.lower().isin(placeholders)
        replaced += int(mask.sum())
        df[col] = stripped.mask(mask, np.nan)
    return replaced


def _coerce_numeric_columns(
    df: pd.DataFrame,
    exclude: set[str],
    threshold: float,
) -> dict[str, int]:
    """Coerce predominantly numeric text columns to numbers in place.

    A text column is coerced when at least ``threshold`` of its non-missing
    values parse as numbers; entries that do not parse become NaN. Columns in
    ``exclude`` and genuinely textual columns are left unchanged.

    Returns a mapping of coerced column name -> number of entries turned NaN.
    """
    coerced: dict[str, int] = {}
    for col in df.columns:
        if col in exclude or not _is_text_column(df[col]):
            continue
        original = df[col]
        non_missing = int(original.notna().sum())
        if non_missing == 0:
            continue
        converted = pd.to_numeric(original, errors="coerce")
        parsed = int(converted.notna().sum())
        ratio = parsed / non_missing
        if ratio >= threshold:
            df[col] = converted
            coerced[col] = non_missing - parsed
        elif parsed > 0:
            logger.warning(
                "Column %r is only %.1f%% numeric (threshold %.1f%%); "
                "left as text.", col, 100 * ratio, 100 * threshold,
            )
    return coerced


def load_and_clean_data(
    input_path: str | Path,
    output_dir: str | Path = PROCESSED_DIR,
    output_filename: str | None = None,
    encoding: str = "utf-8",
    label_columns: Iterable[str] = (),
    placeholders: Iterable[str] | None = None,
    numeric_threshold: float = DEFAULT_NUMERIC_THRESHOLD,
) -> pd.DataFrame:
    """Load a CSV file, clean it, and save the result.

    Cleaning steps:
        1. Strip leading/trailing whitespace from all column names.
        2. Strip whitespace from text cells and replace placeholder strings
           (``"N/A"``, ``"null"``, ``"?"``, empty/whitespace-only, ...;
           case-insensitive) with ``NaN``.
        3. Coerce predominantly numeric text columns (excluding
           ``label_columns``) to numbers; unparseable entries become ``NaN``.
        4. Replace ``np.inf`` and ``-np.inf`` with ``NaN``.
        5. Drop every row containing at least one missing value.

    Args:
        input_path: Path to the source CSV file.
        output_dir: Directory where the cleaned CSV is written.
            Defaults to ``<project_root>/data/processed``.
        output_filename: Name of the output file. Defaults to
            ``<input_stem>_cleaned.csv``.
        encoding: Text encoding of the input file.
        label_columns: Target/label columns that are never coerced to numeric
            (placeholders in them are still treated as missing).
        placeholders: Strings treated as missing. Defaults to
            ``DEFAULT_PLACEHOLDERS``. Matching is case-insensitive and ignores
            surrounding whitespace.
        numeric_threshold: Fraction (0 < x <= 1) of non-missing values in a
            text column that must parse as numbers for it to be coerced.

    Returns:
        The cleaned ``DataFrame`` (index reset).

    Raises:
        FileNotFoundError: If ``input_path`` does not exist or is not a file.
        ValueError: If the file is not a ``.csv``, is empty, cannot be
            parsed, has duplicate column names after stripping, references
            unknown label columns, has an invalid threshold, or has no rows
            left after cleaning.
    """
    input_path = Path(input_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    if not 0 < numeric_threshold <= 1:
        raise ValueError(
            f"numeric_threshold must be in (0, 1], got {numeric_threshold}"
        )
    placeholder_set = (
        DEFAULT_PLACEHOLDERS
        if placeholders is None
        else frozenset(p.strip().lower() for p in placeholders)
    )

    # ---- Validate input -------------------------------------------------
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if input_path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv file, got: {input_path.name}")

    # ---- Load -------------------------------------------------------------
    try:
        df = pd.read_csv(input_path, encoding=encoding, low_memory=False)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Input file is empty: {input_path}") from exc
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Failed to parse CSV {input_path}: {exc}") from exc

    initial_rows = len(df)
    logger.info("Loaded %d rows x %d columns from %s",
                initial_rows, df.shape[1], input_path)

    # ---- Clean column names ----------------------------------------------
    df.columns = df.columns.astype(str).str.strip()
    duplicated = df.columns[df.columns.duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(
            f"Duplicate column names after stripping whitespace: {duplicated}"
        )

    labels = {str(c).strip() for c in label_columns}
    unknown = sorted(labels - set(df.columns))
    if unknown:
        raise ValueError(f"Label columns not found in data: {unknown}")

    # ---- Replace string placeholders with NaN ----------------------------
    n_placeholders = _replace_placeholders(df, placeholder_set)
    logger.info("Converted %d placeholder cells to NaN", n_placeholders)

    # ---- Coerce numeric columns ------------------------------------------
    coerced = _coerce_numeric_columns(df, labels, numeric_threshold)
    for col, n_bad in coerced.items():
        logger.info("Coerced column %r to numeric (%d unparseable -> NaN)",
                    col, n_bad)

    # ---- Replace infinities with NaN -------------------------------------
    # Done after coercion so that "inf" strings parsed above are caught too.
    df = df.replace([np.inf, -np.inf], np.nan)

    # ---- Drop rows with missing values -----------------------------------
    df = df.dropna(how="any").reset_index(drop=True)
    dropped = initial_rows - len(df)
    logger.info("Dropped %d rows containing missing/invalid values "
                "(%d remaining)", dropped, len(df))

    if df.empty:
        raise ValueError(
            "No rows remain after cleaning; refusing to write an empty dataset."
        )

    # ---- Save ------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_filename is None:
        output_filename = f"{input_path.stem}_cleaned.csv"
    output_path = output_dir / output_filename

    if output_path == input_path:
        raise ValueError("Output path must differ from input path.")

    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info("Saved cleaned dataset to %s", output_path)

    return df


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean a raw CSV file and save it to data/processed/."
    )
    parser.add_argument("input_path", type=Path, help="Path to the raw CSV file.")
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR,
                        help=f"Output directory (default: {PROCESSED_DIR}).")
    parser.add_argument("--output-filename", default=None,
                        help="Output filename (default: <input>_cleaned.csv).")
    parser.add_argument("--encoding", default="utf-8",
                        help="Input file encoding (default: utf-8).")
    parser.add_argument("--label-columns", nargs="*", default=[],
                        help="Target/label columns excluded from numeric coercion.")
    parser.add_argument("--numeric-threshold", type=float,
                        default=DEFAULT_NUMERIC_THRESHOLD,
                        help="Min fraction of numeric values for a text column "
                             f"to be coerced (default: {DEFAULT_NUMERIC_THRESHOLD}).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    try:
        load_and_clean_data(
            args.input_path,
            args.output_dir,
            args.output_filename,
            args.encoding,
            label_columns=args.label_columns,
            numeric_threshold=args.numeric_threshold,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
