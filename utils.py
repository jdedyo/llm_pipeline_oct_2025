from SETTINGS import *
import pandas as pd
from pathlib import Path
from typing import List
import math
import re
from fractions import Fraction

def load_data(path: Path) -> pd.DataFrame:
    """
    Load a dataset from any common file format into a pandas DataFrame.

    Supported extensions:
        - .csv, .tsv, .txt
        - .xlsx, .xls
        - .parquet
        - .feather
        - .json
        - .pkl / .pickle
        - .dta

    Raises:
        FileNotFoundError: if path does not exist
        ValueError: if file extension is unsupported
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    ext = path.suffix.lower()

    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in [".tsv", ".txt"]:
        return pd.read_csv(path, sep="\t")
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    elif ext == ".json":
        return pd.read_json(path)
    elif ext == ".parquet":
        return pd.read_parquet(path)
    elif ext == ".feather":
        return pd.read_feather(path)
    elif ext in [".pkl", ".pickle"]:
        return pd.read_pickle(path)
    elif ext == ".dta":
        return pd.read_stata(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def save_data(df: pd.DataFrame, path: Path, **kwargs) -> None:
    """
    Save a pandas DataFrame to disk, using the file format implied by `path`.

    Supported extensions:
        - .csv, .tsv, .txt
        - .xlsx, .xls
        - .parquet
        - .feather
        - .json
        - .pkl / .pickle
        - .dta

    Extra keyword args (**kwargs) are passed to the underlying pandas writer.

    Raises:
        ValueError: if file extension is unsupported
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()

    if ext == ".csv":
        df.to_csv(path, index=False, **kwargs)
    elif ext in [".tsv", ".txt"]:
        df.to_csv(path, sep="\t", index=False, **kwargs)
    elif ext in [".xlsx", ".xls"]:
        df.to_excel(path, index=False, **kwargs)
    elif ext == ".json":
        df.to_json(path, orient="records", lines=False, **kwargs)
    elif ext == ".parquet":
        df.to_parquet(path, index=False, **kwargs)
    elif ext == ".feather":
        df.to_feather(path, **kwargs)
    elif ext in [".pkl", ".pickle"]:
        df.to_pickle(path, **kwargs)
    elif ext == ".dta":
        df.to_stata(path, write_index=False, **kwargs)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def normalize_match_formula(x):
        if isinstance(x, str) and x.strip().lower() == "yes":
            return "Yes"
        else:
            return "More complicated"

def float_to_str(number):
    try:
        # Handle fractions like "2/3"
        if isinstance(number, str) and "/" in number:
            n = float(Fraction(number))
        else:
            n = float(number)
    except Exception:
        return "NA"
    # correct for floating point error
    n = round(n, 5)
    if n <= 0:
        return "NA"
    if math.floor(n) == math.ceil(n):
        return str(int(n))
    return str(n)

# length of nonzero entries in list; assumes list has 6 entries
def nonzero_length(inputList):
    if inputList[0] == 0:
        return 0
    if inputList[2] == 0:
        return 2
    if inputList[4] == 0:
        return 4
    return 6

# given list of 6 floats with cumulative caps, convert to marginal
# does not check the validity of the inputList, like whether it's already marginal or not 
def convert_marginal(inputList):
    output = inputList
    if nonzero_length(output) == 4:# and output[3] > output[1]:
        output[3] -= output[1]
    elif nonzero_length(output) == 6:# and output[3] > output[1] and output[5] > output[3]:
        output[5] -= output[3]
        output[3] -= output[1]
    return output

# Convert to latin-1 encoding to be able to save to .dta format
def make_latin(s):
    s = str(s)
    string_encode = s.encode("latin-1", "ignore")
    return string_encode.decode('latin-1')


def generate_correct_matching_table(df: pd.DataFrame) -> List[str]:
    cols = [MATCH_RATE_1_COL, CAP_1_COL, MATCH_RATE_2_COL, CAP_2_COL, MATCH_RATE_3_COL, CAP_3_COL]
    out: List[str] = []

    for _, row in df.iterrows():
        # Decide whether this row is a simple 'Yes' formula
        val = row.get(MATCH_FORMULA_COL)
        is_yes = isinstance(val, str) and val.strip().lower() == "yes"

        if not is_yes:
            out.append("More complicated")
            continue
        
        table_info = [row.get(c) for c in cols]
        if CONVERT_MARGINAL_BOOL:
            table_info = convert_marginal(table_info)
        
        # table_info = [float_to_str(x) if not pd.isna(x) else 'NA' for x in table_info]
        # Format each cell: NA -> "NA", else float_to_str if numeric, else str(x)
        formatted = []
        for x in table_info:
            if pd.isna(x):
                formatted.append("NA")
                continue
            try:
                formatted.append(float_to_str(float(x)))
            except (TypeError, ValueError):
                formatted.append(str(x))
        
        output = "match_rate_1 | cap_1 | match_rate_2 | cap_2 | match_rate_3 | cap_3"
        output += ("\n" + "------------------------------------------------------------------")
        output += ("\n" + " | ".join(formatted))

        out.append(output)

    return out

def generate_correct_matching_snippet_col(df: pd.DataFrame) -> List[str]:
    """
    Clean and UTF-8-sanitize the RAW_MATCHING_SNIPPET_COL column.
    - Replaces NaN with ''
    - Forces string dtype
    - Encodes/decodes as UTF-8, replacing invalid bytes
    - Returns a list of clean strings
    """
    out = (
        df[RAW_MATCHING_SNIPPET_COL]
        .fillna('No mention of employer matching.')
        .astype(str)
        .apply(lambda x: x.strip() or 'No mention of employer matching.')
        .str.encode("latin1", errors="ignore")
        .str.decode("utf-8", errors="ignore")
        .tolist()
    )
    return out

def load_ocr_data(path: Path=ALL_OCR_TEXT_DATA_PATH) -> pd.DataFrame:
    """
    Loads only the ack_id and ALL_OCR_TEXT_COL from any common file format into a pandas DataFrame.

    Supported extensions:
        - .csv, .tsv, .txt
        - .xlsx, .xls
        - .parquet
        - .feather
        - .json
        - .pkl / .pickle
        - .dta

    Raises:
        FileNotFoundError: if path does not exist
        ValueError: if file extension is unsupported
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    usecols = ["ack_id", ALL_OCR_TEXT_COL]
    ext = path.suffix.lower()

    if ext == ".csv":
        df = pd.read_csv(path, usecols=usecols)
    elif ext in [".tsv", ".txt"]:
        df = pd.read_csv(path, sep="\t", usecols=usecols)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path, usecols=usecols)
    elif ext == ".json":
        df = pd.read_json(path)
    elif ext == ".parquet":
        df = pd.read_parquet(path, columns=usecols)
    elif ext == ".feather":
        df = pd.read_feather(path, columns=usecols)
    elif ext in [".pkl", ".pickle"]:
        df = pd.read_pickle(path)
    elif ext == ".dta":
        df = pd.read_stata(path, columns=usecols)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    # Filter explicitly in case usecols isn't supported
    df = df[[c for c in usecols if c in df.columns]]
    return df

def parse_timestamp(path: Path, prefix: str) -> datetime | None:
    name_no_ext = path.stem
    if not name_no_ext.startswith(prefix + "_"):
        return None

    timestamp_str = name_no_ext.removeprefix(prefix + "_")
    try:
        return datetime.strptime(timestamp_str, "%m_%d_%Y-%I_%p")
    except ValueError:
        return None

def remove_timestamp(path: Path) -> Path:
    """
    Remove the datetime suffix from a filename following the pattern:
    g1_MM_DD_YYYY-HH_AM/PM → g1

    Works on either Path or string inputs and returns a Path with the
    same parent directory and extension (if any).

    Examples:
        g1_10_08_2025-09_PM  → g1
        g1_10_08_2025-10_AM  → g1
        g2_10_09_2025-11_PM  → g2
    """
    name = path.stem  # remove extension if any
    # Regex to match suffix like "_10_08_2025-09_PM"
    new_name = re.sub(r"_\d{2}_\d{2}_\d{4}-\d{2}_(AM|PM)$", "", name)
    return path.with_name(new_name + path.suffix)

def get_most_recent_file(p: Path):
    loc = remove_timestamp(p)
    base_name = loc.stem
    parent = loc.parent

    # Collect candidates that start with base_name + '_'
    candidates = list(parent.glob(base_name + "_*"))

    if not candidates:
        # Check if the non-timestamped version exists
        non_timestamped = parent / f"{base_name}{p.suffix}"
        if non_timestamped.exists():
            return non_timestamped
        else:
            raise FileNotFoundError(
                f"No timestamped or base file found matching '{base_name}' in {parent}"
            )

    # Debug print without exhausting an iterator later
    # print([str(p) for p in candidates])

    # Keep only those whose names parse to a timestamp
    parsed = [(p, parse_timestamp(p, base_name)) for p in candidates]
    parsed = [(p, dt) for p, dt in parsed if dt is not None]

    if not parsed:
        raise FileNotFoundError(
            f"Found {len(candidates)} candidates, but none matched the timestamp pattern "
            f"'{base_name}_MM_DD_YYYY-HH_AM/PM'."
        )

    # Pick the latest by timestamp
    most_recent = max(parsed, key=lambda x: x[1])[0]
    return most_recent

def get_chunk_num_range() -> List[int]:
    pattern = re.compile(r"^oos_chunk_(\d+)\.csv$")
    nums = []
    for path in OOS_DATA_DIR.glob("oos_chunk_*.csv"):
        m = pattern.match(path.name)
        if m:
            nums.append(int(m.group(1)))
    return sorted(nums)

def get_oos_chunk_path_from_num(chunk_num: int, chunk_dir: Path=OOS_DATA_DIR) -> pd.DataFrame:
    p = chunk_dir / f"oos_chunk_{chunk_num}.csv"
    if not p.exists():
        raise FileNotFoundError(f"No chunk file found for number {chunk_num}: {p}")
    return p

def load_oos_chunk_from_num(chunk_num: int) -> pd.DataFrame:
    return load_data(get_oos_chunk_path_from_num(chunk_num))

def save_oos_chunk_results(chunk_num: int, results: List[str], cfg: ModelRegistry):
    col = cfg['result_col']
    
    df = load_oos_chunk_from_num(chunk_num)

    df[col] = results

    p = Path(cfg['oos_results_dir'])
    p.mkdir(parents=True, exist_ok=True)

    save_data(df, cfg['oos_results_dir'] / f"oos_chunk_{chunk_num}.csv")

    return df


def extract_entries_from_llm_table(table: str):
    """
    Extracts the entries from the bottom row of a Markdown-style table.

    Example input:
        match_rate_1 | cap_1 | match_rate_2 | cap_2 | match_rate_3 | cap_3
        ------------------------------------------------------------------
        0.5 | 0.06 | NA | NA | NA | NA

    Returns:
        ['0.5', '0.06', 'NA', 'NA', 'NA', 'NA']
    """
    # Find the last line that contains at least one '|' and a non-dash character
    lines = [line.strip() for line in table.strip().splitlines() if "|" in line]
    if not lines:
        return []
    
    bottom_line = lines[-1]

    # Extract values between | ... |, stripping extra spaces
    matches = re.findall(r"\|\s*([^|]+?)\s*(?=\|)", f"|{bottom_line}|")

    return matches

def extract_and_clean_entries_from_llm_table(table: str):
    return [float_to_str(x) for x in extract_entries_from_llm_table(table)]


def extract_headers_from_llm_table(table: str):
    """
    Extracts the entries from the bottom row of a Markdown-style table.

    Example input:
        match_rate_1 | cap_1 | match_rate_2 | cap_2 | match_rate_3 | cap_3
        ------------------------------------------------------------------
        0.5 | 0.06 | NA | NA | NA | NA

    Returns:
        ['match_rate_1', 'cap_1', 'match_rate_2', 'cap_2', 'match_rate_3', 'cap_3']
    """
    # Find the last line that contains at least one '|' and a non-dash character
    lines = [line.strip() for line in table.strip().splitlines() if "|" in line]
    if not lines:
        return []
    
    top_line = lines[0]

    # Extract values between | ... |, stripping extra spaces
    matches = re.findall(r"\|\s*([^|]+?)\s*(?=\|)", f"|{top_line}|")

    return matches

def check_more_complicated(table: str):
    pattern = re.compile(r"\bcomplicated\b", re.IGNORECASE)
    return bool(pattern.search(table))

def check_same_table(t1: str, t2: str):
    if check_more_complicated(t1) and check_more_complicated(t2):
        return True
    elif check_more_complicated(t1) or check_more_complicated(t2):
        return False

    l1 = [float_to_str(x) for x in extract_entries_from_llm_table(t1)] or []
    l2 = [float_to_str(x) for x in extract_entries_from_llm_table(t2)] or []
    return l1 == l2

def get_boolean_accuracy_col(llm_output: list, correct_col: [list | pd.Series]):
    df = pd.DataFrame({'llm': llm_output, 'ans': correct_col})
    return df.apply(lambda row: check_same_table(row['llm'], row['ans']), axis=1).rename("is_correct")

# correct_col should only contain tables and More complicated
def check_accuracy(llm_output: list, correct_col: [list | pd.Series]):
    score = get_boolean_accuracy_col(llm_output, correct_col)
    return score.mean()

def table_to_dict(table: str):
    if check_more_complicated(table):
        raise ValueError(f"Tried to make a dictionary from a non-table: {table}")
    keys = extract_headers_from_llm_table(table)
    vals = extract_and_clean_entries_from_llm_table(table)
    return {k: v for k, v in zip(keys, vals)}

def check_proper_table(table: str, correct_cols: List[str]):
    if check_more_complicated(table):
        return True
    
    cols = extract_headers_from_llm_table(table)

    for x in cols:
        if x not in correct_cols:
            return False
    
    return True
