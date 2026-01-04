from SETTINGS import *
import pandas as pd
from pathlib import Path
from typing import List, Literal
import math
import re
from fractions import Fraction
from functools import reduce
import numpy as np
from utils import get_full_results_df, inner_join_on_ack_id, create_snippets_df

########################   Constants    ########################
TableType = Literal['matching', 'autoenrollment', 'vesting']

dataset_cols_dict = {
    "matching": [MATCH_RATE_1_COL, CAP_1_COL, MATCH_RATE_2_COL, CAP_2_COL, MATCH_RATE_3_COL, CAP_3_COL],
    "autoenrollment": [AE_OFFERED_COL, AE_INIT_DEF_COL, AE_AINC_OFFERED_COL, AE_AINC_AMT_COL, AE_AINC_CAP_COL],
    "vesting": [VESTING_OFFERED_COL] + [VESTING_YEAR_COL_STEM+f'{i}' for i in range(NUM_VESTING_YEARS)]
}

llm_cols_dict = {
    "matching": ['match_formula', MATCH_RATE_1_COL, CAP_1_COL, MATCH_RATE_2_COL, CAP_2_COL, MATCH_RATE_3_COL, CAP_3_COL],
    "autoenrollment": ["auto_enrollment_offered", "initial_deferral", "auto_increase_offered", "auto_increase_amount", "auto_increase_cap"],
    "vesting": [VESTING_OFFERED_COL+'_offered'] + [VESTING_YEAR_COL_STEM+f'{i}' for i in range(NUM_VESTING_YEARS)]
}

llm_tabulator_names_dict = {
    "matching": ['f1', 'f2'],
    "autoenrollment": ['f1_AE', 'f2_AE'],
    "vesting": ['f1_VESTING', 'f2_VESTING']
}

llm_snippet_names_dict = {
    "matching": ['g1', 'g2'],
    "autoenrollment": ['g1_AE', 'g2_AE'],
    "vesting": ['g1_VESTING', 'g2_VESTING']
}

snippet_col_dict = {
    'matching': RAW_MATCHING_SNIPPET_COL,
    'autoenrollment': RAW_AE_SNIPPET_COL,
    'vesting': RAW_VESTING_SNIPPET_COL,
}

snippet_no_mention_dict = {
    'matching': "No mention of employer matching.",
    'autoenrollment': "No mention of auto-enrollment.",
    'vesting': "No mention of vesting.",
}
######################## Pre-processing ########################

def normalize_feature_formula(x: str):
    try:
        s = x.strip()
        if (s == 'Yes') or (s == 'Unknown'):
            return s
        return "More complicated"
    except:
        return "Unknown"

def format_table_entries(table_info: List[str | float]):
    formatted = []
    for x in table_info:
        if pd.isna(x):
            formatted.append("NA")
            continue
        try:
            formatted.append(float_to_str(float(x)))
        except (TypeError, ValueError):
            formatted.append(str(x).strip())
    return formatted

def generate_correct_table(df: pd.DataFrame, table_type: TableType):
    out: List[str] = []

    for _, row in df.iterrows():
        formatted = get_table_info(row, table_type)
        output = " | ".join(llm_cols_dict[table_type])
        output += ("\n" + "-"*len(output))
        output += ("\n" + " | ".join(formatted))
        out.append(output)

    return out

def get_table_info(row: pd.DataFrame, table_type: TableType):
    cols = dataset_cols_dict[table_type]
    # table_info = [row.get(c) for c in cols]

    if table_type == 'matching':
        match_formula = normalize_feature_formula(row.get(MATCH_FORMULA_COL))
        table_info = [row.get(c) for c in cols]
        
        if CONVERT_MARGINAL_BOOL:
            table_info = convert_marginal(table_info)

        table_info = [match_formula] + table_info
        
        if match_formula != 'Yes':
            table_info[1:] = [np.nan]*len(table_info[1:])

    elif table_type == 'autoenrollment':
        ae_offered = normalize_feature_formula(row.get(AE_OFFERED_COL))
        ainc_offered = normalize_feature_formula(row.get(AE_AINC_OFFERED_COL))   

        table_info = [row.get(c) for c in cols]

        if ae_offered != "Yes":
            table_info[1:2] = [np.nan]
        if ainc_offered !="Yes":
            table_info[2:] = [np.nan]*len(table_info[2:])

        table_info[0] = ae_offered
        table_info[2] = ainc_offered
        # cols = ["auto_enrollment_offered", "initial_deferral", "auto_increase_offered", "auto_increase_amount", "auto_increase_cap"]

    elif table_type == 'vesting':
        vesting_offered = normalize_feature_formula(row.get(VESTING_OFFERED_COL))
        table_info = [row.get(c) for c in cols]
        table_info[0] = vesting_offered
        
        if vesting_offered != "Yes":
            table_info[1:] = [np.nan]*len(table_info[1:])

        # cols[0] += '_offered'

    else:
        raise ValueError(f"Unsupported table type given: {table_type}")

    return format_table_entries(table_info)#, cols

def generate_correct_snippet_col(df: pd.DataFrame, table_type: TableType) -> List[str]:
    """
    Clean and UTF-8-sanitize the snippet column.
    - Replaces NaN with no_mention_string
    - Forces string dtype
    - Trims whitespace; empty -> no_mention_string
    - Replaces literal "Missing" (case-insensitive, ignoring surrounding whitespace) -> no_mention_string
    - Encodes/decodes to coerce to valid UTF-8
    - Returns a list of clean strings
    """
    colname = snippet_col_dict[table_type]
    no_mention_string = snippet_no_mention_dict[table_type]

    s = (
        df[colname]
        .fillna(no_mention_string)
        .astype(str)
        .str.strip()
        .str.replace(r"^\s*missing\s*$", no_mention_string, regex=True, flags=re.IGNORECASE)
        .replace("", no_mention_string)
        .str.encode("latin1", errors="ignore")
        .str.decode("utf-8", errors="ignore")
    )

    return s.tolist()

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
        return str(int(n)).strip()
    return str(n).strip()

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

######################## Post-processing ########################

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

def is_numeric(x) -> bool:
    return not pd.isna(pd.to_numeric(x, errors="coerce"))

def extract_and_clean_entries_from_llm_table(table: str):
    return [float_to_str(x) if is_numeric(x) else str(x).strip() for x in extract_entries_from_llm_table(table)]


def extract_headers_from_llm_table(table: str):
    """
    Extracts the entries from the top row of a Markdown-style table.

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

def check_same_table(t1: str, t2: str):
    l1 = [x for x in extract_and_clean_entries_from_llm_table(t1)] or []
    l2 = [x for x in extract_and_clean_entries_from_llm_table(t2)] or []
    return l1 == l2

def get_boolean_accuracy_col(llm_output: [list | pd.Series], correct_col: [list | pd.Series]):
    df = pd.DataFrame({'llm': llm_output, 'ans': correct_col})
    return df.apply(lambda row: check_same_table(row['llm'], row['ans']), axis=1).rename("is_correct")

# correct_col should only contain tables and More complicated
def check_accuracy(llm_output: list, correct_col: [list | pd.Series]):
    score = get_boolean_accuracy_col(llm_output, correct_col)
    return score.mean()

def table_to_dict(table: str):
    # if check_more_complicated(table):
    #     raise ValueError(f"Tried to make a dictionary from a non-table: {table}")
    keys = extract_headers_from_llm_table(table)
    vals = extract_and_clean_entries_from_llm_table(table)
    return {k: v for k, v in zip(keys, vals)}

def check_proper_table(table: str, table_type: TableType):
    cols = extract_headers_from_llm_table(table)
    vals = extract_and_clean_entries_from_llm_table(table)

    correct_cols = llm_cols_dict[table_type]

    if len(cols) != len(vals):
        return False
    
    if len(cols) != len(correct_cols):
        return False

    for x in cols:
        if x not in correct_cols:
            return False
    
    return True

def create_no_mention_of_feature_flag(df, snippet_cols, table_type: TableType):
    no_mention_snip = snippet_no_mention_dict[table_type]
    snippets = df[snippet_cols].apply(lambda col: col.str.strip())
    eq_series = (snippets.nunique(axis=1) == 1) & (snippets.iloc[:, 0].astype(str).str.strip() == no_mention_snip)
    return eq_series


def process_feature_results_df(df: pd.DataFrame, model_name: str, result_col: str, table_type: TableType):
    llm_results = df[result_col].astype(str).tolist()
    
    expanded_dicts = []
    error = []
    for t in llm_results:
        
        good_table_bool = check_proper_table(t, table_type)

        error.append(not good_table_bool)
        if good_table_bool:
            d = table_to_dict(t)
            b = {f"{model_name}_{k}": v for k, v in d.items()}
            expanded_dicts.append(b)
        else:
            expanded_dicts.append({})   # no columns added for this row

    # Turn list of dicts into a DataFrame (handles ragged keys automatically)
    expanded_df = pd.DataFrame(expanded_dicts, index=df.index).replace("NA", None)
    

    # Optional: ensure consistent column order (key-sorted)
    # expanded_df = expanded_df.reindex(sorted(expanded_df.columns), axis=1)

    # Add a flag column for complicated rows
    expanded_df[f"{model_name}_error"] = error

    # Merge back to original
    out = pd.concat([df, expanded_df], axis=1)
    return out


def get_feature_results_df(table_type: TableType):
    df_list = []
    model_names = llm_tabulator_names_dict[table_type]
    for m in model_names:
        cfg = REGISTRY.get(m)

        m_df = get_full_results_df(cfg['name'], 
                                   cfg['result_col'], 
                                   cfg['oos_results_dir'])

        m_df_processed = process_feature_results_df(m_df, cfg['name'], 
                                                          cfg['result_col'], 
                                                          table_type)

        df_list.append(m_df_processed)
    
    df = inner_join_on_ack_id(df_list)
    return df

def post_process_feature_results(table_type: TableType):
    df = get_feature_results_df(table_type)
    
    tabulator_names = llm_tabulator_names_dict[table_type]
    cfg_1, cfg_2 = REGISTRY.get(tabulator_names[0]), REGISTRY.get(tabulator_names[1])

    llm_col_1, llm_col_2 = df[cfg_1['result_col']].astype(str), df[cfg_2['result_col']].astype(str)

    agree_rate = check_accuracy(llm_col_1, llm_col_2)

    print(f"LLMs agree on {agree_rate*100:.3f}% of plans ({agree_rate*len(df)} rows)")

    df[f'{table_type}_models_agree'] = get_boolean_accuracy_col(llm_col_1, llm_col_2)

    snippets_df = create_snippets_df(llm_snippet_names_dict[table_type])
    df = df.merge(snippets_df, on='ack_id')

    snippet_names = llm_snippet_names_dict[table_type]
    cfg_3, cfg_4 = REGISTRY.get(snippet_names[0]), REGISTRY.get(snippet_names[1])

    snippet_result_col_lst = [cfg_3['result_col'], cfg_4['result_col']]

    df[f'no_mention_of_{table_type}'] = create_no_mention_of_feature_flag(df, snippet_result_col_lst, table_type)

    print(df)
    for x in df.columns:
        print(x)

    if df['year_x'].equals(df['year_y']):
        df.drop(columns='year_y', inplace=True)
        df.rename(columns={'year_x': 'year'}, inplace=True)

    df = df.convert_dtypes()

    df.to_stata(FINAL_DATASETS_SAVE_DIR / f'{table_type}_full_dataset_unfiltered.dta', write_index=False, version=118)
    df2 = df.drop(columns=snippet_result_col_lst)
    df2.to_stata(FINAL_DATASETS_SAVE_DIR / f'{table_type}_full_dataset_unfiltered_no_snippets.dta', write_index=False, version=118)

    df_filtered = df[df[f'{table_type}_models_agree']].reset_index(drop=True)
    
    df_filtered.to_stata(FINAL_DATASETS_SAVE_DIR / f'{table_type}_full_dataset_filtered_models_agree.dta', write_index=False, version=118)

    df_filtered2 = df_filtered.drop(columns=snippet_result_col_lst)

    df_filtered2.to_stata(FINAL_DATASETS_SAVE_DIR / f'{table_type}_full_dataset_filtered_models_agree_no_snippets.dta', write_index=False, version=118)

    return df, df_filtered, df_filtered2
