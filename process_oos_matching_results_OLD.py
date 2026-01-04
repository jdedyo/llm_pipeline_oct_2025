from SETTINGS import *
from pathlib import Path
import pandas as pd
from functools import reduce
from utils_OLD import (
    table_to_dict, check_same_table, check_accuracy, 
    check_more_complicated, check_proper_table, get_boolean_accuracy_col,
    create_matching_snippets_df, create_no_mention_of_matching_flag
    )

TABLE_COLS = ['match_rate_1', 'cap_1', 'match_rate_2', 'cap_2', 'match_rate_3', 'cap_3']

SAVE_PATH = OOS_RESULTS_DIR / 'datasets'
SAVE_PATH.mkdir(parents=True, exist_ok=True)

def glob_oos_results(model: str, base_dir: Path=OOS_RESULTS_DIR):
    res = (base_dir / model).glob("*.csv")
    out = list(res)
    # print(out)
    return out#[:3]

def get_full_results_df(model: str, result_col: str=None):
    col = result_col or f"{model}_table"

    df_list = [pd.read_csv(x, usecols=['ack_id', 'year', col]) for x in glob_oos_results(model)]
    # df_list = [pd.read_csv(x) for x in glob_oos_results(model)]
    res = pd.concat(df_list, ignore_index=True)
    res["ack_id"] = res["ack_id"].astype(str).str.strip()
    print(f"Loaded {len(df_list)} dataframes for {model}, totaling {len(res):,} rows!")
    return res


def inner_join_on_ack_id(df_list):
    if not df_list:
        return pd.DataFrame()

    # normalize key just in case
    for df in df_list:
        df['ack_id'] = df['ack_id'].astype(str).str.strip()

    if len(df_list) == 1:
        return df_list[0]

    return reduce(lambda l, r: pd.merge(l, r, on='ack_id', how='inner'), df_list)

def process_df(df: pd.DataFrame, model: str, result_col: str=None):
    col = result_col or f"{model}_table"
    llm_results = df[col].astype(str).tolist()
    
    expanded_dicts = []
    complicated = []
    error = []
    for t in llm_results:
        
        good_table_bool = check_proper_table(t, TABLE_COLS)

        error.append(not good_table_bool)
        if good_table_bool:
            if check_more_complicated(t):
                expanded_dicts.append({})   # no columns added for this row
                complicated.append(True)
            else:
                d = table_to_dict(t)
                b = {f"{model}_{k}": v for k, v in d.items()}
                expanded_dicts.append(b)
                complicated.append(False)
        else:
            expanded_dicts.append({})   # no columns added for this row
            complicated.append(False)

    # Turn list of dicts into a DataFrame (handles ragged keys automatically)
    expanded_df = pd.DataFrame(expanded_dicts, index=df.index).replace("NA", None)
    

    # Optional: ensure consistent column order (key-sorted)
    # expanded_df = expanded_df.reindex(sorted(expanded_df.columns), axis=1)

    # Add a flag column for complicated rows
    expanded_df[f"{model}_more_complicated_flag"] = complicated
    expanded_df[f"{model}_error_flag"] = error

    # Merge back to original
    out = pd.concat([df, expanded_df], axis=1)
    return out


model_names = ["f1", "f2"]
# table_cols = [x+"_table" for x in model_names]

def main():
    df_list = []
    for m in model_names:
        
        m_df = get_full_results_df(m)
        m_df_processed = process_df(m_df, m)

        df_list.append(m_df_processed)
    
    df = inner_join_on_ack_id(df_list)

    agree_rate = check_accuracy(df['f1_table'].astype(str), df['f2_table'].astype(str))

    print(f"LLMs agree on {agree_rate*100:.3f}% of plans ({agree_rate*len(df)} rows)")

    df['models_agree_flag'] = get_boolean_accuracy_col(df['f1_table'].astype(str), df['f2_table'].astype(str))

    snippets_df = create_matching_snippets_df()
    df = df.merge(snippets_df, on='ack_id')

    df['no_snippet_found_flag'] = create_no_mention_of_matching_flag(df, ['g1_snippet', 'g2_snippet'])

    print(df)
    for x in df.columns:
        print(x)

    if df['year_x'].equals(df['year_y']):
        df.drop(columns='year_y', inplace=True)
        df.rename(columns={'year_x': 'year'}, inplace=True)

    df = df.convert_dtypes()

    df.to_stata(SAVE_PATH / 'full_dataset_unfiltered_v2.dta', write_index=False, version=118)
    df2 = df.drop(columns=['g1_snippet', 'g2_snippet'])
    df2.to_stata(SAVE_PATH / 'full_dataset_unfiltered_no_snippets_v2.dta', write_index=False, version=118)

    df_filtered = df.loc[df['models_agree_flag']].reset_index(drop=True)
    
    df_filtered.to_stata(SAVE_PATH / 'full_dataset_filtered_models_agree_v2.dta', write_index=False, version=118)

    df_filtered2 = df_filtered.drop(columns=['g1_snippet', 'g2_snippet'])

    df_filtered2.to_stata(SAVE_PATH / 'full_dataset_filtered_models_agree_no_snippets_v2.dta', write_index=False, version=118)

    return df, df_filtered

if __name__ == "__main__":
    main()