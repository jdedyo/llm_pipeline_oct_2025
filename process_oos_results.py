from SETTINGS import *
from pathlib import Path
import pandas as pd
import argparse
from utils import *
from data_processing import *

# TABLE_COLS = ['vesting_offered'] + [f'vesting_year{i}' for i in range(NUM_VESTING_YEARS)]

# SAVE_PATH = OOS_RESULTS_DIR / 'datasets'
# SAVE_PATH.mkdir(parents=True, exist_ok=True)

def process_df(df: pd.DataFrame, model: str, result_col: str=None):
    col = result_col or f"{model}_table"
    llm_results = df[col].astype(str).tolist()
    
    expanded_dicts = []
    error = []
    for t in llm_results:
        
        good_table_bool = check_proper_vesting_table(t, TABLE_COLS)

        error.append(not good_table_bool)
        if good_table_bool:
            d = vesting_table_to_dict(t)
            b = {f"{model}_{k}": v for k, v in d.items()}
            expanded_dicts.append(b)
        else:
            expanded_dicts.append({})   # no columns added for this row

    # Turn list of dicts into a DataFrame (handles ragged keys automatically)
    expanded_df = pd.DataFrame(expanded_dicts, index=df.index).replace("NA", None)
    

    # Optional: ensure consistent column order (key-sorted)
    # expanded_df = expanded_df.reindex(sorted(expanded_df.columns), axis=1)

    # Add a flag column for complicated rows
    expanded_df[f"{model}_error_flag"] = error

    # Merge back to original
    out = pd.concat([df, expanded_df], axis=1)
    return out


# model_names = ["f1_VESTING", "f2_VESTING"]
# table_cols = [x+"_table" for x in model_names]

# def main(feature_type: str):
#     df_list = []
#     for m in model_names:
        
#         m_df = get_full_results_df(m)
#         m_df_processed = process_df(m_df, m)

#         df_list.append(m_df_processed)
    
#     df = inner_join_on_ack_id(df_list)

#     agree_rate = check_vesting_accuracy(df['f1_VESTING_table'].astype(str), df['f2_VESTING_table'].astype(str))

#     print(f"LLMs agree on {agree_rate*100:.3f}% of plans ({agree_rate*len(df)} rows)")

#     df['VESTING_models_agree_flag'] = get_boolean_vesting_accuracy_col(df['f1_VESTING_table'].astype(str), df['f2_VESTING_table'].astype(str))

#     snippets_df = create_snippets_df(['g1_VESTING', 'g2_VESTING'])
#     df = df.merge(snippets_df, on='ack_id')

#     df['no_mention_of_vesting_flag'] = create_no_mention_of_vesting_flag(df, ['g1_VESTING_snippet', 'g2_VESTING_snippet'])

#     print(df)
#     for x in df.columns:
#         print(x)

#     if df['year_x'].equals(df['year_y']):
#         df.drop(columns='year_y', inplace=True)
#         df.rename(columns={'year_x': 'year'}, inplace=True)

#     df = df.convert_dtypes()

#     df.to_stata(SAVE_PATH / 'vesting_full_dataset_unfiltered_v1.dta', write_index=False, version=118)
#     df2 = df.drop(columns=['g1_VESTING_snippet', 'g2_VESTING_snippet'])
#     df2.to_stata(SAVE_PATH / 'vesting_full_dataset_unfiltered_no_snippets_v1.dta', write_index=False, version=118)

#     df_filtered = df.loc[df['AE_models_agree_flag']].reset_index(drop=True)
    
#     df_filtered.to_stata(SAVE_PATH / 'vesting_full_dataset_filtered_models_agree_v1.dta', write_index=False, version=118)

#     df_filtered2 = df_filtered.drop(columns=['g1_VESTING_snippet', 'g2_VESTING_snippet'])

#     df_filtered2.to_stata(SAVE_PATH / 'vesting_full_dataset_filtered_models_agree_no_snippets_v1.dta', write_index=False, version=118)

#     return df, df_filtered

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Choose a plan feature type for which to process the results.")
    parser.add_argument(
        "--feature_type",
        type=str,
        required=True,
        choices=['matching', 'autoenrollment', 'vesting'],
        help=f"Name of the model to use. Must be one of: {', '.join(['matching', 'autoenrollment', 'vesting'])}."
    )

    args = parser.parse_args()

    post_process_feature_results(args.feature_type)