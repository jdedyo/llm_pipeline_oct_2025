from SETTINGS import *
import pandas as pd
from pathlib import Path
from utils import *
from data_processing import *

def partition_df(df: pd.DataFrame, frac: float = 0.5, random_state: int = 42):
    if df[PARTITION_COL].isna().any():
        raise ValueError(f"NaN values in the partition column: {PARTITION_COL}")
    groups = df[PARTITION_COL].unique()
    g1 = set(pd.Series(groups).sample(frac=frac, random_state=random_state))
    df1 = df[df[PARTITION_COL].isin(g1)].copy()
    df2 = df[~df[PARTITION_COL].isin(g1)].copy()
    return df1, df2

if __name__ == "__main__":
    print(f"Loading {ALL_TRAIN_DATA}...")
    all_train_data = load_data(ALL_TRAIN_DATA)
    print(f"Loaded file!")

    # all_train_data = all_train_data.loc[all_train_data[DATA_AVAILABLE_FLAG_COL].str.strip().eq("Yes")]
    all_train_data[MATCH_FORMULA_COL] = all_train_data[MATCH_FORMULA_COL].apply(normalize_feature_formula)
    all_train_data[CORRECT_MATCHING_TABLE_COL] = generate_correct_table(all_train_data, 'matching')
    all_train_data[CORRECT_MATCHING_SNIPPET_COL] = generate_correct_snippet_col(all_train_data, 'matching')

    # all_train_data[MATCH_FORMULA_COL] = all_train_data[MATCH_FORMULA_COL].apply(normalize_feature_formula)
    all_train_data[CORRECT_AE_TABLE_COL] = generate_correct_table(all_train_data, 'autoenrollment')
    all_train_data[CORRECT_AE_SNIPPET_COL] = generate_correct_snippet_col(all_train_data, 'autoenrollment')

    # all_train_data[MATCH_FORMULA_COL] = all_train_data[MATCH_FORMULA_COL].apply(normalize_feature_formula)
    all_train_data[CORRECT_VESTING_TABLE_COL] = generate_correct_table(all_train_data, 'vesting')
    all_train_data[CORRECT_VESTING_SNIPPET_COL] = generate_correct_snippet_col(all_train_data, 'vesting')

    all_train_data = all_train_data.copy()
    print(len(all_train_data))

    print(f"Loading {ALL_OCR_TEXT_DATA_PATH}...")
    all_ocr = load_ocr_data()
    print(f"Loaded file!")

    print("Merging OCR text onto training data...")
    all_train_data = inner_join_on_ack_id([all_train_data, all_ocr])#.merge(all_ocr, on="ack_id", how="inner")
    print(len(all_train_data))
    print("Merge complete!")

    print(f"Partitioning data")
    partition_1, partition_2 = partition_df(all_train_data)
    print(len(partition_1))
    print(len(partition_2))
    print(f"Data partitioned!")

    print(f"Saving partition 1...")
    save_data(partition_1, PARTITION_1_PATH)
    print(f"Partition 1 saved!")

    print(f"Saving partition 2...")
    save_data(partition_2, PARTITION_2_PATH)
    print(f"Partition 2 saved!")