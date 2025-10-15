from SETTINGS import *
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import math
import csv

# def count_rows(csv_path):
#     with open(csv_path, newline='', encoding='utf-8') as f:
#         reader = csv.reader(f)
#         header = next(reader, None)  # skip header
#         return sum(1 for _ in reader)

def save_chunk(chunk: pd.DataFrame, chunk_num: int):
    save_name = OOS_DATA_DIR / f"oos_chunk_{chunk_num}.csv"
    chunk.to_csv(save_name, index=False)

if __name__ == "__main__":

    # with open(RAW_OOS_DATA_LOC, "r") as f:
    #     total_lines = sum(1 for _ in f) - 1  # subtract header line

    # n_rows = count_rows(RAW_OOS_DATA_LOC)

    # total_num_chunks = math.ceil(total_lines / OOS_CHUNK_SIZE)

    for i, chunk in tqdm(
        enumerate(pd.read_csv(RAW_OOS_DATA_LOC, chunksize=OOS_CHUNK_SIZE)),
        desc="Splitting OOS CSV into chunks",
        dynamic_ncols=True
    ):
    
        save_chunk(chunk, i)
        print(f" Saved chunk {i} ({len(chunk):,} rows)")