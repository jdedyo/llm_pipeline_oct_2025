import numpy as np
import pandas as pd
import time
import torch
import os
import sys
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import subprocess
import traceback
from tqdm import tqdm
from datasets import Dataset
from datetime import datetime
from testing_funcs import *
from SETTINGS import *
import argparse
from llm_helpers import generate_prompts
from utils import *

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Test LLM with a specified model name.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=REGISTRY.choices(),
        help=f"Name of the model to use. Must be one of: {', '.join(REGISTRY.choices())}."
    )

    args = parser.parse_args()

    cfg = REGISTRY.get(args.model)
    print(f"Using {cfg['name']}: type={cfg['type']}, id={cfg['model_id']}")
    print(f"  base_model_path        = {cfg['base_model_path']}")
    print(f"  save_path              = {cfg['save_path']}")
    print(f"  test_data              = {cfg['test_data_path']}")
    print(f"  test_correct_ans_col   = {cfg['test_correct_ans_col']}")
    print(f"  test_data_save_path    = {cfg['test_data_save_path']}")
    print(f"  prompt                 = {cfg['prompt_path']}")
    print(f"  result_col             = {cfg['result_col']}")

    model, tokenizer = get_most_recent_model_and_tokenizer(cfg)

    test_prompts = generate_prompts( # TODO: make this reflect test stuff
        prompt_path=cfg['prompt_path'],
        data_path=cfg['test_data_path'],
        tokenizer=tokenizer,
        # correct_ans_col_name=cfg['train_ans_col'],
        data_col_name_for_rag=cfg['test_rag_data_col'],
        rag_corpus_data_path=cfg['test_rag_corpus_data_path'],
        rag_corpus_data_col_name=cfg['test_rag_corpus_data_col'],
        rag_corpus_ans_col_name=cfg['test_rag_corpus_ans_col'],
        rag_corpus_data_year_col_name='year',
        data_year_col_name='year',
        id_col_for_rag=PARTITION_COL
    )

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, temperature=0.01, max_new_tokens=1000)
    llm_output = run_batched_inference(test_prompts, pipe)

    all_test_data = load_data(cfg['test_data_path'])

    save_test_results(all_test_data, llm_output, cfg)

    if cfg['test_check_accuracy'] and cfg['test_correct_ans_col']:
        correct_ans = all_test_data[cfg['test_correct_ans_col']]
        accuracy = check_accuracy(llm_output, correct_ans)
        print(f"The model is {accuracy:.3f}% accurate on its test set.")