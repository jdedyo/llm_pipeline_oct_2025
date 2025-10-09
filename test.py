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
    print(f"  base_model_path     = {cfg['base_model_path']}")
    print(f"  save_path           = {cfg['save_path']}")
    print(f"  test_data           = {cfg['test_data_path']}")
    print(f"  test_data_save_path = {cfg['test_data_save_path']}")
    print(f"  prompt              = {cfg['prompt_path']}")
    print(f"  result_col          = {cfg['result_col']}")

    model, tokenizer = get_most_recent_model_and_tokenizer(cfg)

    test_prompts = generate_test_prompts(cfg, tokenizer)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, temperature=0.01, max_new_tokens=1000)
    llm_output = run_batched_inference(test_prompts, pipe)

    all_test_data = load_data(cfg['train_data_path'])

    save_test_results(all_test_data, llm_output, cfg)