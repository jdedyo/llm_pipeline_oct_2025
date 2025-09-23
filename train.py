import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from datasets import load_dataset, Dataset
from transformers import Trainer, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
import sys
import accelerate
from SETTINGS import *
from training_funcs import *
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Train LLM with a specified model name.")
parser.add_argument(
    "--model",
    type=str,
    required=True,
    choices=registry.choices(),
    help=f"Name of the model to use. Must be one of: {', '.join(registry.choices())}."
)

args = parser.parse_args()

cfg = REGISTRY.get(args.model)
print(f"Using {cfg['name']}: type={cfg['type']}, id={cfg['model_id']}")
print(f"  base_model_path = {cfg['base_model_path']}")
print(f"  save_path       = {cfg['save_path']}")
print(f"  train_epochs    = {cfg['train_epochs']}")

model, tokenizer = get_llm_and_tokenizer(cfg['model_id'], cfg['base_model_path'])

TRAIN_SPLIT = sys.argv[1]
RESUME = False # bool whether to resume training from a checkpoint

if TRAIN_SPLIT not in ["1", "2"]:
    print("First command line argument not 1 or 2, cannot choose a half to train on.\n")
    sys.exit(1)

model_dir = HOME+"/models/llama3point1-inst" # Change this to your path to the base model

if TRAIN_SPLIT == "1":
    save_dir = HOME+"/models/"+ str(NUMEPOCHS) +"_epoch_first_half_ae_snippets_model"+NOW # Change this to the path you want to save the finetuned model to
    train_data_path = '../../train_data/firsthalf_all_ocr_2003-2018_AE_with_tables.xlsx' # Change this to your path to the csv or xlsx with the training data
elif TRAIN_SPLIT == "2":
    save_dir = HOME+"/models/"+ str(NUMEPOCHS) +"_epoch_second_half_ae_snippets_model"+NOW # Change this to the path you want to save the finetuned model to
    train_data_path = '../../train_data/secondhalf_all_ocr_2003-2018_AE_with_tables.xlsx' # Change this to your path to the csv or xlsx with the training data
else:
    print("TRAIN_SPLIT variable not 1 or 2, cannot choose a half to train on.\n")
    sys.exit(1)
    
print("Training on: ", train_data_path)
print("Training for: ", NUMEPOCHS)
print("Saving to: ", save_dir)

prompt_path = '../../prompts/snippet_extract_v8_autoenrollment.txt' # Change this to your path to the prompt file

CHAT_TEMPLATE = True # bool whether to put the prompts into chat template (True for fine-tuning, False for inference)
INCLUDE_ANS = True # bool whether to include the answer in the prompts (True for fine-tuning, False for inference)

FLASH_ATTN = True # bool whether to use flash attention 2 for faster model (True if using certain NVIDIA GPUs)

# Load model
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, cache_dir=model_dir, device_map='auto', attn_implementation="flash_attention_2")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=model_dir)

tokenizer.pad_token = "!" #Not EOS, will explain another time.

model = get_peft_model(model, TRAIN_CONFIG)

print_trainable_parameters(model)

print("dataset", raw_data)




trainer.train(resume_from_checkpoint=RESUME)

training_log = trainer.state.log_history

trainer.save_model(save_dir)

log_hist = pd.DataFrame(trainer.state.log_history)
log_hist.to_csv(save_dir + '/log_file', encoding='utf-8', index=False)
log_hist = log_hist[log_hist['loss'].notna()]

print(log_hist)