from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
from SETTINGS import *
from typing import Tuple
import torch

def get_llm_and_tokenizer(model_id: str, model_dir: str flash_attn_bool: bool=True) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:    
    if flash_attn_bool:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, cache_dir=model_dir, device_map='auto', attn_implementation="flash_attention_2")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, cache_dir=model_dir, device_map='auto')

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=model_dir)

    tokenizer.pad_token = "!" #Not EOS, will explain another time.

    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def tokenize(tokenizer: AutoTokenizer, prompt: str):
    x=tokenizer(
        prompt,# + tokenizer.eos_token,
        truncation=True,
        return_special_tokens_mask=True,
        max_length=CUTOFF_LEN,
        # padding=True
    )

    if len(x["input_ids"]) >= CUTOFF_LEN:
        warnings.warn(
            f"Prompt was truncated to {CUTOFF_LEN} tokens!",
            category=UserWarning,
            stacklevel=2
        )

    return x