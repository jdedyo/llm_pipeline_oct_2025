import pandas as pd
from datasets import load_dataset, Dataset
from SETTINGS import *
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from llm_helpers import *
from tqdm import tqdm
from rag_funcs import *
from typing import Union
from utils import *
import torch
from typing import List
from peft import PeftModelForCausalLM
import time
import sys


# Define this function for use in creating the prompts
def chat_template(prompt: str, tokenizer: AutoTokenizer, answer: str='', apply_template: bool=False) -> str:
    if answer:
        chat = [
       {"role": "user", "content": f"{prompt}"},
       {"role": "assistant", "content": f"{answer}"}
        ] 
    else:
        chat = [
       {"role": "user", "content": f"{prompt}"}
        ]

    if apply_template:
        chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
    return chat

def generate_test_prompts(cfg: ModelRegistry, tokenizer: AutoTokenizer):
    
    prompt_path = Path(cfg['prompt_path'])
    data = load_data(cfg['train_data_path'])

    if cfg['train_rag_data_path'] is not None:
        rag_data = load_data(cfg['train_rag_data_path'])
    else:
        rag_data = None

    plans = data[PLAN_TEXT_COL]
    years = data['year']
    # ans   = data[cfg['train_ans_col']]

    n_plans = len(plans)

    if rag_data is not None and len(rag_data) != n_plans:
        raise ValueError(
            f"Length mismatch: rag_data({len(rag_data)}) must match data({n_plans})."
        )

    cleanprompt = prompt_path.read_text(encoding="utf-8")

    prompts = pd.Series(
        [
            chat_template(cleanprompt.replace("[YYYY]", str(y)).replace("[PLAN]", str(p)), tokenizer)
            for p, y in zip(plans, years)
        ],
        index=data.index,
    )

    if rag_data is not None:
        
        rag_model = load_rag_model()
        rag_plan_ids = rag_data[PARTITION_COL].tolist()
        rag_snips = rag_data[cfg["train_rag_data_col"]]
        rag_tables = rag_data[cfg["train_ans_col"]]
        rag_years = rag_data["year"]
        
        corpus_data = prep_plan_data_for_rag(rag_model, 
                                rag_plan_ids, 
                                rag_snips,
                                rag_tables, 
                                rag_years)
        
        for p in prompts:
            query_data = make_query(str(snippet), query_docs[i], query_plan_ids[i], query_years[i])
        
        del rag_model
        torch.cuda.empty_cache()
        # TODO: COMPLETE RAG PART

    processed_prompts = prompts.tolist()
    res = [{"prompt": x} for x in processed_prompts]

    test_data = Dataset.from_list(res)

    return test_data


def run_batched_inference(dataset, pipe, batch_size: int=BATCH_SIZE):
    all_snippets = []
    # Process the dataset in batches
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size), desc="Running inference"):
            batch = dataset[i:i+batch_size]["prompt"]  # Get a batch of prompts

            start = time.time()

            results = pipe(batch)  # Perform inference using the pipeline

            inference_time = time.time() - start

            # Print the results for each item in the batch
            for j, result in enumerate(results):
                # print(f"\nPrompt: {batch[j]}")
                print(f"Generated Text: {result[0]['generated_text'][-1]['content'].strip()}\n")    
                sys.stdout.flush()

            print(f'This took {inference_time} seconds\n')

            snippets = [x[0]['generated_text'][-1]['content'].strip() for x in results]

            all_snippets.extend(snippets)
#             print(all_snippets)
    return all_snippets

def save_test_results(df: pd.DataFrame, results: List[str], cfg: ModelRegistry):
    save_dir = cfg['test_data_save_path']

    if DEBUG:
        p = save_dir.with_name(f"{save_dir.name}_{NOW}")
    else:
        p = save_dir

    p.mkdir(parents=True, exist_ok=True)

    col = cgf['result_col']
    
    df2 = df.copy()

    df2[col] = results

    save_data(df2, cfg['test_data_save_path'])

    return df2

def get_most_recent_model_and_tokenizer(cfg: ModelRegistry):
    """
    If DEBUG=True, look for the most recent timestamped variant of cfg['save_path'].
    Otherwise, just return cfg['save_path'].
    """
    loc = Path(cfg["save_path"])
    if DEBUG:
        # remove_timestamp should strip suffix like '_MM_DD_YYYY-HH_AM/PM'
        loc = remove_timestamp(loc)

    base_name = loc.name
    parent = loc.parent

    # Collect candidates that start with base_name + '_'
    candidates = list(parent.glob(base_name + "_*"))

    if not candidates:
        raise FileNotFoundError(
            f"No timestamped copies matching '{base_name}_*' in {parent}"
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

    model, tokenizer = get_llm_and_tokenizer(cfg['model_id'], cfg['base_model_path'])

    model = PeftModelForCausalLM.from_pretrained(model, most_recent)
    model = model.merge_and_unload()

    return model, tokenizer




