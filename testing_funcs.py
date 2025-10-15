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
    if DEBUG:
        data = load_data(get_most_recent_file(cfg['test_data_path']))
    else:
        data = load_data(cfg['test_data_path'])

    if cfg['test_rag_corpus_data_path'] is not None:
        if DEBUG:
            rag_data = load_data(get_most_recent_file(cfg['test_rag_corpus_data_path']))
        else:
            rag_data = load_data(cfg['test_rag_corpus_data_path'])
    else:
        rag_data = None

    plans = data[PLAN_TEXT_COL]
    years = data['year']

    cleanprompt = prompt_path.read_text(encoding="utf-8")

    prompts = pd.Series(
        [
            chat_template(cleanprompt.replace("[YYYY]", str(y)).replace("[PLAN]", str(p)), tokenizer)
            for p, y in zip(plans, years)
        ],
        index=data.index,
    )

    if rag_data is not None:
        print("Loading rag model")
        rag_model = load_rag_model()
        rag_plan_ids = rag_data[PARTITION_COL].tolist()
        rag_snips = rag_data[cfg["test_rag_corpus_data_col"]]
        rag_tables = rag_data[cfg["test_rag_corpus_ans_col"]]
        rag_years = rag_data["year"]
        
        print("Prepping plan data for rag")
        corpus_data = prep_plan_data_for_rag(rag_model, 
                                rag_plan_ids, 
                                rag_snips,
                                rag_tables, 
                                rag_years)
        

        query_plan_ids = data[PARTITION_COL].tolist()
        query_years    = years.tolist()
        query_snippets = data[cfg["test_rag_data_col"]].astype(str).tolist()
        
        rag_prompts = []
        for i, p in tqdm(
                    enumerate(prompts),
                    total=len(prompts),
                    desc="Generating RAG-augmented prompts",
                    dynamic_ncols=True
                ):
                
            query_data = make_query(query_snippets[i], query_plan_ids[i], query_years[i])
            match_snips, match_tables, match_years, match_plan_ids = rag_generator(rag_model, query_data, corpus_data)
            rag_prompts.append(add_rag_examples(p, match_snips, match_tables, match_years))
        
        del rag_model
        torch.cuda.empty_cache()

        prompts = rag_prompts
        # TODO: COMPLETE RAG PART
        with open("debug_test_prompts.txt", "w", encoding="utf-8") as f:
            f.write("\n\n+++++++++++++++++++++\n\n".join(str(p) for p in prompts[:100]))

    processed_prompts = list(prompts)
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
    # save_name = cfg['test_data_save_path']

    # # if DEBUG:
    # #     p = save_dir.with_name(f"{save_dir.name}_{NOW}")
    # # else:
    # #     p = save_dir

    # p=save_name

    # # p.mkdir(parents=True, exist_ok=True)

    col = cfg['result_col']
    
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




