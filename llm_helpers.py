from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModelForCausalLM
from datasets import Dataset
import warnings
from SETTINGS import *
from typing import Tuple
import torch
from utils import *
from rag_funcs import *
from tqdm import tqdm
import numpy as np
import time
import sys

def get_llm_and_tokenizer(model_id: str, model_dir: str, flash_attn_bool: bool=True) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:    
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

def generate_prompts(prompt_path: Path,
                    data_path: Path,
                    tokenizer: AutoTokenizer,
                    correct_ans_col_name: str=None,
                    data_col_name_for_rag: str=None,
                    rag_corpus_data_path: Path=None,
                    rag_corpus_data_col_name: str=None,
                    rag_corpus_ans_col_name: str=None,
                    rag_corpus_data_year_col_name: str='year',
                    data_year_col_name: str='year',
                    id_col_for_rag: str=PARTITION_COL,
                    train_bool: bool=False) -> Dataset:
    
    # if oos_bool and train_bool:
    #     raise AssertionError(f"Cannot have oos_bool={oos_bool} and train_bool={train_bool}. At least one must be false.")

    if DEBUG:
        newest_data = get_most_recent_file(data_path)
    else:
        newest_data = data_path
    
    print(f"Using {newest_data}...")

    data = load_data(newest_data)

    if rag_corpus_data_path is not None:
        if DEBUG:
            newest_rag_data = get_most_recent_file(rag_corpus_data_path)
        else:
            newest_rag_data = rag_corpus_data_path

        print(f"Using {newest_rag_data} for RAG examples...")    
        rag_data = load_data(newest_rag_data)
    else:
        print("Not using RAG...")
        rag_data = None

    plans = data[PLAN_TEXT_COL]
    years = data[data_year_col_name]

    cleanprompt = prompt_path.read_text(encoding="utf-8")

    prompts = pd.Series(
        [
            cleanprompt.replace("[YYYY]", str(y)).replace("[PLAN]", str(p))
            for p, y in zip(plans, years)
        ],
        index=data.index,
    )

    if rag_data is not None:
        print("Loading rag model")
        rag_model = load_rag_model()
        rag_plan_ids = rag_data[id_col_for_rag].tolist()
        rag_snips = rag_data[rag_corpus_data_col_name]
        rag_tables = rag_data[rag_corpus_ans_col_name]
        rag_years = rag_data[rag_corpus_data_year_col_name]
        
        print("Prepping plan data for rag")
        corpus_data = prep_plan_data_for_rag(rag_model, 
                                rag_plan_ids, 
                                rag_snips,
                                rag_tables, 
                                rag_years)
        

        if id_col_for_rag in data.columns:
            query_plan_ids = data[id_col_for_rag].tolist()
        else:
            warnings.warn(f"Column '{id_col_for_rag}' not found in data; filling with NaN for RAG filtering.", stacklevel=1)
            query_plan_ids = [np.nan] * len(data)

        query_years    = years.tolist()
        query_snippets = data[data_col_name_for_rag].astype(str).tolist()
        
        rag_prompts = []
        for i, p in tqdm(
                    enumerate(prompts),
                    total=len(prompts),
                    desc="Generating RAG-augmented prompts",
                    dynamic_ncols=True
                ):
                
            try: # TODO: remove try-except after debugging.
                query_data = make_query(query_snippets[i], query_plan_ids[i], query_years[i])
                match_snips, match_tables, match_years, match_plan_ids = rag_generator(rag_model, query_data, corpus_data)
                rag_prompts.append(add_rag_examples(p, match_snips, match_tables, match_years))
            except:
                print("p: ", p)
                print("match_snips: ", match_snips)
                print("match_tables: ", match_tables)
                print("match_years: ", match_years)
                print("match_plan_ids: ", match_plan_ids)
                raise
        del rag_model
        torch.cuda.empty_cache()

        prompts = rag_prompts
        # TODO: COMPLETE RAG PART
        with open("debug_generate_prompts.txt", "w", encoding="utf-8") as f:
            f.write("\n\n+++++++++++++++++++++\n\n".join(str(p) for p in prompts[:100]))

    if train_bool:
        ans = data[correct_ans_col_name]
        prompts = [
                chat_template(p, tokenizer, a, train_bool)
                for p, y, a in zip(prompts, years, ans)
            ]
    else:
        prompts = [
                chat_template(p, tokenizer)
                for p, y in zip(prompts, years)
            ]

    processed_prompts = list(prompts)
    
    n = len(processed_prompts)
    print(f"Generated {n} prompts.")
    if n:
        print(f"Example prompt:\n{processed_prompts[-1]}")

    if train_bool:
        raw = Dataset.from_dict({"messages": processed_prompts})

        train_ds = raw.map(lambda row: tokenize(tokenizer, row["messages"]), remove_columns=["messages"])
        
        return train_ds
    
    res = [{"prompt": x} for x in processed_prompts]

    prompts_ds = Dataset.from_list(res)

    return prompts_ds

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
        save_path = max(parsed, key=lambda x: x[1])[0]
        
    else:
        save_path = Path(cfg['save_path'])

    model, tokenizer = get_llm_and_tokenizer(cfg['model_id'], cfg['base_model_path'])

    model = PeftModelForCausalLM.from_pretrained(model, save_path)
    model = model.merge_and_unload()

    return model, tokenizer

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