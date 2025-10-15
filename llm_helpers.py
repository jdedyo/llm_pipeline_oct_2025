from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import warnings
from SETTINGS import *
from typing import Tuple
import torch
from utils import *
from rag_funcs import *
from tqdm import tqdm

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
                    train_bool: bool=False) -> Dataset:
    
    # prompt_path = Path(cfg['prompt_path'])

    if DEBUG:
        data = load_data(get_most_recent_file(data_path))
    else:
        data = load_data(data_path)

    if rag_corpus_data_path is not None:
        if DEBUG:
            rag_data = load_data(get_most_recent_file(rag_corpus_data_path))
        else:
            rag_data = load_data(rag_corpus_data_path)
    else:
        rag_data = None

    plans = data[PLAN_TEXT_COL]
    years = data[data_year_col_name]
    ans   = data[correct_ans_col_name]

    # n_plans = len(plans)

    # # if rag_data is not None and len(rag_data) != n_plans:
    # #     raise ValueError(
    # #         f"Length mismatch: rag_data({len(rag_data)}) must match data({n_plans})."
    # #     )

    cleanprompt = prompt_path.read_text(encoding="utf-8")

    prompts = pd.Series(
        [
            chat_template(cleanprompt.replace("[YYYY]", str(y)).replace("[PLAN]", str(p)), tokenizer, a, train_bool)
            for p, y, a in zip(plans, years, ans)
        ],
        index=data.index,
    )

    if rag_data is not None:
        print("Loading rag model")
        rag_model = load_rag_model()
        rag_plan_ids = rag_data[PARTITION_COL].tolist()
        rag_snips = rag_data[rag_corpus_data_col_name]
        rag_tables = rag_data[rag_corpus_ans_col_name]
        rag_years = rag_data[rag_corpus_data_year_col_name]
        
        print("Prepping plan data for rag")
        corpus_data = prep_plan_data_for_rag(rag_model, 
                                rag_plan_ids, 
                                rag_snips,
                                rag_tables, 
                                rag_years)
        

        query_plan_ids = data[PARTITION_COL].tolist()
        query_years    = years.tolist()
        query_snippets = data[data_col_name_for_rag].astype(str).tolist()
        
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
        with open("debug_generate_prompts.txt", "w", encoding="utf-8") as f:
            f.write("\n\n+++++++++++++++++++++\n\n".join(str(p) for p in prompts[:100]))

    processed_prompts = list(prompts)
    
    n = len(processed_prompts)
    print(f"Generated {n} prompts.")
    if n:
        print(f"Example prompt:\n{processed_prompts[-1]}")

    if train_bool:
        raw = Dataset.from_dict({"messages": processed_prompts})

        # def tokenize(prompt):
        #     x=tokenizer(
        #         prompt,# + tokenizer.eos_token,
        #         truncation=True,
        #         return_special_tokens_mask=True,
        #         max_length=CUTOFF_LEN,
        #         # padding=True
        #     )
        #     return x

        train_ds = raw.map(lambda row: tokenize(tokenizer, row["messages"]), remove_columns=["messages"])
        
        return train_ds
    
    res = [{"prompt": x} for x in processed_prompts]

    prompts_ds = Dataset.from_list(res)

    return prompts_ds