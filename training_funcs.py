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


# Define this function for use in creating the prompts
def chat_template(prompt: str, tokenizer: AutoTokenizer, answer: str='') -> str:
    if answer:
        chat = [
       {"role": "user", "content": f"{prompt}"},
       {"role": "assistant", "content": f"{answer}"}
        ] 
    else:
        chat = [
       {"role": "user", "content": f"{prompt}"}
        ]
    chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
    return chat

def print_trainable_parameters(m: AutoModelForCausalLM) -> None:
    trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in m.parameters())
    print(f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params}")


def prep_training_data(train_data_path: Union[str, Path], 
                       prompt_path: Union[str, Path],
                       ans_col: str,
                       ocr_col: str=PLAN_TEXT_COL,
                       chat_template_bool: bool=True,
                       include_ans_bool: bool=True) -> Dataset:
    
    train_data_path = Path(train_data_path)
    prompt_path = Path(prompt_path)

    # Load data
    ext = train_data_path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(train_data_path)
    elif ext in {".xlsx", ".xls"}:
        df = pd.read_excel(train_data_path)
    else:
        raise ValueError(f"Unsupported file type: {ext} (expected .csv, .xlsx, or .xls)")

    # Read prompt
    cleanprompt = prompt_path.read_text(encoding="utf-8")
    print(f"Using prompt: {cleanprompt}")

    print("Preparing prompts...")

    processed_prompts = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Building prompts"):
        prompt = (cleanprompt
                .replace("[YYYY]", str(row.get("year", "")))
                .replace("[PLAN]", str(row.get(ocr_col, ""))))

        if chat_template_bool:
            if include_ans_bool:
                prompt = chat_template(prompt, str(row.get(ans_col, "")))
            else:
                prompt = chat_template(prompt)

        processed_prompts.append(prompt)

    assert len(processed_prompts) == len(df), "Length mismatch between prompts and rows."

    n = len(processed_prompts)
    print(f"Training on {n} examples.")
    if n:
        print(f"Example prompt:\n{processed_prompts[-1]}")

    raw = Dataset.from_dict({"messages": processed_prompts})
    train_ds = raw.map(lambda row: tokenize(row["messages"]), remove_columns=["messages"])

    return train_ds

def get_trainer(model: AutoModelForCausalLM, 
                tokenizer: AutoTokenizer, 
                training_args: TrainingArguments, 
                train_data: Dataset) -> Trainer:
    
    trainer = Trainer(
    model=model,
    train_dataset=train_data,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    return trainer
#generate_train_prompts(cfg, tokenizer)
def generate_train_prompts(cfg: ModelRegistry, tokenizer: AutoTokenizer):
    
    prompt_path = Path(cfg['prompt_path'])

    if DEBUG:
        data = load_data(get_most_recent_file(cfg['train_data_path']))
    else:
        data = load_data(cfg['train_data_path'])

    if cfg['train_rag_corpus_data_path'] is not None:
        if DEBUG:
            rag_data = load_data(get_most_recent_file(cfg['train_rag_corpus_data_path']))
        else:
            rag_data = load_data(cfg['train_rag_corpus_data_path'])
    else:
        rag_data = None

    plans = data[PLAN_TEXT_COL]
    years = data['year']
    ans   = data[cfg['train_ans_col']]

    # n_plans = len(plans)

    # # if rag_data is not None and len(rag_data) != n_plans:
    # #     raise ValueError(
    # #         f"Length mismatch: rag_data({len(rag_data)}) must match data({n_plans})."
    # #     )

    cleanprompt = prompt_path.read_text(encoding="utf-8")

    prompts = pd.Series(
        [
            chat_template(cleanprompt.replace("[YYYY]", str(y)).replace("[PLAN]", str(p)), tokenizer, a)
            for p, y, a in zip(plans, years, ans)
        ],
        index=data.index,
    )

    if rag_data is not None:
        print("Loading rag model")
        rag_model = load_rag_model()
        rag_plan_ids = rag_data[PARTITION_COL].tolist()
        rag_snips = rag_data[cfg["train_rag_corpus_data_col"]]
        rag_tables = rag_data[cfg["train_rag_corpus_ans_col"]]
        rag_years = rag_data["year"]
        
        print("Prepping plan data for rag")
        corpus_data = prep_plan_data_for_rag(rag_model, 
                                rag_plan_ids, 
                                rag_snips,
                                rag_tables, 
                                rag_years)
        

        query_plan_ids = data[PARTITION_COL].tolist()
        query_years    = years.tolist()
        query_snippets = data[cfg["train_rag_data_col"]].astype(str).tolist()
        
        rag_prompts = []
        for i, p in tqdm(
                    enumerate(prompts),
                    total=len(prompts),
                    desc="Generating RAG-augmented prompts",
                    dynamic_ncols=True
                ):
                
            query_data = make_query(query_snippets[i], query_plan_ids[i], query_years[i])
            match_snips, match_tables, match_years, match_plan_ids = rag_generator(rag_model, query_data, corpus_data)
            try:
                rag_prompts.append(add_rag_examples(p, match_snips, match_tables, match_years))
            except:
                print(len(match_snips), match_snips)
                print(len(match_tables), match_tables)
                print(len(match_years), match_years)
                print(len(match_plan_ids), match_plan_ids)
        
        del rag_model
        torch.cuda.empty_cache()

        prompts = rag_prompts
        # TODO: COMPLETE RAG PART
        with open("debug_train_prompts.txt", "w", encoding="utf-8") as f:
            f.write("\n\n+++++++++++++++++++++\n\n".join(str(p) for p in prompts[:100]))

    processed_prompts = list(prompts)
    
    n = len(processed_prompts)
    print(f"Training on {n} examples.")
    if n:
        print(f"Example prompt:\n{processed_prompts[-1]}")

    raw = Dataset.from_dict({"messages": processed_prompts})

    def tokenize(prompt):
        x=tokenizer(
            prompt,# + tokenizer.eos_token,
            truncation=True,
            return_special_tokens_mask=True,
            max_length=CUTOFF_LEN,
            # padding=True
        )
        return x

    train_ds = raw.map(lambda row: tokenize(row["messages"]), remove_columns=["messages"])

    return train_ds