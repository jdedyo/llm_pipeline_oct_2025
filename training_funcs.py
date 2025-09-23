import pandas as pd
from datasets import load_dataset, Dataset
from SETTINGS import *
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from llm_helpers import *
from tqdm import tqdm

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

def print_trainable_parameters(m: AutoModelForCausualLM) -> None:
    trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in m.parameters())
    print(f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params}")


def prep_training_data(train_data_path: Union[str, Path], 
                       prompt_path: Union[str, Path], 
                       chat_template_bool: bool=False,
                       include_ans_bool: bool=False,
                       ans_col: str="",
                       ocr_col: str="content") -> Dataset:
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

def get_trainer(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, train_data: Dataset) -> Trainer:
    
    training_args = TRAINING_ARGS
    
    trainer = Trainer(
    model=model,
    train_dataset=train_data,
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    return trainer

def get_base_model()