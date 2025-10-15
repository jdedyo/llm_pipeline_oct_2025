from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from datasets import load_dataset, Dataset
from transformers import Trainer, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
import accelerate
from SETTINGS import *
import argparse
from training_funcs import *
from llm_helpers import generate_prompts

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train LLM with a specified model name.")
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
    print(f"  base_model_path = {cfg['base_model_path']}")
    print(f"  save_path       = {cfg['save_path']}")
    print(f"  train_epochs    = {cfg['train_epochs']}")
    print(f"  train_data_path = {cfg['train_data_path']}")
    print(f"  prompt          = {cfg['prompt_path']}")
    print(f"  train_ans_col   = {cfg['train_ans_col']}")

    model, tokenizer = get_llm_and_tokenizer(cfg['model_id'], cfg['base_model_path'])

    model = get_peft_model(model, TRAIN_CONFIG)

    print_trainable_parameters(model)

    # train_data = prep_training_data(train_data_path = cfg['train_data_path'], 
                                    # prompt_path = cfg['prompt_path'],
                                    # ans_col = cfg['train_ans_col'])

    train_data = generate_prompts(
        prompt_path=cfg['prompt_path'],
        data_path=cfg['train_data_path'],
        tokenizer=tokenizer,
        correct_ans_col_name=cfg['train_ans_col'],
        data_col_name_for_rag=cfg['train_rag_data_col'],
        rag_corpus_data_path=cfg['train_rag_corpus_data_path'],
        rag_corpus_data_col_name=cfg['train_rag_corpus_data_col'],
        rag_corpus_ans_col_name=cfg['train_rag_corpus_ans_col'],
        rag_corpus_data_year_col_name='year',
        data_year_col_name='year',
        train_bool=True
    )

    print("dataset", train_data)

    training_args = cfg['training_args']
    training_args.num_train_epochs = cfg['train_epochs']
    training_args.output_dir = cfg['save_path'] / 'checkpoints'

    trainer = get_trainer(model=model, 
                        tokenizer=tokenizer,
                        training_args=training_args, 
                        train_data=train_data)

    trainer.train()

    training_log = trainer.state.log_history

    trainer.save_model(cfg['save_path'])
    tokenizer.save_pretrained(cfg['save_path'])

    log_hist = pd.DataFrame(trainer.state.log_history)
    log_hist.to_csv(cfg['save_path'] / 'log_file', encoding='utf-8', index=False)
    log_hist = log_hist[log_hist['loss'].notna()]

    print(log_hist)