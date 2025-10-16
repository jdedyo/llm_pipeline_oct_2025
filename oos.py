from SETTINGS import *
import re
import argparse
from utils import *
from llm_helpers import *
from transformers import pipeline

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run LLM with a specified model name on specified OOS chunk.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=REGISTRY.choices(),
        help=f"Name of the model to use. Must be one of: {', '.join(REGISTRY.choices())}."
    )
    parser.add_argument(
        "--chunk",
        type=int,
        required=True,
        choices=get_chunk_num_range(),
        help=f"Name of the oos chunk to inference on. Must be one of: {', '.join([str(x) for x in get_chunk_num_range()])}."
    )

    args = parser.parse_args()

    cfg = REGISTRY.get(args.model)
    print(f"Using {cfg['name']}:   type={cfg['type']}, id={cfg['model_id']}")
    print(f"  base_model_path          = {cfg['base_model_path']}")
    print(f"  save_path                = {cfg['save_path']}")
    print(f"  prompt                   = {cfg['prompt_path']}")
    print(f"  result_col               = {cfg['result_col']}")
    print(f"  oos_start_dir            = {cfg['oos_start_dir']}")
    print(f"  oos_results_dir          = {cfg['oos_results_dir']}")
    print(f"  oos_rag_corpus_data_col  = {cfg['oos_rag_corpus_data_col']}")
    print(f"  oos_rag_corpus_data_path = {cfg['oos_rag_corpus_data_path']}")

    oos_chunk_path = get_oos_chunk_path_from_num(args.chunk, cfg['oos_start_dir'])
    print(f"Inferencing on OOS chunk {args.chunk} ({oos_chunk_path})...")

    model, tokenizer = get_most_recent_model_and_tokenizer(cfg)

    oos_prompts = generate_prompts( # TODO: make this reflect OOS stuff
        prompt_path=cfg['prompt_path'],
        data_path=oos_chunk_path,
        tokenizer=tokenizer,
        # correct_ans_col_name=cfg['train_ans_col'],
        data_col_name_for_rag=cfg['oos_rag_data_col'],
        rag_corpus_data_path=cfg['oos_rag_corpus_data_path'],
        rag_corpus_data_col_name=cfg['oos_rag_corpus_data_col'],
        rag_corpus_ans_col_name=cfg['oos_rag_corpus_ans_col'],
        rag_corpus_data_year_col_name='year',
        data_year_col_name='year'
    )

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, temperature=0.01, max_new_tokens=1000)
    llm_output = run_batched_inference(oos_prompts, pipe)

    save_oos_chunk_results(args.chunk, llm_output, cfg)