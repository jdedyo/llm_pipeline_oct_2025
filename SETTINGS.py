from pathlib import Path
from datetime import datetime
from peft import LoraConfig
from transformers import TrainingArguments
from models_registry import ModelRegistry
from copy import deepcopy

DEBUG = True

HOME = Path.home()

PROMPTS_DIR = Path("./prompts")
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

RAW_OOS_DATA_LOC = Path("/nfs/roberts/project/pi_co337/jmd324/text_extraction_pipeline_sep_2025/results/all_ocr_oos.csv")
OOS_CHUNK_SIZE = 30_000

OOS_DIR = Path("./oos")

OOS_DATA_DIR = OOS_DIR / "oos_data"
OOS_DATA_DIR.mkdir(parents=True, exist_ok=True)

OOS_DATA_WITH_SNIPPETS_DIR = OOS_DIR / "oos_data_with_snippets"
OOS_DATA_WITH_SNIPPETS_DIR.mkdir(parents=True, exist_ok=True)

OOS_RESULTS_DIR = OOS_DIR / "oos_results"
OOS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS_LOC = Path("/home/jmd324/scratch_pi_co337/jmd324") / "models"
MODELS_LOC.mkdir(parents=True, exist_ok=True)

TABULATOR_BASE_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct" # Model id from the huggingface website
SNIPPET_BASE_MODEL_ID = TABULATOR_BASE_MODEL_ID
RAG_MODEL_ID = 'sentence-transformers/all-MiniLM-L12-v2'

TABULATOR_BASE_MODEL_DIR = MODELS_LOC / "llama_3p1_8b_inst"
SNIPPET_BASE_MODEL_DIR = TABULATOR_BASE_MODEL_DIR
RAG_MODEL_DIR = MODELS_LOC / "MiniLM-L12-v2"

NUM_RAG_EXAMPLES = 5

TRAIN_DATA = Path("./train_data")

ALL_TRAIN_DATA = TRAIN_DATA / "sample_7500_v22Mar24.dta"

PARTITION_1_PATH = TRAIN_DATA / "partition_1.csv"
PARTITION_2_PATH = TRAIN_DATA / "partition_2.csv"

SMALL_SAMPLE_PATH = TRAIN_DATA / "small_sample.csv"

PARTITION_COL = "plan_id"

MATCH_RATE_1_COL = "match_rate_1"
MATCH_RATE_2_COL = "match_rate_2"
MATCH_RATE_3_COL = "match_rate_3"
CAP_1_COL = "cap_1"
CAP_2_COL = "cap_2"
CAP_3_COL = "cap_3"

RAW_MATCHING_SNIPPET_COL = "employer_matching_text"

MATCH_FORMULA_COL = "match_formula"

DATA_AVAILABLE_FLAG_COL = 'data_availability'

CONVERT_MARGINAL_BOOL = True
CORRECT_MATCHING_TABLE_COL = "correct_matching_table"
CORRECT_MATCHING_SNIPPET_COL = "correct_matching_snippet"

TABULATOR_ANSWER_COL = CORRECT_MATCHING_TABLE_COL
SNIPPET_ANSWER_COL = CORRECT_MATCHING_SNIPPET_COL

TABULATOR_PROMPT_PATH = PROMPTS_DIR / "one_shot_RAG_nodistinction_prompt_v1.txt"
SNIPPET_PROMPT_PATH = PROMPTS_DIR / "snippet_extract_v8_contributions_matching_only.txt"

NUM_TABULATOR_TRAIN_EPOCHS = 4
NUM_SNIPPET_TRAIN_EPOCHS = 3

NOW = datetime.now().strftime("%m_%d_%Y-%I_%p") # suffix for filenames so that we know which are newest

DEFAULT_TRAINING_ARGS = TrainingArguments( # Automatically set num_train_epochs and save_dir based on registry
        per_device_train_batch_size=4,
        auto_find_batch_size=True,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        bf16=True,
        logging_steps=10,
        optim="adamw_torch",
        save_strategy="epoch",
    )

TABULATOR_TRAINING_ARGS = deepcopy(DEFAULT_TRAINING_ARGS)
TABULATOR_TRAINING_ARGS.num_train_epochs = NUM_TABULATOR_TRAIN_EPOCHS

SNIPPET_TRAINING_ARGS = deepcopy(DEFAULT_TRAINING_ARGS)
SNIPPET_TRAINING_ARGS.num_train_epochs = NUM_SNIPPET_TRAIN_EPOCHS

REGISTRY = ModelRegistry()
REGISTRY.add(
    "f1",
    type="TABULATOR",
    model_id=TABULATOR_BASE_MODEL_ID,
    save_path=MODELS_LOC / "f1",
    base_model_path=TABULATOR_BASE_MODEL_DIR,
    prompt_path=TABULATOR_PROMPT_PATH,

    train_data_path=PARTITION_1_PATH,
    train_epochs=NUM_TABULATOR_TRAIN_EPOCHS,
    train_ans_col=TABULATOR_ANSWER_COL,
    training_args=DEFAULT_TRAINING_ARGS,

    train_rag_corpus_data_path=PARTITION_1_PATH,
    train_rag_corpus_data_col="g2_snippet",
    train_rag_corpus_ans_col=TABULATOR_ANSWER_COL,
    # train_rag_data_path=PARTITION_1_PATH,
    train_rag_data_col="g2_snippet",

    test_data_path=PARTITION_2_PATH,
    test_data_save_path=PARTITION_2_PATH,
    result_col="f1_table",

    test_rag_corpus_data_path=PARTITION_2_PATH,
    test_rag_corpus_data_col="g1_snippet",
    test_rag_corpus_ans_col=TABULATOR_ANSWER_COL,

    oos_start_dir=OOS_DATA_WITH_SNIPPETS_DIR / "g1",
    oos_results_dir=OOS_RESULTS_DIR / "f1",

    oos_rag_corpus_data_col="g1_snippet",
    oos_rag_corpus_data_path=PARTITION_2_PATH,
    oos_rag_data_col="g1_snippet",
    oos_rag_corpus_ans_col=TABULATOR_ANSWER_COL
)
REGISTRY.add(
    "f2",
    type="TABULATOR",
    model_id=TABULATOR_BASE_MODEL_ID,
    save_path=MODELS_LOC / "f2",
    base_model_path=TABULATOR_BASE_MODEL_DIR,
    prompt_path=TABULATOR_PROMPT_PATH,

    train_data_path=PARTITION_2_PATH,
    train_epochs=NUM_TABULATOR_TRAIN_EPOCHS,
    train_ans_col=TABULATOR_ANSWER_COL,
    training_args=DEFAULT_TRAINING_ARGS,

    train_rag_corpus_data_path=PARTITION_2_PATH,
    train_rag_corpus_data_col="g1_snippet",
    train_rag_corpus_ans_col=TABULATOR_ANSWER_COL,
    # train_rag_data_path=PARTITION_2_PATH,
    train_rag_data_col="g1_snippet",
    
    test_data_path=PARTITION_1_PATH,
    test_data_save_path=PARTITION_1_PATH,
    result_col="f2_table",

    test_rag_corpus_data_path=PARTITION_1_PATH,
    test_rag_corpus_data_col="g2_snippet",
    test_rag_corpus_ans_col=TABULATOR_ANSWER_COL,
    
    oos_start_dir=OOS_DATA_WITH_SNIPPETS_DIR / "g2",
    oos_results_dir=OOS_RESULTS_DIR / "f2",

    oos_rag_corpus_data_col="g2_snippet",
    oos_rag_corpus_data_path=PARTITION_1_PATH,
    oos_rag_data_col="g2_snippet",
    oos_rag_corpus_ans_col=TABULATOR_ANSWER_COL
)
REGISTRY.add(
    "g1",
    type="SNIPPET",
    model_id=SNIPPET_BASE_MODEL_ID,
    save_path=MODELS_LOC / "g1",
    base_model_path=SNIPPET_BASE_MODEL_DIR,
    prompt_path=SNIPPET_PROMPT_PATH,

    train_data_path=PARTITION_1_PATH,
    train_epochs=NUM_SNIPPET_TRAIN_EPOCHS,
    train_ans_col=SNIPPET_ANSWER_COL,
    training_args=DEFAULT_TRAINING_ARGS,
    
    test_data_path=PARTITION_2_PATH,
    test_data_save_path=PARTITION_2_PATH,
    result_col="g1_snippet",

    oos_start_dir=OOS_DATA_DIR,
    oos_results_dir=OOS_DATA_WITH_SNIPPETS_DIR / "g1"
)
REGISTRY.add(
    "g2",
    type="SNIPPET",
    model_id=SNIPPET_BASE_MODEL_ID,
    save_path=MODELS_LOC / "g2",
    base_model_path=SNIPPET_BASE_MODEL_DIR,
    prompt_path=SNIPPET_PROMPT_PATH,

    train_data_path=PARTITION_2_PATH,
    train_epochs=NUM_SNIPPET_TRAIN_EPOCHS,
    train_ans_col=SNIPPET_ANSWER_COL,
    training_args=DEFAULT_TRAINING_ARGS,
    
    test_data_path=PARTITION_1_PATH,
    test_data_save_path=PARTITION_1_PATH,

    result_col="g2_snippet",

    oos_start_dir=OOS_DATA_DIR,
    oos_results_dir=OOS_DATA_WITH_SNIPPETS_DIR / "g2"
)
REGISTRY.add( # TODO: remove when done debugging RAG stuff
    "small_snippet_extractor",
    type="SNIPPET",
    model_id=SNIPPET_BASE_MODEL_ID,
    save_path=MODELS_LOC / "g2",
    base_model_path=SNIPPET_BASE_MODEL_DIR,
    train_data_path=PARTITION_2_PATH,
    prompt_path=SNIPPET_PROMPT_PATH,
    test_data_path=SMALL_SAMPLE_PATH,
    test_data_save_path=SMALL_SAMPLE_PATH,
    train_epochs=NUM_SNIPPET_TRAIN_EPOCHS,
    train_ans_col=SNIPPET_ANSWER_COL,
    training_args=DEFAULT_TRAINING_ARGS,
    result_col="g2_snippet",

    oos_start_dir=OOS_DATA_DIR,
    oos_results_dir=OOS_DATA_WITH_SNIPPETS_DIR / "g2"

)
REGISTRY.add( # TODO: remove after debugging rag
    "small_tabulator",
    type="TABULATOR",
    model_id=TABULATOR_BASE_MODEL_ID,
    save_path=MODELS_LOC / "f1",
    base_model_path=TABULATOR_BASE_MODEL_DIR,
    train_data_path=SMALL_SAMPLE_PATH,
    train_rag_corpus_data_path=SMALL_SAMPLE_PATH,
    train_rag_corpus_data_col="g2_snippet",
    train_rag_corpus_ans_col=TABULATOR_ANSWER_COL,
    prompt_path=TABULATOR_PROMPT_PATH,
    test_data_path=PARTITION_2_PATH,
    test_data_save_path=PARTITION_2_PATH,
    train_epochs=NUM_TABULATOR_TRAIN_EPOCHS,
    train_ans_col=TABULATOR_ANSWER_COL,
    training_args=DEFAULT_TRAINING_ARGS,
    result_col="f1_table",
    # train_rag_data_path=PARTITION_1_PATH,
    train_rag_data_col="g2_snippet",
    oos_rag_corpus_data_col="g1_snippet",
    oos_rag_corpus_data_path=PARTITION_2_PATH,

    oos_start_dir=OOS_DATA_DIR,
    oos_results_dir=OOS_DATA_WITH_SNIPPETS_DIR / "g2"
)


# optional debug suffix
if DEBUG:
    REGISTRY.append_suffix_to_model_save_paths(NOW)
    REGISTRY.append_suffix_to_test_save_paths(NOW)

TRAIN_CONFIG = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules = ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj"]
)

CUTOFF_LEN = 8000  #Our dataset has long text

NUM_TRAIN_PARTITIONS = 2

PLAN_TEXT_COL = "ocr_text"

ALL_OCR_TEXT_DATA_PATH = TRAIN_DATA / "all_ocr_train.csv"
ALL_OCR_TEXT_COL = "ocr_text"

BATCH_SIZE = 32