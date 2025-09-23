from pathlib import Path
from datetime import datetime
from peft import LoraConfig
from transformers import TrainingArguments
from models_registry import ModelRegistry
from copy import deepcopy

DEBUG = True

HOME = Path.home()

MODELS_LOC = Path("/home/jmd324/scratch_pi_co337/jmd324") / "models"

TABULATOR_BASE_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct" # Model id from the huggingface website
SNIPPET_BASE_MODEL_ID = TABULATOR_BASE_MODEL_ID

TABULATOR_BASE_MODEL_DIR = MODELS_LOC / "llama_3p1_8b_inst"
SNIPPET_BASE_MODEL_DIR = TABULATOR_BASE_MODEL_DIR

PARTITION_1_PATH = Path()
PARTITION_2_PATH = Path()

TABULATOR_ANSWER_COL = ""
SNIPPET_ANSWER_COL = ""

TABULATOR_PROMPT_PATH = Path()
SNIPPET_PROMPT_PATH = Path()

NUM_TABULATOR_TRAIN_EPOCHS = 4
NUM_SNIPPET_TRAIN_EPOCHS = 3

NOW = datetime.now().strftime("_%m_%d_%Y-%I_%p") # suffix for filenames so that we know which are newest

DEFAULT_TRAINING_ARGS = TrainingArguments( # Automatically set num_train_epochs and save_dir based on registry
        per_device_train_batch_size=4,
        auto_find_batch_size=True,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        bf16=True,
        logging_steps=2,
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
    train_data_path=PARTITION_1_PATH,
    prompt_path=TABULATOR_PROMPT_PATH,
    train_epochs=NUM_TABULATOR_TRAIN_EPOCHS,
    train_ans_col=TABULATOR_ANSWER_COL,
    training_args=DEFAULT_TRAINING_ARGS,
)
REGISTRY.add(
    "f2",
    type="TABULATOR",
    model_id=TABULATOR_BASE_MODEL_ID,
    save_path=MODELS_LOC / "f2",
    base_model_path=TABULATOR_BASE_MODEL_DIR,
    train_data_path=PARTITION_2_PATH,
    prompt_path=TABULATOR_PROMPT_PATH,
    train_epochs=NUM_TABULATOR_TRAIN_EPOCHS,
    train_ans_col=TABULATOR_ANSWER_COL,
    training_args=DEFAULT_TRAINING_ARGS,
)
REGISTRY.add(
    "g1",
    type="SNIPPET",
    model_id=SNIPPET_BASE_MODEL_ID,
    save_path=MODELS_LOC / "g1",
    base_model_path=SNIPPET_BASE_MODEL_DIR,
    train_data_path=PARTITION_1_PATH,
    prompt_path=SNIPPET_PROMPT_PATH,
    train_epochs=NUM_SNIPPET_TRAIN_EPOCHS,
    train_ans_col=SNIPPET_ANSWER_COL,
    training_args=DEFAULT_TRAINING_ARGS,
)
REGISTRY.add(
    "g2",
    type="SNIPPET",
    model_id=SNIPPET_BASE_MODEL_ID,
    save_path=MODELS_LOC / "g2",
    base_model_path=SNIPPET_BASE_MODEL_DIR,
    train_data_path=PARTITION_2_PATH,
    prompt_path=SNIPPET_PROMPT_PATH,
    train_epochs=NUM_SNIPPET_TRAIN_EPOCHS,
    train_ans_col=SNIPPET_ANSWER_COL,
    training_args=DEFAULT_TRAINING_ARGS,
)

# optional debug suffix
if DEBUG:
    REGISTRY.append_suffix_to_save_paths(NOW)

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