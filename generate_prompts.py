from SETTINGS import *
import pandas as pd
from pathlib import Path

def generate_train_prompts(data: pd.Series, 
                            ans: pd.Series, 
                            year: pd.Series, 
                            prompt_path: Path, 
                            rag_data: pd.Series = None):
    
    n_data, n_ans, n_year = len(data), len(ans), len(year)
    if not (n_data == n_ans == n_year):
        raise ValueError(
            f"Length mismatch: data({n_data}), ans({n_ans}), year({n_year}) must be equal."
        )
    if rag_data is not None and len(rag_data) != n_data:
        raise ValueError(
            f"Length mismatch: rag_data({len(rag_data)}) must match data({n_data})."
        )

    prompt_path = Path(prompt_path)
    cleanprompt = prompt_path.read_text(encoding="utf-8")

    prompts = pd.Series(
        [
            chat_template(cleanprompt.replace("[YYYY]", str(y)).replace("[PLAN]", str(p)), a)
            for p, y, a in zip(data, year, ans)
        ],
        index=data.index,
    )

# import argparse

# if __name__ == "__main__":
#     # Set up argument parser
#     parser = argparse.ArgumentParser(description="Generate prompts for specified LLM.")
#     parser.add_argument(
#         "--model",
#         type=str,
#         required=True,
#         choices=registry.choices(),
#         help=f"Name of the model to use. Must be one of: {', '.join(registry.choices())}."
#     )

#     args = parser.parse_args()

#     cfg = REGISTRY.get(args.model)

