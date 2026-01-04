from SETTINGS import *
from pathlib import Path
import pandas as pd
import argparse
from utils import *
from data_processing import *

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Choose a plan feature type for which to process the results.")
    parser.add_argument(
        "--feature_type",
        type=str,
        required=True,
        choices=['matching', 'autoenrollment', 'vesting'],
        help=f"Name of the model to use. Must be one of: {', '.join(['matching', 'autoenrollment', 'vesting'])}."
    )

    args = parser.parse_args()

    post_process_feature_results(args.feature_type)