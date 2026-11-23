#!/bin/bash

export PYTHONPATH=$(pwd)

# data preprocessing; store at experiments/data
# generate duplicate reddit post hash
python preprocessing/check_duplicate_data.py

# compute npmi based on training set
python preprocessing/compute_npmi.py

# retrieve contexts from reddit posts
python preprocessing/retrieve_context.py

# generate evaluation data for wsd
python preprocessing/generate_eval_wsd.py

