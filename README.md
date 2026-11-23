# llm-ingroup-lang


### Environment
1. Install Python packages.
```
pip install sentence_transformers datasets jsonlines gensim scikit-learn transformers nltk accelerate bert_score peft trl wandb google-genai

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
2. Set up the API key for Gemini in `models/api.py`.

### Data Processing
1. Download the in-group language glossary, split it into valid, test and holdout set, and put it at data dir (from https://github.com/lucy3/ingroup_lang/blob/master/data/glossaries.csv).
```
mkdir data/lucy_glossary
```

2. Download the model for computing cosine similarity and put it in the experimental data dir (cc.en.300.bin, from https://fasttext.cc/docs/en/crawl-vectors.html).
```
mkdir experiments/data
```
3. Copy NER annotation to experimental data dir.
```
cp data/NER_wnut.csv experiments/data
```

4. Run the data preprocessing script, including generating duplicate Reddit post hashes, computing NPMI based on the training set, retrieving contexts for Reddit posts, and generating evaluation data for WSD.
```
sh preprocessing.sh
```

### Simulation
1. Script for simulation validation on different scales of the model.
```
sh simulation_llama_ablation.sh
```
2. Script for running simulation to produce lexical innovations for training. Modify the `data_start` and `data_end` to adjust the simulation amount.
```
sh simulation_llama.sh
```
3. Script for evaluating simulation performance. Replace `result_json_path` with the actual path.
```
python evaluation/metrics.py --task_type simulation --result_json result_json_path
```

### Base Model Inference
Script for base model (model not further fine-tuned) inference. 
```
sh baseline_gemini.sh
sh baseline_llama.sh
```

### Model Training
1. Script for model training in main experiments. Note that the path for `simulation_data` needs to be specified manually.
```
sh training_llama.sh
```
2. Script for analysis study. Note that the path for `simulation_data` needs to be specified manually.
```
sh ablation_llama.sh
```
3. Script for LLM-as-a-judge evaluation. Note that `eval_set` needs to be specified manually.
```
sh score.sh
```
4. Evaluation script to produce metric results. Note that `seed` needs to be specified manually.
```
sh evaluate.sh seed
```
