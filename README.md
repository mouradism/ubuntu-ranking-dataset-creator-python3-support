
# Fix Python 2 → Python 3 compatibility issues 
# Improve code style and/or refactor
# Update README or documentation

# Ubuntu Dialogue Corpus v2.0

This repository contains the files and scripts for generating the **Ubuntu Dialogue Corpus v2.0**, a large-scale multi-turn dialogue dataset designed for research on dialogue systems.

---

## Updates from Ubuntu Corpus v1.0

Version 2.0 includes important updates and bug fixes compared to v1.0. These changes mean results on the two datasets are **not directly comparable**, but models trained on v1.0 should transfer reasonably well to v2.0 with some hyperparameter tuning:

- **Train/validation/test split by time:**  
  Training data covers from 2004 to April 27, 2012; validation from April 27 to August 7, 2012; and test from August 7 to December 1, 2012. This simulates real-world scenarios by training on past data to predict future data.

- **Context length sampling changed:**  
  Validation and test sets now sample context lengths uniformly (between 2 and max size) instead of an inverse distribution, increasing average context length to better model long-term dependencies.

- **Tokenization and entity replacement removed:**  
  Earlier aggressive tokenization and entity replacement were removed. Users may apply their preferred preprocessing.

- **Explicit end-of-utterance (`__eou__`) and end-of-turn (`__eot__`) tokens:**  
  Unlike v1, where consecutive utterances from the same user were concatenated, v2 marks utterance boundaries clearly, improving consistency.

- **Bug fix in false response distributions:**  
  The distribution of false responses now matches true responses more closely in length, preventing models from exploiting length differences.

---

## Corpus Generation Files

### `generate.sh`

Shell script that runs `create_ubuntu_dataset.py` with parameters to generate dataset splits.

Example usage:  
```bash
./generate.sh -t -s -l
```

### `create_ubuntu_dataset.py`

Main Python script to generate training, validation, and test datasets from Ubuntu Dialogue Corpus v1.0 dialogs. It downloads raw 1-on-1 dialogs and samples positive and negative examples to create ranking datasets.

**Key arguments:**

| Argument       | Description                                          | Default |
|----------------|------------------------------------------------------|---------|
| --data_root    | Directory to download/extract 1-on-1 dialogs         | `.`     |
| --seed        | Random seed for reproducibility                       | `1234`  |
| -o, --output  | Output CSV file path                                  | None    |
| -t, --tokenize| Tokenize output using `nltk.word_tokenize`            | False   |
| -s, --stem    | Apply stemming with `SnowballStemmer` (only if tokenize enabled) | False |
| -l, --lemmatize| Apply lemmatization with `WordNetLemmatizer` (only if tokenize enabled) | False |

*Note: If both `-s` and `-l` are specified, stemming is applied before lemmatization.*

---

### Subcommands

- **train**: Generate training data  
  Arguments:  
  - `-p` Positive example ratio (default 0.5)  
  - `-e`, `--examples` Number of examples to generate (default 1,000,000)

- **valid**: Generate validation data  
  Arguments:  
  - `-n` Number of distractors per context (default 9)

- **test**: Generate test data  
  Arguments:  
  - `-n` Number of distractors per context (default 9)

---

## Meta folder files

- `trainfiles.csv`, `valfiles.csv`, `testfiles.csv`:  
  Map original dialogue files to train, validation, and test splits.

---

## Generated Corpus Files

- **train.csv**:  
  Training data with columns: context, candidate response, and label (1 for true response, 0 for randomly sampled false response).  
  ~463 MB, 1 million examples.

- **valid.csv**:  
  Validation set with one context, one true response, and 9 distractors per row.  
  ~27 MB, 19,561 rows.

- **test.csv**:  
  Same format as validation.  
  ~27 MB, 18,921 rows.

---

## Baseline Results

| Model            | Recall@1 (1 in 2) | Recall@1 (1 in 10) | Recall@2 (1 in 10) | Recall@5 (1 in 10) |
|------------------|-------------------|--------------------|--------------------|--------------------|
| Dual Encoder LSTM | 0.869             | 0.552              | 0.721              | 0.924              |
| Dual Encoder RNN  | 0.777             | 0.379              | 0.561              | 0.836              |
| TF-IDF           | 0.749             | 0.488              | 0.587              | 0.763              |

---

## Hyperparameters Used

### Dual Encoder LSTM

\`\`\`
act_penalty=500
batch_size=256
conv_attn=False
corr_penalty=0.0
emb_penalty=0.001
fine_tune_M=True
fine_tune_W=False
forget_gate_bias=2.0
hidden_size=200
is_bidirectional=False
lr=0.001
lr_decay=0.95
max_seqlen=160
n_epochs=100
n_recurrent_layers=1
optimizer='adam'
penalize_activations=False
penalize_emb_drift=False
penalize_emb_norm=False
pv_ndims=100
seed=42
shuffle_batch=False
sort_by_len=False
sqr_norm_lim=1
use_pv=False
xcov_penalty=0.0
\`\`\`

### Dual Encoder RNN

\`\`\`
act_penalty=500
batch_size=512
conv_attn=False
corr_penalty=0.0
emb_penalty=0.001
fine_tune_M=False
fine_tune_W=False
forget_gate_bias=2.0
hidden_size=100
is_bidirectional=False
lr=0.0001
lr_decay=0.95
max_seqlen=160
n_epochs=100
n_recurrent_layers=1
optimizer='adam'
penalize_activations=False
penalize_emb_drift=False
penalize_emb_norm=False
pv_ndims=100
seed=42
shuffle_batch=False
sort_by_len=False
sqr_norm_lim=1
use_pv=False
xcov_penalty=0.0
\`\`\`

---

## Project Structure

\`\`\`
project_root/
├── dialogs/
│   ├── 1/
│   │   ├── 1.tsv
│   │   ├── 2.tsv
│   │   └── ...
│   ├── 2/
│   │   ├── 1.tsv
│   │   ├── 2.tsv
│   │   └── ...
│   ├── 3/
│   │   └── ...
│   └── 5/
│       └── ...
└── src/
    └── meta/
        ├── trainfiles.csv
        ├── valfiles.csv
        ├── testfiles.csv
    ├── create_ubuntu_dataset.py
    ├── download_punkt.py
    ├── generate.sh
    └── ...
\`\`\`

---

## Credits

This work builds upon the original Ubuntu Dialogue Corpus v1.0 and its updates by R. Kadlec et al. and other contributors. The baseline model hyperparameters were provided by the authors of the Dual Encoder models and related research.

---

## Usage

To generate datasets, simply run:

\`\`\`bash
./src/generate.sh
\`\`\`
