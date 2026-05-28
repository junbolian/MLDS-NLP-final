# Multi-LexSum: Hierarchical Summarization and Civil Rights Lawsuit Classification

> Northwestern MLDS · NLP Final Project · Spring 2026  
> Team: Junbo Lian (Jacob), Yujun Sun, Feng Xiong, Jianong Xu

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](#)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#)

**Live demo**: [Hugging Face Space](https://huggingface.co/spaces/EvelynXuNU/NLP_FinalProject_demo)  
**Space entrypoint**: [`app.py`](app.py)

## Project Overview

This project studies long-form legal NLP on the **Multi-LexSum** dataset. The underlying cases are extremely long federal civil-rights lawsuits, often far beyond the input window of standard transformer models. We built an end-to-end pipeline that:

1. reduces and summarizes long case text at three granularities: **long**, **short**, and **tiny**
2. predicts whether the case **sought class-action status**
3. predicts the case's **grouped case type**
4. exposes the full workflow in a **Gradio demo app** with explanation artifacts

The core challenge is length. Many cases are tens of thousands of tokens long, so the project uses a hierarchical strategy: first extract a compact evidence packet, then generate summaries, then classify on a concise representation.

## Demo

The public demo lives on Hugging Face Spaces:

- [https://huggingface.co/spaces/EvelynXuNU/NLP_FinalProject_demo](https://huggingface.co/spaces/EvelynXuNU/NLP_FinalProject_demo)

The app has three tabs:

- **Summaries**: long, short, and tiny outputs
- **Predictions**: class-action and grouped case-type predictions with confidence
- **Explainability**: LR SHAP token attributions and a lightweight BERT attention heatmap

Because the public Space must fit under the free-tier storage cap, the deployed bundle is intentionally slim:

- `case_type` uses a bundled Legal-BERT checkpoint
- `class_action_sought` currently falls back to Logistic Regression
- `facebook/bart-large-cnn` loads from the Hugging Face model hub at runtime
- the bundled tiny-summary T5 checkpoint is a smoke fine-tune artifact, suitable for demo use but not a full final retraining claim

## Dataset

We use [`allenai/multi_lexsum`](https://huggingface.co/datasets/allenai/multi_lexsum), a legal summarization dataset built from U.S. civil-rights litigation.

After filtering to cases with complete reference summaries and classification metadata, the working dataset contains **1,602 cases**:

- **Train / val / test**: `1129 / 161 / 312`
- **Raw case lengths**: median `44,789` tokens, 95th percentile `335,302`, max `3,002,324`
- **Targets**:
  - `class_action_sought` (binary)
  - `case_type` grouped from 24 raw labels into 5 categories

The grouped case-type labels are:

- Criminal Justice
- Civil Rights & Equality
- Healthcare & Disability
- Immigration & Education
- Speech & Voting

This preprocessing step turns a very sparse, long-tailed label space into a more learnable and interpretable multi-class problem.

## Data Preparation

The cleaned dataset is produced by the preprocessing pipeline in [`src/data.py`](src/data.py), [`src/cleaning.py`](src/cleaning.py), and [`src/case_type_grouping.py`](src/case_type_grouping.py).

Key cleaning steps include:

- removing legal citations such as reporter references and statute references
- stripping page markers and footnote-style artifacts
- normalizing whitespace
- grouping 24 raw case-type labels into 5 higher-level categories

The final canonical artifact is:

- `data/multilexsum_clean.parquet`

Notebook references:

- [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb)
- [`notebooks/02_cleaning.ipynb`](notebooks/02_cleaning.ipynb)

## Summarization Pipeline

The summarization system is hierarchical:

`full cleaned case -> extractive reduction -> abstractive long summary -> short summary -> tiny summary`

### 1. Extractive reduction

Implemented in [`src/summarize/extractive.py`](src/summarize/extractive.py).

- Default method: **LexRank**
- Target evidence packet: about **3,500 tokens**
- Purpose: compress extremely long case text into a tractable summary input while preserving key factual sentences

On the test split, median source length drops from about `43,466` tokens to `3,498`, a reduction of roughly **92%**.

### 2. Abstractive summarization

Implemented in [`src/summarize/abstractive.py`](src/summarize/abstractive.py).

- Default model: **`facebook/bart-large-cnn`**
- The reduced evidence packet is chunked and summarized into a **long** summary
- The **short** summary is generated from the long summary

### 3. Tiny summary

Implemented in [`src/summarize/tiny.py`](src/summarize/tiny.py).

- Model family: **T5-small**
- Task: generate a one-sentence **tiny** summary from the short summary

### Evaluation

Summarization evaluation is unified in [`src/evaluate.py`](src/evaluate.py) and supports:

- ROUGE-1
- ROUGE-2
- ROUGE-L
- BERTScore

Current checked-in summary evaluation artifacts are smoke-test outputs, not a full final large-scale benchmark run:

- [`results/abstractive_summaries.csv`](results/abstractive_summaries.csv)
- [`results/summary_eval.csv`](results/summary_eval.csv)
- [`results/summary_eval_by_granularity.csv`](results/summary_eval_by_granularity.csv)

Notebook reference:

- [`notebooks/03_summarization.ipynb`](notebooks/03_summarization.ipynb)

## Classification Pipeline

The project includes two classification tasks:

1. **Class-action prediction**: whether a class action was sought
2. **Case-type prediction**: 5-way grouped case-type classification

The baseline training setup uses the dataset's human-written `long_ref` summary as input text, which keeps all models on the same concise representation.

Implemented models:

- **Naive Bayes**
- **Logistic Regression**
- **Bi-LSTM**
- **Legal-BERT**

Relevant code:

- [`src/classify/train.py`](src/classify/train.py)
- [`src/classify/classical.py`](src/classify/classical.py)
- [`src/classify/lstm.py`](src/classify/lstm.py)
- [`src/classify/bert.py`](src/classify/bert.py)

### Headline results

#### `class_action_sought`

| Model | Accuracy | Macro-F1 | AUC |
|---|---:|---:|---:|
| Naive Bayes | 0.814 | 0.809 | 0.920 |
| Logistic Regression | 0.894 | 0.890 | 0.968 |
| Bi-LSTM | 0.872 | 0.857 | 0.929 |
| **Legal-BERT** | **0.952** | **0.948** | **0.971** |

#### grouped `case_type`

| Model | Accuracy | Macro-F1 | AUC |
|---|---:|---:|---:|
| Naive Bayes | 0.776 | 0.684 | 0.944 |
| Logistic Regression | 0.760 | 0.630 | 0.971 |
| Bi-LSTM | 0.744 | 0.636 | 0.914 |
| **Legal-BERT** | **0.897** | **0.854** | **0.972** |

The strongest classification takeaway is that **Legal-BERT clearly outperforms the baselines**, especially on the imbalanced multi-class case-type task. It is also the only model that consistently recovers the rare **Healthcare & Disability** class.

Notebook references:

- [`notebooks/04_classification_class_action.ipynb`](notebooks/04_classification_class_action.ipynb)
- [`notebooks/05_classification_case_type.ipynb`](notebooks/05_classification_case_type.ipynb)
- [`notebooks/06_classification_colab.ipynb`](notebooks/06_classification_colab.ipynb)

## Interactive App and Explainability

The interactive app lives in:

- [`app/gradio_app.py`](app/gradio_app.py)
- [`app/inference.py`](app/inference.py)

The main inference path is:

`raw case text -> LexRank reduction -> BART summaries -> tiny T5 summary -> classification -> explanation artifacts`

The app returns:

- generated long / short / tiny summaries
- class-action prediction with confidence
- grouped case-type prediction with confidence
- LR SHAP explanation images
- a BERT attention heatmap
- metadata about reduction and model fallback behavior

Explainability helpers are exposed through:

- [`src/explain.py`](src/explain.py)
- [`src/classify/explain.py`](src/classify/explain.py)

The design intentionally uses:

- **Legal-BERT** as the strongest predictive model when available
- **Logistic Regression + SHAP** as the most transparent explanation companion

## Error Analysis

Error analysis is documented in:

- [`notebooks/06_error_analysis.ipynb`](notebooks/06_error_analysis.ipynb)
- [`results/error_cases.md`](results/error_cases.md)

Per-case prediction exports are produced by:

- [`app/export_predictions.py`](app/export_predictions.py)

This workflow makes it easy to inspect:

- disagreement across model families
- high-confidence mistakes
- low-confidence correct predictions
- summary/reference divergence
- rare-class failures and recoveries

Some of the main observed patterns:

- boundary cases that mix immigration, labor, prison, or RICO language often split the models
- prison-heavy wording can pull all models toward **Criminal Justice**
- Legal-BERT is much more reliable than the baselines on the rarest grouped case type

## Repository Structure

```text
MLDS-NLP-final/
├── app/                      # Gradio app, inference wrapper, export helper
├── data/                     # cached parquet data (gitignored)
├── models/                   # trained checkpoints (gitignored)
├── notebooks/                # EDA, cleaning, summarization, classification, error analysis
├── presentation/             # slide deck
├── results/                  # figures, CSVs, evaluation outputs, error-analysis notes
├── src/
│   ├── classify/             # classical models, LSTM, Legal-BERT
│   ├── summarize/            # extractive, abstractive, tiny summarizers
│   ├── case_type_grouping.py
│   ├── cleaning.py
│   ├── data.py
│   ├── evaluate.py
│   ├── explain.py
│   └── features.py
├── app.py                    # Hugging Face Spaces entrypoint
├── requirements.txt
└── README.md
```

## Local Setup

```bash
git clone https://github.com/junbolian/MLDS-NLP-final.git
cd MLDS-NLP-final
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the App Locally

After the required model checkpoints are available under `models/`:

```bash
python -m src.data --download
python -m src.cleaning --force
python -m app.gradio_app
```

If port `7860` is already in use:

```bash
GRADIO_SERVER_PORT=7861 python -m app.gradio_app
```

## Reproducing the Main Artifacts

### Data

```bash
python -m src.data --download
python -m src.cleaning --force
```

### Extractive summarization artifact

```bash
python -m src.summarize.extractive \
  --input data/multilexsum_clean.parquet \
  --output results/extractive_lengths.csv \
  --split test \
  --method lexrank \
  --target-tokens 3500
```

### Abstractive summaries

```bash
python -m src.summarize.abstractive \
  --input data/multilexsum_clean.parquet \
  --output results/abstractive_summaries.csv \
  --model-key bart-large-cnn \
  --split test \
  --extractive-tokens 3500
```

### Tiny T5 training

```bash
python -m src.summarize.tiny \
  --train \
  --input data/multilexsum_clean.parquet \
  --output-dir models/t5_tiny_summarizer \
  --epochs 3 \
  --batch-size 4 \
  --lr 5e-5
```

### Classification training

```bash
python -m src.classify.train --task class_action --model nb
python -m src.classify.train --task class_action --model lr
python -m src.classify.train --task case_type --model nb
python -m src.classify.train --task case_type --model lr
python -m src.classify.train --task class_action --model lstm --device cuda
python -m src.classify.train --task case_type --model lstm --device cuda
python -m src.classify.train --task class_action --model bert --device cuda
python -m src.classify.train --task case_type --model bert --device cuda
```

### Evaluation

```bash
python -m src.evaluate \
  --task summarization \
  --predictions results/abstractive_summaries.csv \
  --references data/multilexsum_clean.parquet \
  --output results/summary_eval.csv
```

```bash
python -m app.export_predictions --all --split test
```

## Limitations

- The public demo uses a storage-constrained deployment bundle, so it does not include every checkpoint locally.
- The bundled tiny-summary T5 artifact is a smoke fine-tune checkpoint, not a full final training run.
- The checked-in summarization evaluation CSVs are smoke-test artifacts, not the result of a full expensive batch run.
- The app currently classifies generated summaries using models trained on `long_ref`, so there is still a train/inference mismatch in the fully live path.

## Future Work

- retrieval-augmented summarization for better factual grounding
- retraining classifiers on machine-generated long summaries
- stronger legal-domain generation models for higher factual consistency

## Citation

If you use this project or the underlying dataset:

```bibtex
@inproceedings{shen-etal-2022-multi-lexsum,
  title     = {{Multi-LexSum}: Real-World Summaries of Civil Rights Lawsuits at Multiple Granularities},
  author    = {Shen, Zejiang and Lo, Kyle and Yu, Lauren and Dahlberg, Nathan and Schlanger, Margo and Downey, Doug},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2022}
}
```

## License

MIT
