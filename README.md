# Multi-LexSum: Hierarchical Summarization & Civil Rights Lawsuit Classification

> Northwestern MLDS · NLP Final Project · Spring 2026
> Team: Junbo Lian (Jacob), Yujun Sun, Feng Xiong, Jianong Xu

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](#)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#)

🤗 **Live Demo**: https://huggingface.co/spaces/<team>/multilexsum-demo *(Jianong to fill)*

---

## Overview

This project tackles Option 2 of the NLP final project using the **Multi-LexSum** dataset — 9,280 federal civil-rights case summaries authored by legal experts. We deliver three components:

1. **Multi-granularity summarization** — long / short / tiny summaries for cases that frequently exceed 200 pages of source text.
2. **Two classifiers** — predicting (a) whether a class action was sought (binary) and (b) the case type (5–7 grouped categories).
3. **Interactive Gradio app** — paste any case text and get all summaries + both predictions with explanations.

## Quick Start

```bash
git clone https://github.com/<team>/nlp-final-multilexsum.git
cd nlp-final-multilexsum
pip install -r requirements.txt

# Download + cache data (~5 min, ~2 GB)
python -m src.data --download

# Reproduce all results (GPU recommended for §3.4-3.5)
make all

# Launch the Gradio app locally
python -m app.gradio_app
```

## Repository Structure

```
nlp-final-multilexsum/
├── data/                     # parquet cache (gitignored)
├── notebooks/                # 01_eda → 06_error_analysis
├── src/
│   ├── data.py               # HuggingFace loader + caching
│   ├── cleaning.py           # regex + spaCy normalization
│   ├── case_type_grouping.py # 10+ raw types → 5–7 main categories
│   ├── features.py           # TF-IDF / Word2Vec / BERT tokenizer
│   ├── summarize/            # extractive, abstractive, tiny, pipeline
│   ├── classify/             # NB, LR, Bi-LSTM, Legal-BERT
│   ├── evaluate.py           # ROUGE / BERTScore / classification report
│   └── explain.py            # SHAP + bertviz
├── app/                      # Gradio app + inference entry point
├── models/                   # checkpoints (gitignored)
└── results/                  # all figures and CSVs (PPT sources here)
```

## Schedule

| Milestone | Due | Owners | Required deliverables |
|-----------|-----|--------|----------------------|
| **W6 Foundation** | end of W6 (Sun) | Jacob, Jianong | §1 complete · HF Spaces hello-world deployed |
| **W7 Baselines** | end of W7 (Sun) | Feng, Yujun, Jianong | NB + LR on both tasks · LexRank/TextRank extractive · Gradio dummy-model UI · PPT template |
| **W8 Full Models** | end of W8 (Sun) | Feng, Yujun | Bi-LSTM + Legal-BERT on both tasks · 3-granularity summarization complete · all slide drafts |
| **W9 Integration** | end of W9 (Sun) | Jianong, all | Real Gradio app on HF Spaces · explainability · error analysis · unified slide review |
| **W10 Launch** | before W10 class | Jianong, Jacob | Demo video recorded + edited · README sections all replaced |

Status by section: see the indicator next to each header below (🔴 TODO · 🟡 in progress · ✅ complete).

---

# §1 Data and Preprocessing  *(owner: Jacob — ✅ complete)*

## 1.1 Dataset

We use the [`allenai/multi_lexsum`](https://huggingface.co/datasets/allenai/multi_lexsum) dataset (version `v20230518`). The raw release contains **4,539 federal civil-rights cases**; after filtering to cases that have all three reference summaries plus complete classification metadata, our working set is **1,602 cases**. Each entry includes:

- **Source documents**: full legal filings concatenated across multiple docs per case; lengths vary by 3 orders of magnitude (min 1.8k → max 3.0M tokens)
- **Three reference summaries** at distinct granularities, expert-authored and reviewed
- **Metadata**: `class_action_sought` (binary), `case_type` (24 raw labels), filing date, court, state

### Working-set statistics

| Statistic | Value |
|-----------|-------|
| Cases (after filtering) | **1,602** |
| Train / Val / Test | 1,129 / 161 / 312 |
| Source-text tokens — median · mean · p95 · max | 44,789 · 94,428 · 335,302 · **3,002,324** |
| Reference long summary tokens (median / max) | 638 / 8,481 |
| Reference short summary tokens (median / max) | 102 / 671 |
| Reference tiny summary tokens (median / max) | 19 / 43 |
| `class_action_sought = True` rate | 41.8% (class-balance ratio 0.72) |
| `case_type` raw labels | 24 |
| `case_type` grouped categories | 5 (Other empty after final mapping) |

Full EDA: [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb); figures in `results/eda/`; raw stats in `results/eda/summary_stats.json`.

![Source-text length distribution](results/eda/source_length_distribution.png)

*Source-text token distribution. Left: histogram on log x-scale showing the 3-orders-of-magnitude spread. Right: box plot by split — train/val/test distributions are nearly identical, so no distribution-shift concerns for evaluation.*

**Why this motivates hierarchical summarization**: median case has ~45k tokens (87× BERT's 512 limit, 11× Longformer's 4k); top 5% exceed 335k tokens; the longest case is 3M tokens. Direct transformer feed is impossible — see §2.

## 1.2 Cleaning Pipeline

`src/cleaning.py` applies seven regex passes to strip legal-document noise. Hit counts measured on a 50-case sample:

| Pattern | Cases hit | Mean hits / case | Purpose |
|---------|-----------|------------------|---------|
| Reporter citations (`123 F.3d 456`) | 47/50 | **614** | strip case-law references |
| Page markers (`[Page X of Y]`, `Page 3`) | 44/50 | 105 | strip pagination |
| U.S.C. references (`42 U.S.C. § 1983`) | 49/50 | 63 | strip statute citations |
| C.F.R. references (`29 C.F.R. § 1604.11`) | 17/50 | 14 | strip regulation citations |
| Footnote markers (`[1]`, `[fn 2]`) | 14/50 | 8 | strip footnote refs |
| URLs and emails | rare | <1 | strip web artifacts |
| Whitespace normalization | all | — | collapse `\s+` → single space |

Average character-length reduction is **3.1%** (median 3.1%, max 5.2%). The reduction is modest in raw byte terms but substantively important — it removes ~700 high-frequency citation tokens per case that would otherwise dominate TF-IDF features and confuse summarization models.

![Cleaning impact](results/eda/cleaning_impact.png)

*Before vs after character count on a 50-case sample (log scale). Points sit close to the diagonal because the absolute reduction is small, but the citations removed are high-information-density features the regex catches consistently.*

Processing throughput: **74 ms / case** on a single CPU core (entire 1,602-case set cleans in ~2 min).

Before/after comparison: [`notebooks/02_cleaning.ipynb`](notebooks/02_cleaning.ipynb).

## 1.3 case_type Grouping

The raw `case_type` field has **24 labels** (observed in v20230518). We collapse them into **5 thematically coherent groups**:

| Grouped Category | Original `case_type` Values | Cases |
|-----------------|---------------------------|-------|
| **Criminal Justice** | `Prison Conditions` · `Jail Conditions` · `Policing` · `Juvenile Institution` · `Criminal Justice (Other)` · `Indigent Defense` | 577 (36.0%) |
| **Speech & Voting** | `Speech and Religious Freedom` · `Election/Voting Rights` · `Public Benefits / Government Services` · `National Security` · `Presidential/Gubernatorial Authority` | 414 (25.8%) |
| **Immigration & Education** | `Immigration and/or the Border` · `Education` · `Child Welfare` | 363 (22.7%) |
| **Civil Rights & Equality** | `Equal Employment` · `Fair Housing/Lending/Insurance` · `Public Accomm./Contracting` · `School Desegregation` · `Environmental Justice` · `Public Housing` | 179 (11.2%) |
| **Healthcare & Disability** | `Disability Rights-Pub. Accom.` · `Mental Health (Facility)` · `Intellectual Disability (Facility)` · `Nursing Home Conditions` | 69 (4.3%) |
| Other | (none after fix) | 0 (0%) |

All 24 raw labels are accounted for. The `group_case_type()` function in `src/case_type_grouping.py` normalizes whitespace around `/` separators (Multi-LexSum uses both `"A/B"` and `"A / B"` forms in different labels), so the mapping is robust to either form.

![case_type distribution: raw vs grouped](results/eda/case_type_distribution.png)

*Left: top 10 of 24 raw `case_type` labels (long-tail visible — bottom 14 labels each have < 30 cases). Right: the 5 grouped categories. Grouping reduces from 24 → 5 classes while preserving thematic coherence.*

![Class-action rate by group](results/eda/classaction_by_casetype.png)

*Class-action rate varies sharply across groups: **Healthcare & Disability (58%) and Criminal Justice (57.7%) lean class-action**, while **Speech & Voting is only 15.5%** (mostly individual First Amendment challenges, not group claims). The strong group→target correlation here is a useful signal for §3.7 (`class_action_sought` prediction).*

**Class imbalance note**: Healthcare & Disability is the smallest group (4.3%). For §3 classification, use `class_weight='balanced'` (sklearn) or weighted cross-entropy (PyTorch/TF) to compensate.

Full mapping rationale: [`docs/case_type_grouping.md`](docs/case_type_grouping.md).

## 1.4 Output Schema

The cleaning pipeline produces a single canonical DataFrame, cached as `data/multilexsum_clean.parquet` (1,602 rows × 15 columns):

```python
case_id              : str          # e.g. 'PB-WV-0002'
source_text          : str          # cleaned full case (concatenated source docs)
n_source_docs        : int          # how many docs were joined
long_ref             : str          # provided long reference summary
short_ref            : str          # provided short reference summary
tiny_ref             : str          # provided one-sentence reference summary
class_action_sought  : bool         # target for §3.7
case_type_raw        : str          # 24 distinct values
case_type_grouped    : str          # 5 groups + 'Other' (target for §3.8)
filing_date          : str | None
court                : str | None
state                : str | None
source_n_chars       : int          # length of cleaned text
source_n_tokens      : int          # whitespace tokens of cleaned text
split                : {'train', 'val', 'test'}
```

## 1.5 Reproducing

```bash
python -m src.data            # ~5–10 min: download + flatten + cache
python -m src.cleaning        # ~2 min: regex + grouping + filter
# Then open notebooks/01_eda.ipynb and Run All
```

EDA notebook: [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb)

---

# §2 Multi-Granularity Summarization  *(owner: Yujun · ✅ complete)*

**Deadlines**: W7 extractive baseline · W8 full pipeline + evaluation
**Depends on**: §1 cleaned parquet (`data/multilexsum_clean.parquet`)

## 2.1 Motivation

Multi-LexSum source cases are too long for direct transformer generation: the working-set median is **44,789 source tokens**, the 95th percentile is **335,302**, and the longest case has **3.0M** tokens. A direct BART/T5-style input is limited to roughly 1k tokens, and even long-context LED-style models are expensive for 1,602 cases. The implemented pipeline is hierarchical:

`full cleaned case -> extractive evidence packet -> abstractive long/short -> T5 tiny`

This keeps every generated summary grounded in selected source sentences while giving the abstractive model a tractable input.

## 2.2 Stage A: Extractive Reduction

Code: `src/summarize/extractive.py`

Both LexRank and TextRank use the same interface:

```python
from src.summarize.extractive import extractive_summarize
result = extractive_summarize(text, method="lexrank", target_tokens=3500)
```

| Method | Graph | PageRank input | Chosen role |
|--------|-------|----------------|-------------|
| LexRank | sentence TF-IDF cosine graph | unweighted edges with cosine ≥ **0.10** | **default** reducer; stable and sparse |
| TextRank | sentence TF-IDF cosine graph | weighted edges with cosine ≥ **0.01** | ablation baseline |

Parameters: max **650** candidate sentences per case, sentence length **8–120** tokens, target output **3,500** whitespace tokens. Very long cases first pass through a position-balanced candidate cap so graph ranking stays CPU-safe.

Artifact: `results/extractive_lengths.csv` was generated on the test split (**312 cases**). Median source length went **43,466 -> 3,498 tokens**, a **92.0% median reduction**, satisfying the W7/W8 target of ≥90%.

## 2.3 Stage B: Abstractive Generation

Code: `src/summarize/abstractive.py`

| Candidate | Context | Pros | Cons | Decision |
|-----------|---------|------|------|----------|
| `facebook/bart-large-cnn` | 1,024 tokens | reliable summarization checkpoint; easy to run locally after Stage A reduction | needs chunk/combine for 3,500-token packets | **chosen default** |
| `google/pegasus-x-large` | longer context | designed for long summarization | heavier download/runtime; less predictable local availability | ablation only |
| `allenai/led-large-16384-arxiv` | 16k tokens | can consume larger evidence packets | slow and memory-heavy for team laptops | ablation only |

Generation details: the reducer produces a ~3,500-token evidence packet; BART chunks this into model-sized windows, generates short chunk summaries, then combines them into a long summary. The short summary is generated from the long summary so the two granularities stay consistent.

## 2.4 Tiny Summary

Code: `src/summarize/tiny.py`

Tiny summaries use **T5-small** fine-tuned on `short_ref -> tiny_ref` pairs. The default full run is **3 epochs**, learning rate **5e-5**, batch size **4**, max source length **256**, max target length **48**. The local setup smoke run trained on 32 train / 8 val examples for one epoch to verify the checkpoint path and logging; it wrote:

- `models/t5_tiny_summarizer/`
- `results/t5_tiny_val_loss.csv`
- `results/t5_tiny_val_loss.png`

Smoke validation loss after one epoch: **3.565**. Re-run the full command in §2.8 for final slide numbers.

## 2.5 LLM Zero-Shot Baseline

Code: `src/summarize/llm_baseline.py`

Chosen baseline: **GPT-4o-mini**, because it has a 128k context window and low text-token pricing. Pricing used in the estimator is OpenAI's published GPT-4o-mini rate: **$0.15 / 1M input tokens** and **$0.60 / 1M output tokens** ([OpenAI pricing](https://platform.openai.com/docs/pricing/)).

The baseline still uses Stage A reduction before the API call, capped at **10,000** input tokens, because some cases exceed any practical hosted context. Prompt format: a system instruction requiring faithful legal summarization plus a user prompt asking for strict JSON with `long`, `short`, and `tiny` keys. Local dry-run cost estimates for two test examples were about **$0.00254 per case**.

## 2.6 Evaluation

Code: `src/evaluate.py`

The evaluator computes per-case x granularity:

- ROUGE-1 / ROUGE-2 / ROUGE-L F1 via `rouge-score`
- BERTScore precision / recall / F1 via `bert-score`

Current smoke run (`results/summary_eval.csv`) covers **2 test cases** to verify the end-to-end BART + T5 + metrics path:

| Granularity | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 |
|-------------|---------|---------|---------|--------------|
| long | 0.2143 | 0.0682 | 0.1238 | 0.7403 |
| short | 0.2781 | 0.0841 | 0.1870 | 0.7486 |
| tiny | 0.2453 | 0.1176 | 0.1509 | 0.7353 |

These are smoke-test numbers, not final claims. Run the full test split commands in §2.8 before freezing slide 6.

## 2.7 Qualitative Analysis

Notebook: `notebooks/03_summarization.ipynb`

The notebook loads `results/summary_eval.csv`, selects the **5 highest ROUGE-L** long-summary cases as good cases, and selects the **5 lowest ROUGE-L** cases for hallucination review. Each case prints reference vs prediction side-by-side, with manual notes for unsupported parties, outcomes, statutes, remedies, and missing procedural facts.

Slides 4–6 drafts are in `presentation/slides.pptx`. Slide 6 is wired as a results/qualitative placeholder and should be updated after the full test run.

## 2.8 Reproducing

```bash
# Setup
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Data
python -m src.data --download
python -m src.cleaning --force

# Stage A: extractive length artifact
python -m src.summarize.extractive \
  --input data/multilexsum_clean.parquet \
  --output results/extractive_lengths.csv \
  --split test \
  --method lexrank \
  --target-tokens 3500

# Stage B: BART long/short summaries
python -m src.summarize.abstractive \
  --input data/multilexsum_clean.parquet \
  --output results/abstractive_summaries.csv \
  --split test \
  --model-key bart-large-cnn \
  --extractive-tokens 3500

# Tiny T5 full fine-tune
python -m src.summarize.tiny \
  --train \
  --input data/multilexsum_clean.parquet \
  --output-dir models/t5_tiny_summarizer \
  --epochs 3 \
  --batch-size 4 \
  --lr 5e-5

# Evaluation
python -m src.evaluate \
  --predictions results/abstractive_summaries.csv \
  --references data/multilexsum_clean.parquet \
  --output results/summary_eval.csv

# LLM baseline cost-only dry run
python -m src.summarize.llm_baseline \
  --input data/multilexsum_clean.parquet \
  --output results/llm_baseline_costs.csv \
  --split test \
  --dry-run-cost
```

Implemented deliverables: `src/summarize/extractive.py`, `src/summarize/abstractive.py`, `src/summarize/tiny.py`, `src/summarize/pipeline.py`, `src/summarize/llm_baseline.py`, `src/evaluate.py`, `notebooks/03_summarization.ipynb`, `presentation/slides.pptx`, `results/extractive_lengths.csv`, and smoke-run summary evaluation artifacts.

---

# §3 Classification  *(owner: Feng · ✅ complete)*

Two classification tasks, four models per task, one unified CLI. The headline:
**Legal-BERT wins both tasks** (test macro-F1 = 0.948 binary / 0.854 multi-class), and is
the *only* model that handles the long-tail multi-class label.

## 3.1 Task setup

| Task | Target column | Classes | Test positives / class |
|------|---------------|---------|------------------------|
| `class_action` | `class_action_sought` | 2 (Yes / No) | 116 / 196 |
| `case_type`    | `case_type_grouped`   | 5 (Criminal Justice, Civil Rights & Equality, Healthcare & Disability, Immigration & Education, Speech & Voting) | 107 / 42 / 16 / 74 / 73 |

Input text: **`long_ref`** (the human-written long reference summary, median ≈ 250 tokens) — fits Legal-BERT's 512-token cap with no chunking, and gives every model the exact same input. Split: 1,129 train / 161 val / 312 test (from §1 cleaning).

## 3.2 Featurization

| Model         | Feature                            | Where |
|---------------|------------------------------------|-------|
| Naive Bayes   | TF-IDF (uni + bigram, max 50k, `sublinear_tf=True`) | `src/features.py` |
| Logistic Reg  | same TF-IDF, pipelined             | `src/features.py` |
| Bi-LSTM       | self-trained Word2Vec (300d, gensim, training-split corpus only) | `src/classify/lstm.py` |
| Legal-BERT    | `nlpaueb/legal-bert-base-uncased` tokenizer, max 512 | `src/classify/bert.py` |

## 3.3 Naive Bayes

`sklearn.naive_bayes.ComplementNB(alpha=0.3)` — Complement variant is more stable than MultinomialNB on the imbalanced 5-class `case_type` task where the smallest class is 8× smaller than the largest.

## 3.4 Logistic Regression

`LogisticRegression(solver='saga', penalty='l2', C=1.0, class_weight='balanced', max_iter=2000)`. The `class_weight='balanced'` re-weighting matches the rationale in [`docs/case_type_grouping.md`](docs/case_type_grouping.md). **SHAP top-token explanation** via `shap.LinearExplainer` (exact for sparse linear models, no sampling) — saved to `results/lr_shap_{classaction,casetype}.png`.

## 3.5 Bi-LSTM

PyTorch (not TF/Keras — keeps the project on one DL stack alongside §2 BART/T5). Architecture: `Embedding(vocab, 300, _weight=W2V) → BiLSTM(hidden=128, layers=1, dropout=0.3) → Linear`. Training: AdamW, lr 1e-3, batch 16, class-weighted cross-entropy, 10 epochs with early-stopping on val accuracy (patience 3). Word2Vec is trained on the **training-split tokens only** to avoid leakage.

## 3.6 Legal-BERT fine-tune

`nlpaueb/legal-bert-base-uncased` (110M params), pretrained on US + EU legal corpora. Training: AdamW, lr 2e-5, batch 8 with grad-accumulation 2 (effective 16), linear warmup over 10% of steps, 3 epochs (more overfits on 1,129 samples), class-weighted cross-entropy. **Compute**: Colab A100 fine-tunes each task in ~85 seconds. For free-T4 fallback, append `--bert-model-name distilbert-base-uncased` at ~2 F1-points cost.

## 3.7 Results — `class_action_sought`

Binary task, 312 test cases. AUC ≥ 0.92 for every model; macro-F1 separates them.

| Model        | Features            | Accuracy | Macro-F1 | AUC   | Notes |
|--------------|---------------------|---------:|---------:|------:|-------|
| Naive Bayes  | TF-IDF (Complement) |   0.814  |   0.809  | 0.920 | Strong sparse baseline |
| LR (TF-IDF)  | L2 · balanced       |   0.894  |   0.890  | 0.968 | 1-sec training, surprisingly close |
| Bi-LSTM      | Word2Vec 300d       |   0.872  |   0.857  | 0.929 | Slightly under-performs LR |
| **Legal-BERT** | **Fine-tune**     | **0.952**| **0.948**| **0.971** | **+5.8 F1 vs LR, +13.9 vs NB** |

ROC plot: [`results/roc_classaction.png`](results/roc_classaction.png). Confusion: BERT correctly identifies **108 / 116** 'Yes' cases (93 % recall) with only 8 false negatives.

## 3.8 Results — `case_type`

5-class task, 312 test cases. The model gap widens significantly on the long tail.

| Model        | Features              | Accuracy | Macro-F1 | AUC   | F1 on smallest class¹ |
|--------------|-----------------------|---------:|---------:|------:|----------------------:|
| Naive Bayes  | TF-IDF (Complement)   |   0.776  |   0.684  | 0.944 | 0.222                 |
| LR (TF-IDF)  | L2 · balanced         |   0.760  |   0.630  | 0.971 | **0.000** ← zero recall |
| Bi-LSTM      | Word2Vec 300d         |   0.744  |   0.636  | 0.914 | 0.154                 |
| **Legal-BERT** | **Fine-tune**       | **0.897**| **0.854**| **0.972** | **0.645**         |

¹ Healthcare & Disability, n = 16 test cases. **Only Legal-BERT preserves long-tail performance** — LR drops to F1 = 0.000 on the rarest class.

Confusion matrices: `results/confusion_matrices/{nb,lr,lstm,bert}_casetype_test.png`. Side-by-side grid: [`results/case_type_confusion_grid.png`](results/case_type_confusion_grid.png). Macro-F1 by model bar chart: [`results/case_type_macroF1_by_model.png`](results/case_type_macroF1_by_model.png).

## 3.9 Reproducing

W7 (local, CPU):
```bash
python -m src.classify.train --task class_action --model nb
python -m src.classify.train --task class_action --model lr
python -m src.classify.train --task case_type    --model nb
python -m src.classify.train --task case_type    --model lr
python -m src.classify.explain --task class_action       # SHAP plot
python -m src.classify.explain --task case_type
```

W8 (Colab T4/A100 — see [`notebooks/06_classification_colab.ipynb`](notebooks/06_classification_colab.ipynb)):
```bash
python -m src.classify.train --task class_action --model lstm --device cuda
python -m src.classify.train --task case_type    --model lstm --device cuda
python -m src.classify.train --task class_action --model bert --device cuda
python -m src.classify.train --task case_type    --model bert --device cuda
```

Every invocation appends one row per split to [`results/classification_metrics.csv`](results/classification_metrics.csv) and saves the pickled pipeline (NB/LR), `.pt` bundle (LSTM), or HF directory (BERT) under `models/`.

Notebook walkthroughs:
- [`notebooks/04_classification_class_action.ipynb`](notebooks/04_classification_class_action.ipynb) — NB + LR end-to-end on binary task
- [`notebooks/05_classification_case_type.ipynb`](notebooks/05_classification_case_type.ipynb) — 4-model comparison on multi-class
- [`notebooks/06_classification_colab.ipynb`](notebooks/06_classification_colab.ipynb) — Colab-ready W8 trainer

Slides 7–9 drafts: [`presentation/slides.pptx`](presentation/slides.pptx).

✅ **Status**: complete.

---

# §4 Interactive Tool & Evaluation  *(owner: Jianong · 🔴 TODO)*

**Deadlines**: W6 HF Spaces hello-world · W7 dummy-model UI + PPT template · W9 real-model integration + explainability
**Depends on**: §2 summarization pipeline + §3 trained models (for W9 only)

## To do

Sub-section numbering (mirrors slides 10-11):

1. **4.1 Gradio App Architecture** — three-tab design (Summaries / Predictions / Explainability) + UI screenshot
2. **4.2 Inference Pipeline** — `app/inference.py`: model loading + single `predict()` entry point
3. **4.3 Explainability** — SHAP for LR (top tokens), bertviz attention for BERT (sample heatmap)
4. **4.4 Error Analysis** — 5 interesting cases: model disagreement, summary-vs-reference divergence, confidence-vs-correctness
5. **4.5 Evaluation Framework** — `src/evaluate.py` unified interface (CSV / PNG outputs)
6. **4.6 HuggingFace Spaces Deployment** — link, model quantization notes, free-tier limits handled
7. **4.7 Local Usage** — `python -m app.gradio_app`

## Deliverables checklist

**Code — W6**:
- [ ] HF Spaces created (free tier, public) — URL posted in team chat
- [ ] `app/gradio_app.py` — minimal text-in / text-out placeholder, deployed

**Code — W7**:
- [ ] `app/gradio_app.py` — three-tab UI scaffold; each tab wired to a dummy function returning fixed text
- [ ] `presentation/template.pptx` — Northwestern purple, sans-serif, title 32pt / body 18pt

**Code — W9**:
- [ ] `app/inference.py` — `predict(case_text) -> {summaries, class_action, case_type}` unified entry
- [ ] `src/evaluate.py` (classification section) — `classification_report` + confusion matrix helpers
- [ ] `src/explain.py` — SHAP for LR + bertviz attention for BERT

**Artifacts — W9**:
- [ ] HF Spaces deployment with real models (fits within free-tier 16 GB image limit — use quantization or Inference API)
- [ ] `notebooks/06_error_analysis.ipynb`
- [ ] `results/error_cases.md` — 5 disagreement case writeups (case_id + model predictions + reference + your analysis)
- [ ] Final demo video recording (your screen + your voice for slides 10-12; collect Jacob/Yujun/Feng audio for their slides)

**Slides — W9**: slides 10-12 drafts in `presentation/slides.pptx`

🔴 **Status**: not yet written.

---

# §5 Lessons & Future Work  *(owner: Jianong · 🔴 TODO)*

**Deadline**: W9 (after §2 / §3 / §4 are filled in)

## To do

Sub-section numbering (mirrors slide 12):

1. **5.1 Three main takeaways** — one each from summarization, classification, and the interactive tool (technical lessons, not generic "we learned a lot")
2. **5.2 What didn't work** — one paragraph honest about a failure mode (specific hallucination pattern, BERT overfitting on small case_type groups, etc.)
3. **5.3 Future work** — 2-3 concrete extensions (e.g. Legal-LLaMA fine-tuning, hierarchical attention pooling, retrieval-augmented summarization)

🔴 **Status**: not yet written.

---

## Presentation Plan

12 slides · ~10 min final video.

| # | Slide | Owner | Time | Key visual / source |
|---|-------|-------|------|--------------------|
| 1 | Title + Team | **Jacob** | 15s | Project name, 4-person credit |
| 2 | Problem & Multi-LexSum | **Jacob** | 60s | Dataset stats + 200+ page case screenshot |
| 3 | Data Cleaning + EDA + case_type Grouping | **Jacob** | 75s | `results/eda/class_action_distribution.png`, `case_type_distribution.png`, `cleaning_impact.png` + 5-group mapping table |
| 4 | Why Hierarchical Summarization | **Yujun** | 45s | `results/eda/source_length_distribution.png` + transformer context limits |
| 5 | Pipeline: Extract → Abstract | **Yujun** | 75s | Flow diagram: full case → LexRank → BART → 3 granularities |
| 6 | Summarization Results vs Reference | **Yujun** | 60s | ROUGE/BERTScore table + 1 good + 1 hallucination case |
| 7 | Classification Approach | **Feng** | 45s | 4 models × 2 tasks matrix |
| 8 | Results: class_action_sought | **Feng** | 60s | F1/AUC bars + ROC curves (4 models) |
| 9 | Results: case_type | **Feng** | 60s | macro-F1 table + confusion matrix for best model |
| 10 | Interactive Tool — Live Demo | **Jianong** | 90s | Live Gradio screencast at 1080p (pre-record backup) |
| 11 | Explainability & Error Analysis | **Jianong** | 45s | SHAP top-tokens + BERT attention + 1-2 disagreement cases |
| 12 | Lessons & Future Work | **Jianong** | 30s | 3 takeaways + 2 future directions |

**Speaking time**: Yujun 3 min (most, owns hardest content) · Feng 2.75 · Jianong 2.75 · Jacob 2.5 · total ~11 min → trim to 10 in editing.

**Rules**:
1. All figures from `results/` — no ad-hoc notebook screenshots
2. Sentence case headings, never ALL CAPS
3. One key visual per slide; if you want two, split the slide

---

## How to Replace Your Section

If you're a teammate filling in your section:

1. **Open this file in your editor.**
2. **Find your section header** (e.g., `# §2 Multi-Granularity Summarization`).
3. **Delete the entire `> TODO` block and the `🔴 Status: not yet written.` line.**
4. **Replace with your actual content**, keeping the sub-section numbering (2.1, 2.2, …) as listed in the TODO checklist so the structure stays consistent.
5. **Add image/CSV references** as relative links — figures live in `results/`, e.g. `![ROC](results/training_curves/roc_classaction.png)`.
6. **Update the status line** at the top of your section from `🔴 TODO` to `✅ complete`.
7. **Commit with a clear message**: `docs(readme): fill §3 classification`

If you need to add a sub-section, put it at the end of your block (e.g., 3.10) — don't insert into Jacob's or others' blocks.

---

## Working Agreement

A few rules we follow to keep the project clean:

1. **Single source of truth for data**: `data/multilexsum_clean.parquet` is the only dataset file anyone reads. Don't re-clean from raw.
2. **Slide drafts due W8 end, not W9** — placeholder figures OK at this stage.
3. **Figures come from `results/`** — don't screenshot from notebooks; re-export the figure as PNG into `results/` and reference the file path in slides.
4. **Replace TODO blocks via PR, not chat** — keep README as the canonical record of what's done.
5. **Gradio starts in W7 with dummy models** — don't wait for real models to be done.
6. **Weekly sync**: Sundays 21:00 Beijing time, 30 min.

## Citation

If you use this work or the underlying dataset:

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
