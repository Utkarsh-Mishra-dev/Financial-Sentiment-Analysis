# Financial Sentiment Classification using FinBERT

Fine-tuning a pretrained FinBERT Transformer for 3-class financial sentiment
classification (negative / neutral / positive), with a TF-IDF + Logistic
Regression baseline for comparison and a detailed error analysis.

## Problem Statement

Given a sentence from English-language financial news, classify its
sentiment as **negative**, **neutral**, or **positive**, from the
**perspective of an investor** — i.e. whether the news appears favorable or
unfavorable to the company's stock / shareholders, not generic emotional
tone. Many sentences carry no investor-relevant signal at all (e.g. purely
administrative statements) and are labeled neutral by design.

## Dataset

[Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank)
(Malo et al., 2014) — `sentences_50agree` configuration (≥50% annotator
agreement), 4846 sentences.

Labeled by 16 annotators with finance/business backgrounds (3 researchers +
13 Aalto University School of Business master's students in finance,
accounting and economics), explicitly instructed to judge each sentence from
an investor's viewpoint.

**License:** CC BY-NC-SA 3.0 (non-commercial use, with attribution).

**Citation:**
> Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014).
> Good debt or bad debt: Detecting semantic orientations in economic texts.
> *Journal of the Association for Information Science and Technology*, 65(4).
## Annotator agreement txt Cross-referencing
Cross-referencing the four annotator-agreement subsets shows that as the agreement threshold rises, neutral's share increases (59.4%→61.4%) while positive's share decreases (28.1%→25.2%), indicating the positive/neutral boundary is where human annotators disagree most. This matches the model's own error pattern, where positive↔neutral confusions were the most common mistake — suggesting the ambiguity is inherent to the task, not just a model weakness.
### Class distribution (For Sentences_50Agree.txt file )

| Class | Count | % |
|---|---|---|
| Neutral | 2879 | 59.4% |
| Positive | 1363 | 28.1% |
| Negative | 604 | 12.5% |

The dataset is noticeably imbalanced, which is why **macro F1** (not just
accuracy) is used throughout this project to evaluate models fairly across
all three classes.

## Model

**[ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)** — BERT
further pretrained on a financial news corpus (Reuters TRC2 subset), then
originally fine-tuned by Prosus for sentiment classification on Financial
PhraseBank.

This project re-fine-tunes it on a fresh, rigorously split train/val/test
partition of the same dataset: the model's pretrained encoder weights are
reused (transfer learning), but a **new, randomly-initialized 3-class
classification head** is trained from scratch on top. This isn't zero-shot
transfer to a novel domain — it's closer to reproducing FinBERT's own
training recipe on a clean, leak-checked, reproducible split, which still
meaningfully exercises the full fine-tuning pipeline and gives directly
comparable, trustworthy metrics.

**Tokenizer:** WordPiece (uncased) — `do_lower_case=True`, inherited from
`bert-base-uncased`.

## Methodology

1. **EDA** — class distribution, duplicate/conflicting-label checks,
   sentence-length distribution (token-level, using FinBERT's own
   tokenizer), distinctive-word analysis per class (frequency-ratio method),
   and targeted checks for negation and numeric-figure prevalence.
2. **Data preparation** — removed 4 conflicting-label rows and 6 duplicate
   sentences (4846 → 4836), then created a **stratified 80/10/10
   train/val/test split** (seed=42) so class proportions are preserved
   across all splits and the same sentence never appears in more than one
   split (verified programmatically).
3. **Baseline** — TF-IDF (unigrams + bigrams) + Logistic Regression
   (`class_weight="balanced"` to counter class imbalance).
4. **Fine-tuning** — FinBERT fine-tuned with the Hugging Face `Trainer`:
   learning rate 2e-5, batch size 16, up to 4 epochs with early stopping
   (patience=2, monitored on validation macro F1), `max_length=128`
   (covers the 99th percentile sentence length of 68 tokens with margin).
5. **Evaluation** — accuracy, macro F1, per-class precision/recall/F1,
   confusion matrix. Test set touched exactly once, at the end.
6. **Error analysis** — misclassification breakdown by direction,
   cross-checked against negation and numeric-figure prevalence, plus a
   table of representative errors with hypothesized causes.

## Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| Baseline (TF-IDF + Logistic Regression) | 78.3% | 75.4% |
| **FinBERT (fine-tuned)** | **89.3%** | **88.0%** |

### FinBERT per-class performance (test set)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Negative | 0.857 | 0.885 | 0.871 |
| Neutral | 0.904 | 0.923 | 0.914 |
| Positive | 0.883 | 0.831 | 0.856 |

FinBERT improves substantially over the baseline on every class, with the
largest gains on negative and positive recall (72%→88.5% and 70%→83.1%
respectively) — evidence that the Transformer captures context and
implicit sentiment that a bag-of-words model structurally cannot.

## Error Analysis

52 of 484 test sentences were misclassified (10.7% error rate). Key
findings:

- **69% of errors involve the neutral class** in some direction (positive↔neutral,
  negative↔neutral) — the model rarely flips sentiment entirely (only 2 of
  52 errors were positive↔negative), suggesting its mistakes are about
  *degree* of sentiment, not direction.
- **Negation is handled well**: 0% error rate on the 15 test sentences
  containing negation words (small sample, suggestive rather than
  conclusive).
- **Numeric figures don't hurt performance**: sentences containing numbers
  actually had a *lower* error rate (9.4%) than those without (12.2%),
  likely because figures are usually attached to clear directional language
  ("fell 12%", "rose to EUR45m").
- **Dominant failure mode: implicit, context-dependent sentiment** — e.g. a
  sentence describing a cost-cutting salary reduction (positive for
  investors) was misread as neutral, since the surface wording carries no
  explicit positive vocabulary. See `notebooks/05_error_analysis.ipynb` for
  the full representative-error table.

## Project Structure

FinBert Project/
│
├── README.md
├── requirements.txt
│
├── FinancialPhraseBank-v1.0/      # raw dataset (not committed - see License.txt)
│   ├── License.txt
│   ├── README.txt
│   ├── Sentences_50Agree.txt
│   ├── Sentences_66Agree.txt
│   ├── Sentences_75Agree.txt
│   └── Sentences_AllAgree.txt
│
├── notebooks/
│   ├── 01_data_explorations.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_tuning.ipynb            # done on Colab T4 GPU
│   ├── 05_error_analysis.ipynb
│   │
│   ├── data/
│   │   └── processed/             # train.csv, val.csv, test.csv (generated, not committed)
│   │
│   └── results/
│       ├── class_distribution.png
│       ├── length_distribution.png
│       ├── baseline_confusion_matrix.png
│       └── metrics.csv
│
├── finbert_confusion_matrix.png   # from Step 4/5 (Colab), saved at project root
├── finbert_test_metrics.json
└── test_predictions.csv           # used by 05_error_analysis.ipynb

## How to Run

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the notebooks in order (01 → 05). Fine-tuning (Step 4) requires a GPU —
this project was trained on a free Google Colab T4 GPU; CPU-only machines
will be impractically slow for this step.

## Limitations

- Test/validation sets are small (484 sentences each), so individual metric
  values carry some sampling noise.
- FinBERT was originally fine-tuned on Financial PhraseBank itself, so this
  project demonstrates the fine-tuning pipeline on a clean, reproducible
  split rather than zero-shot transfer to an unseen domain.
- No hyperparameter search was performed; standard BERT fine-tuning defaults
  were used deliberately, to keep the project's scope controlled.

## Resume Summary

**Financial Sentiment Classification using FinBERT**
- Fine-tuned a pretrained FinBERT Transformer for 3-class financial
  sentiment classification on Financial PhraseBank, achieving 89.3%
  accuracy / 88.0% macro F1 on a held-out test set — a 13-point macro F1
  improvement over a TF-IDF + Logistic Regression baseline (75.4%).
- Built a leak-free, stratified data pipeline (deduplication, conflict
  removal, verified train/val/test separation) and implemented Transformer
  tokenization, fine-tuning with early stopping, and macro-F1-based model
  selection.
- Performed error analysis identifying implicit/context-dependent sentiment
  as the dominant failure mode, with targeted checks ruling out negation and
  numeric figures as expected difficulties.

## Files for Deployment
1) app.py

2) requirements_deploy.txt

3) DockerFile

4) .dockerignore
