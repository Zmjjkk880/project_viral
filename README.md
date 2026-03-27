# Viral Video Prediction

This project predicts whether a short video will go viral based on features available at upload time.

We formulate this as a **binary classification task**:

- 1 → viral
- 0 → non-viral

---

# Dataset

We use a public dataset from Kaggle:

https://www.kaggle.com/datasets/tarekmasryo/youtube-shorts-and-tiktok-trends-2025/data/code

This dataset contains short video metadata (e.g., text descriptions, engagement metrics, and categorical attributes), which are used to construct features for viral prediction.

---

# How to Use This Repository

Clone the repository:

```bash
git clone https://github.com/Zmjjkk880/project_viral
cd project_viral
```

---

# Pipeline

The workflow consists of **two steps only**:

1. Preprocessing
2. Model Training

---

# Step 1: Preprocessing

Script: `preprocess.py`

This step:

- Assigns labels based on thresholds
- Generates the processed dataset

You define what "viral" means via thresholds:

```bash
python preprocess.py \
--views-threshold 20000 \
--engagement-threshold 0.06
```

This will produce a dataset like:

```
data/processed/processed_data_20000_0.06.csv
```

---

# Step 2: Model Training

> Note: `train.py` (basic MLP) is deprecated.

We mainly use:

- `train2.py` → 2-modal fusion
- `train3.py` → 3-modal fusion

---

## train2.py (2-modal Fusion)

Modalities:

- Text (BERT embeddings)
- Tabular (numeric + categorical combined)

### Example

```bash
python train2.py \
--token-mixer raw_concat \
--data-path data/processed/processed_data_20000_0.06.csv \
--epochs 50 \
--learning-rate 1e-5 \
--hidden-dim 16
```

---

## train3.py (3-modal Fusion)

Modalities:

- Text
- Numeric
- Categorical

Difference from train2:

- train2: tabular features are treated as a single block
- train3: tabular features are split into numeric and categorical

### Example

```bash
python train3.py \
--token-mixer raw_concat \
--data-path data/processed/processed_data_20000_0.06.csv \
--epochs 50 \
--learning-rate 1e-5 \
--hidden-dim 16
```

---

# Token Mixer Methods

The `--token-mixer` argument controls how different modalities are fused.

## 1. raw_concat

- Directly concatenates all features
- No transformation

```
[text | tabular]
```

---

## 2. projected_concat

- Each modality is projected to the same dimension
- Then concatenated

```
text → proj_dim
num  → proj_dim
cat  → proj_dim
concat
```

Helps align feature scales

---

## 3. weighted_sum

- Each modality is projected
- Model learns a weight for each modality
- Final representation is a weighted sum

```
output = α1 * text + α2 * num + α3 * cat
```

Automatically learns modality importance

---

## 4. attention_pool

- Treats each modality as a token
- Applies attention across modalities
- Uses mean pooling to get final vector

```
[text_token, num_token, cat_token]
→ attention
→ pooled output
```

Captures interactions between modalities

---

# Evaluation Metrics

- ROC-AUC (main metric)
- Accuracy
- Precision / Recall / F1-score

---

# Notes

- Text features use Sentence-BERT (`all-MiniLM-L6-v2`)
- Numeric features are standardized
- Categorical features are one-hot encoded
