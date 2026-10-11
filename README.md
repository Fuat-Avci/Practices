# Semi-Supervised Learning for Cervical Cancer Classification Using miRNA Expression Data

A beginner-friendly, reproducible R project comparing **supervised learning** with **single-step self-training** for binary classification of cervical cancer samples using miRNA expression features.

> **Project status:** Supervised and single-step self-training experiments (10 repetitions) are complete. Confidence-based iterative self-training is planned and has **not yet been evaluated**.

## Project overview

![Overview of the dataset, supervised and single-step self-training workflows, four classification algorithms, and ten-repetition accuracy results](figures/Semi-supervised_learning.png)

*Figure 1. Overview of the completed study. The sample counts in the split diagram are schematic approximations; the actual stratified split in this run used 42 training and 16 test samples, with 12 training labels hidden (30% rounded down).*

## Research question

**Does adding pseudo-labeled training samples improve classification performance when only a small number of samples have known labels?**

We compare four classification pipelines under the same train/test splits:

- **Decision Tree** (`caret` / `rpart`)
- **Random Forest** (`caret` / `rf`)
- **XGBoost** (`caret` / `xgbTree`)
- **PCA + Linear Discriminant Analysis** (`caret` / `lda`)

## Dataset

The analysis uses a **cervical miRNA expression dataset** consisting of:

| Property | Value |
|---|---|
| Samples | 58 |
| miRNA expression features | 714 |
| Target classes | Normal and Tumor |
| Machine-learning task | Binary classification |

The working R script imports a local tabular dataset, transposes it to **samples C features**, and creates a `patient_status` factor. In the current workflow, the first 29 samples are assigned `Normal` and the next 29 `Tumor`; **this ordering should be checked against the original dataset metadata before biological interpretation or public reuse**.

> The dataset is small relative to the number of features, so estimated model performance may be highly sensitive to the data split.

## Experimental design

For **each of 10 repetitions**:

1. Create a **stratified 70% training / 30% test split** using `caret::createDataPartition`.
2. Randomly hide **30% of the training labels**. These samples act as the unlabeled pool; their true labels are not used for model training.
3. Train each **supervised baseline** using only the remaining labeled samples.
4. For **single-step self-training**, use the baseline model to predict a pseudo-label for **every** unlabeled training sample.
5. Add the pseudo-labeled samples to the labeled training set and train a new model once.
6. Evaluate both versions on the **same held-out test set**; record test accuracy.
7. Report the **mean** and **sample standard deviation** of test accuracy across the 10 repetitions.

Each repetition uses seed `918 + i` for the data split and label masking. The same split is shared by all algorithms within a repetition. The reported results use `caret`'s default bootstrap resampling for model training/tuning, with `set.seed(123)` before fitting each model.

### Learning strategies

```text
                       Original dataset
                              |
                      Train / Test split
                              |
                     Hide 30% train labels
                              |
                  +-----------+-----------+
                  |                       |
             Labeled data           Unlabeled data
                  |                       |
                  +--> Supervised model --+
                           |              |
                           |       Predict pseudo-labels
                           |              |
                    Test evaluation       v
                                    Combine labeled and
                                     pseudo-labeled data
                                            |
                                     Retrain model once
                                            |
                                      Test evaluation
```

**Supervised learning:** Fit on genuinely labeled samples only.

**Single-step self-training:** Fit on labeled samples, assign pseudo-labels to all unlabeled training samples, combine the two groups, then fit once more. **No confidence threshold was used** in this experiment.

### Special handling for LDA

The number of features (714) greatly exceeds the number of labeled training samples. Therefore, LDA was preceded by training-only preprocessing:

```r
preProcess = c("zv", "center", "scale", "pca")
preProcOptions = list(pcaComp = 5)
```

This removes zero-variance features, centers/scales the others, and reduces the predictor space to **five principal components**. During prediction, the PCA transformation learned from the corresponding training data is applied to the test samples; the test set is **not used to fit PCA**. Five components were fixed as an initial exploratory choice, not selected by test accuracy.

## Results: 10 repeated experiments

The table contains **held-out test accuracy**, averaged over 10 repetitionsb
