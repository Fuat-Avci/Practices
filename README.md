# Semi-Supervised Learning for Cervical Cancer Classification Using miRNA Expression Data

A beginner-friendly, reproducible R project comparing **supervised learning** with **single-step self-training** for binary classification of cervical cancer samples using miRNA expression features.

> **Project status:** The supervised and single-step self-training experiments are complete. Iterative, confidence-based self-training is planned but has **not yet been evaluated**.

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

The working R script imports a local tabular dataset, transposes it to **samples × features**, and creates a `patient_status` factor. In the current workflow, the first 29 samples are assigned `Normal` and the next 29 `Tumor`; **this ordering should be checked against the original dataset metadata before biological interpretation or public reuse**.

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

The table contains **held-out test accuracy**, averaged over 10 repetitions—not the internal bootstrap performance reported when printing a `caret` model.

| Algorithm | Supervised mean accuracy | Single-step self-training mean accuracy | Supervised SD | Self-training SD |
|---|---:|---:|---:|---:|
| Decision Tree | 68.13% | 68.13% | 0.1040 | 0.1040 |
| Random Forest | 78.13% | 75.63% | 0.0943 | 0.1195 |
| XGBoost | 84.38% | 78.75% | 0.0846 | 0.1149 |
| **PCA + LDA** | **85.00%** | **80.63%** | **0.0791** | **0.0688** |

![Study workflow and ten-repetition accuracy comparison](figures/study_overview.png)

### Interpretation

- **PCA + LDA** achieved the highest mean test accuracy in the supervised setting (**85.00%**), followed closely by **XGBoost** (**84.38%**).
- **Single-step self-training did not improve mean test accuracy** for any of the four algorithms in these repetitions.
- Including **all** pseudo-labels, including uncertain predictions, may introduce label noise; this is a **possible explanation**, not a confirmed causal finding.
- The results are **exploratory**: there are only 58 samples, repeated test splits overlap, and 10 repetitions do not establish statistical significance or external generalizability.

## Technologies

- **Language:** R
- **Libraries:** `caret`, `dplyr`, `randomForest`, `xgboost`, `MLSeq`
- **Classifier implementations:** `rpart`, `rf`, `xgbTree`, `lda` through `caret`
- **Validation:** Repeated stratified hold-out evaluation (10 repetitions); internal `caret` bootstrap model resampling

> **Compatibility note:** The XGBoost experiment was run with an older 1.7.x XGBoost release due to compatibility issues observed with `caret` and a newer XGBoost release in the local R environment. Record exact package versions with `sessionInfo()` for full reproducibility.

## Run the project

1. Install the required R packages.
2. Obtain the dataset from its legitimate source and update the `file_path` variable in the R script to point to your local file.
3. Run the preprocessing code to create `full_data` and `patient_status`.
4. Run the supervised and single-step self-training experiment with 10 repetitions.
5. Print `comparison_10` to inspect means and standard deviations.

**Note:** The working script currently uses a Windows-specific absolute data path. Replace it with a relative path before sharing a fully reproducible public repository. Include dataset provenance/licensing and check the class-label ordering against the source metadata.

## Planned work

- Implement **iterative self-training** with confidence-based pseudo-label selection and a stopping rule.
- Evaluate iterative self-training with **the same 10 data splits** for comparison with the single-step method.
- Compare paired run-level outcomes, confusion-matrix metrics, and sensitivity to the pseudo-label confidence threshold.
- Improve reproducibility by recording dependencies and making dataset loading independent of local paths.

## Limitations

This is a small, exploratory educational study, **not a clinically validated cervical cancer diagnostic system**. Class labels, sample provenance, potential batch effects, and data licensing need to be verified before drawing biological or clinical conclusions. The measured accuracies are specific to the current dataset and evaluation design.
