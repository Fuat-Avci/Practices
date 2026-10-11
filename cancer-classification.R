# libraries

library(caret)
library(MLSeq)
library(dplyr)
library(randomForest)
library(xgboost)

# STEP 1 : read data 

file_path <- "C:/Users/HP/Desktop/MachineLearning/Cervical.txt"

cervical_data <- read.table(file_path, header = TRUE)


# STEP 2: Cervical data preparation  (transposed) 

transposed_data <- t(cervical_data)

full_data <- as.data.frame(transposed_data)

full_data <- full_data %>%
  mutate(
    patient_status = factor(c(rep("Normal", 29),
                              rep("Tumor", 29)))
  )


### STEP 3: Train/Test split    

set.seed(918)

train_index <- createDataPartition(
  full_data$patient_status,
  p = 0.7,
  list = FALSE
)

train_data <- full_data[train_index, ]
test_data <- full_data[-train_index, ]

### test data will be used for evaluating the models at the final stage.



## STEP 4 :   NAs are assigned  in the train data 


# Hide 15% of training labels

semi_train_data <- train_data

unlabeled_indices <- sample(
  1:nrow(semi_train_data),
  floor(nrow(semi_train_data) * 0.30)
)

semi_train_data$patient_status[unlabeled_indices] <- NA


# Separate labeled and unlabeled data

unlabeled_data <- dplyr::filter(
  semi_train_data,
  is.na(patient_status)
)

labeled_data <- dplyr::filter(
  semi_train_data,
  !is.na(patient_status)
)


### DECISION TREE MODEL  (supervised Learning) #########


set.seed(123)

dt_supervised <- train(
  x = dplyr::select(labeled_data, -patient_status),
  y = labeled_data$patient_status,
  method = "rpart"
)

## SUPERVISED MODEL EVALUATION ###


# Supervised Decision Tree Predictions

dt_supervised_pred <- predict(
  dt_supervised,
  newdata = test_data
)

# Confusion Matrix

dt_supervised_cm <- confusionMatrix(
  dt_supervised_pred,
  test_data$patient_status,
  positive = "Tumor"
)

dt_supervised_cm


###  DECISION TREE  SELF-TRAINING


# Predict Unlabeled Data

dt_pseudo_labels <- predict(
  dt_supervised,
  newdata = unlabeled_data
)

# Assign Pseudo-labels

dt_unlabeled_data <- unlabeled_data

dt_unlabeled_data$patient_status <- dt_pseudo_labels

# Combine Labeled and Pseudo-labeled Data

dt_combined_data <- rbind(
  labeled_data,
  dt_unlabeled_data
)



### Retraining Decision tree ( pseudo-labels are added in the upper code )

# Decision Tree - Self-training

set.seed(123)

dt_self_training <- train(
  x = dplyr::select(dt_combined_data, -patient_status),
  y = dt_combined_data$patient_status,
  method = "rpart"
)


### Self training model evaluation


# Self-training Predictions

dt_self_pred <- predict(
  dt_self_training,
  newdata = test_data
)

# Confusion Matrix

dt_self_cm <- confusionMatrix(
  dt_self_pred,
  test_data$patient_status,
  positive = "Tumor"
)

dt_self_cm


# Decision Tree Comparison  for supervised and self-trainimg

dt_comparison <- data.frame(
  Model = c("Supervised", "Self-training"),
  Accuracy = c(
    dt_supervised_cm$overall["Accuracy"],
    dt_self_cm$overall["Accuracy"]
  )
)

print(dt_comparison)




####### RANDOM FOREST #######



### Random forest - supervised learning ####

set.seed(123)

rf_supervised <- train(
  x = dplyr::select(labeled_data, -patient_status),
  y = labeled_data$patient_status,
  method = "rf"
)


# Supervised Random Forest Predictions

rf_supervised_pred <- predict(
  rf_supervised,
  newdata = test_data
)

# Confusion Matrix

rf_supervised_cm <- confusionMatrix(
  rf_supervised_pred,
  test_data$patient_status,
  positive = "Tumor"
)

rf_supervised_cm



#### Rndom forest  Pseudo Labeling  #########

# Predict Unlabeled Data

rf_pseudo_labels <- predict(
  rf_supervised,
  newdata = unlabeled_data
)

# Assign Pseudo-labels

rf_unlabeled_data <- unlabeled_data

rf_unlabeled_data$patient_status <- rf_pseudo_labels

# Combine Labeled and Pseudo-labeled Data

rf_combined_data <- rbind(
  labeled_data,
  rf_unlabeled_data
)


###  Retrain  Random Forest  #####


# Random Forest - Self-training

set.seed(123)

rf_self_training <- train(
  x = dplyr::select(rf_combined_data, -patient_status),
  y = rf_combined_data$patient_status,
  method = "rf"
)


### Self Trainign Model Evaluation 


# Self-training Predictions

rf_self_pred <- predict(
  rf_self_training,
  newdata = test_data
)

# Confusion Matrix

rf_self_cm <- confusionMatrix(
  rf_self_pred,
  test_data$patient_status,
  positive = "Tumor"
)

rf_self_cm


### Random  forest  supervised vs self-training  model comparision #####

# 

rf_comparison <- data.frame(
  Model = c("Supervised", "Self-training"),
  Accuracy = c(
    rf_supervised_cm$overall["Accuracy"],
    rf_self_cm$overall["Accuracy"]
  )
)

print(rf_comparison)



######  DESICION TREEE VS RANDOM FOREST 

# Decision Tree vs Random Forest

model_comparison <- data.frame(
  Algorithm = c(
    "Decision Tree",
    "Decision Tree",
    "Random Forest",
    "Random Forest"
  ),
  Learning = c(
    "Supervised",
    "Self-training",
    "Supervised",
    "Self-training"
  ),
  Accuracy = c(
    dt_supervised_cm$overall["Accuracy"],
    dt_self_cm$overall["Accuracy"],
    rf_supervised_cm$overall["Accuracy"],
    rf_self_cm$overall["Accuracy"]
  )
)

print(model_comparison)






#######   XGBOOST ########


# XGBoost - Supervised Learning ####3

set.seed(123)

xgb_supervised <- train(
  x = dplyr::select(labeled_data, -patient_status),
  y = labeled_data$patient_status,
  method = "xgbTree"
)


##### Model evaluation ####

# Supervised XGBoost Predictions

xgb_supervised_pred <- predict(
  xgb_supervised,
  newdata = test_data
)

# Confusion Matrix

xgb_supervised_cm <- confusionMatrix(
  xgb_supervised_pred,
  test_data$patient_status,
  positive = "Tumor"
)

xgb_supervised_cm


### XGBoost Pseudo-labeling  ####


# Predict Unlabeled Data

xgb_pseudo_labels <- predict(
  xgb_supervised,
  newdata = unlabeled_data
)

# Assign Pseudo-labels

xgb_unlabeled_data <- unlabeled_data

xgb_unlabeled_data$patient_status <- xgb_pseudo_labels

# Combine Labeled and Pseudo-labeled Data

xgb_combined_data <- rbind(
  labeled_data,
  xgb_unlabeled_data
)


 
# XGBoost - Self-training  Model   ####
 
set.seed(123)

xgb_self_training <- train(
  x = dplyr::select(xgb_combined_data, -patient_status),
  y = xgb_combined_data$patient_status,
  method = "xgbTree"
)


#  XGBoost - Self-training  Model Evaluation  ####


# Self-training Predictions

xgb_self_pred <- predict(
  xgb_self_training,
  newdata = test_data
)

# Confusion Matrix

xgb_self_cm <- confusionMatrix(
  xgb_self_pred,
  test_data$patient_status,
  positive = "Tumor"
)

xgb_self_cm



#### XGBOOST SUPVERVISED VS SELF-LEARNING   COMPARISION


# XGBoost Comparison

xgb_comparison <- data.frame(
  Model = c("Supervised", "Self-training"),
  Accuracy = c(
    xgb_supervised_cm$overall["Accuracy"],
    xgb_self_cm$overall["Accuracy"]
  )
)

print(xgb_comparison)



######  DT  VS RF VS XGBOOST ######


# Model Comparison

model_comparison <- data.frame(
  Algorithm = rep(
    c("Decision Tree", "Random Forest", "XGBoost"),
    each = 2
  ),
  Learning = rep(
    c("Supervised", "Self-training"),
    times = 3
  ),
  Accuracy = c(
    dt_supervised_cm$overall["Accuracy"],
    dt_self_cm$overall["Accuracy"],
    rf_supervised_cm$overall["Accuracy"],
    rf_self_cm$overall["Accuracy"],
    xgb_supervised_cm$overall["Accuracy"],
    xgb_self_cm$overall["Accuracy"]
  )
)

print(model_comparison)



#### LINEAR DISCRIMINANT ANALYSIS #####



# LDA Supervised Learning

set.seed(123)

lda_supervised <- train(
  x = dplyr::select(labeled_data, -patient_status),
  y = labeled_data$patient_status,
  method = "lda",
  preProcess = c("zv", "center", "scale", "pca"),
  preProcOptions = list(pcaComp = 5)
)

lda_supervised


#### LDA Supervised Model Evaluation ####

# Apply the trained PCA transformation to test data

lda_test_pca <- predict(
  lda_supervised$preProcess,
  newdata = dplyr::select(test_data, -patient_status)
)

# Predict with the trained LDA model

lda_supervised_pred <- predict(
  lda_supervised$finalModel,
  newdata = lda_test_pca
)$class

# Confusion Matrix

lda_supervised_cm <- confusionMatrix(
  lda_supervised_pred,
  test_data$patient_status,
  positive = "Tumor"
)

lda_supervised_cm


#### LDA SELF TRAINING ####

# LDA - Pseudo-labeling

lda_pseudo_labels <- predict(
  lda_supervised$finalModel,
  newdata = predict(
    lda_supervised$preProcess,
    newdata = dplyr::select(unlabeled_data, -patient_status)
  )
)$class

lda_unlabeled_data <- unlabeled_data

lda_unlabeled_data$patient_status <- lda_pseudo_labels

lda_combined_data <- rbind(
  labeled_data,
  lda_unlabeled_data
)


### REtrain the model #####


# LDA - Self-training

set.seed(123)

lda_self_training <- train(
  x = dplyr::select(lda_combined_data, -patient_status),
  y = lda_combined_data$patient_status,
  method = "lda",
  preProcess = c("zv", "center", "scale", "pca"),
  preProcOptions = list(pcaComp = 5)
)


### Evaluate the Self trained LDA model 

# LDA - Self-training Predictions

# LDA - Self-training Predictions

lda_test_pca_self <- predict(
  lda_self_training$preProcess,
  newdata = dplyr::select(test_data, -patient_status)
)

lda_self_pred <- predict(
  lda_self_training$finalModel,
  newdata = lda_test_pca_self
)$class

lda_self_cm <- confusionMatrix(
  lda_self_pred,
  test_data$patient_status,
  positive = "Tumor"
)

lda_self_cm



### COMPARE  SUPERVISED VS SELF TRAINED   LDA 

# LDA Comparison

lda_comparison <- data.frame(
  Model = c("Supervised", "Self-training"),
  Accuracy = c(
    lda_supervised_cm$overall["Accuracy"],
    lda_self_cm$overall["Accuracy"]
  )
)

print(lda_comparison)



#####  10 repeat  Single-step Self-training #####

performanslar <- numeric(5)





### store model accuracies
# Store model accuracies

dt_supervised_acc <- numeric(10)
dt_self_acc <- numeric(10)

rf_supervised_acc <- numeric(10)
rf_self_acc <- numeric(10)

xgb_supervised_acc <- numeric(10)
xgb_self_acc <- numeric(10)

lda_supervised_acc <- numeric(10)
lda_self_acc <- numeric(10)


### 10 repeat 



# 10 Repeated Experiments - Single-step Self-training

for (i in 1:10) {
  
  set.seed(918 + i)
  
  # Train/Test Split
  
  train_index <- createDataPartition(
    full_data$patient_status,
    p = 0.7,
    list = FALSE
  )
  
  train_data <- full_data[train_index, ]
  test_data <- full_data[-train_index, ]
  
  
  # Labeled/Unlabeled Split (30%)
  
  semi_train_data <- train_data
  
  unlabeled_indices <- sample(
    1:nrow(semi_train_data),
    floor(nrow(semi_train_data) * 0.30)
  )
  
  semi_train_data$patient_status[unlabeled_indices] <- NA
  
  unlabeled_data <- dplyr::filter(
    semi_train_data,
    is.na(patient_status)
  )
  
  labeled_data <- dplyr::filter(
    semi_train_data,
    !is.na(patient_status)
  )
  
  
  # --------------------------
  # 1. Decision Tree
  # --------------------------
  
  set.seed(123)
  
  dt_supervised <- train(
    x = dplyr::select(labeled_data, -patient_status),
    y = labeled_data$patient_status,
    method = "rpart"
  )
  
  dt_supervised_pred <- predict(
    dt_supervised,
    newdata = test_data
  )
  
  dt_supervised_acc[i] <- confusionMatrix(
    dt_supervised_pred,
    test_data$patient_status
  )$overall["Accuracy"]
  
  
  # Decision Tree - Self-training
  
  dt_pseudo_labels <- predict(
    dt_supervised,
    newdata = unlabeled_data
  )
  
  dt_unlabeled_data <- unlabeled_data
  dt_unlabeled_data$patient_status <- dt_pseudo_labels
  
  dt_combined_data <- rbind(
    labeled_data,
    dt_unlabeled_data
  )
  
  set.seed(123)
  
  dt_self_training <- train(
    x = dplyr::select(dt_combined_data, -patient_status),
    y = dt_combined_data$patient_status,
    method = "rpart"
  )
  
  dt_self_pred <- predict(
    dt_self_training,
    newdata = test_data
  )
  
  dt_self_acc[i] <- confusionMatrix(
    dt_self_pred,
    test_data$patient_status
  )$overall["Accuracy"]
  
  
  # --------------------------
  # 2. Random Forest
  # --------------------------
  
  set.seed(123)
  
  rf_supervised <- train(
    x = dplyr::select(labeled_data, -patient_status),
    y = labeled_data$patient_status,
    method = "rf"
  )
  
  rf_supervised_pred <- predict(
    rf_supervised,
    newdata = test_data
  )
  
  rf_supervised_acc[i] <- confusionMatrix(
    rf_supervised_pred,
    test_data$patient_status
  )$overall["Accuracy"]
  
  
  # Random Forest - Self-training
  
  rf_pseudo_labels <- predict(
    rf_supervised,
    newdata = unlabeled_data
  )
  
  rf_unlabeled_data <- unlabeled_data
  rf_unlabeled_data$patient_status <- rf_pseudo_labels
  
  rf_combined_data <- rbind(
    labeled_data,
    rf_unlabeled_data
  )
  
  set.seed(123)
  
  rf_self_training <- train(
    x = dplyr::select(rf_combined_data, -patient_status),
    y = rf_combined_data$patient_status,
    method = "rf"
  )
  
  rf_self_pred <- predict(
    rf_self_training,
    newdata = test_data
  )
  
  rf_self_acc[i] <- confusionMatrix(
    rf_self_pred,
    test_data$patient_status
  )$overall["Accuracy"]
  
  
  # --------------------------
  # 3. XGBoost
  # --------------------------
  
  set.seed(123)
  
  xgb_supervised <- train(
    x = dplyr::select(labeled_data, -patient_status),
    y = labeled_data$patient_status,
    method = "xgbTree"
  )
  
  xgb_supervised_pred <- predict(
    xgb_supervised,
    newdata = test_data
  )
  
  xgb_supervised_acc[i] <- confusionMatrix(
    xgb_supervised_pred,
    test_data$patient_status
  )$overall["Accuracy"]
  
  
  # XGBoost - Self-training
  
  xgb_pseudo_labels <- predict(
    xgb_supervised,
    newdata = unlabeled_data
  )
  
  xgb_unlabeled_data <- unlabeled_data
  xgb_unlabeled_data$patient_status <- xgb_pseudo_labels
  
  xgb_combined_data <- rbind(
    labeled_data,
    xgb_unlabeled_data
  )
  
  set.seed(123)
  
  xgb_self_training <- train(
    x = dplyr::select(xgb_combined_data, -patient_status),
    y = xgb_combined_data$patient_status,
    method = "xgbTree"
  )
  
  xgb_self_pred <- predict(
    xgb_self_training,
    newdata = test_data
  )
  
  xgb_self_acc[i] <- confusionMatrix(
    xgb_self_pred,
    test_data$patient_status
  )$overall["Accuracy"]
  
  
  # --------------------------
  # 4. PCA + LDA
  # --------------------------
  
  set.seed(123)
  
  lda_supervised <- train(
    x = dplyr::select(labeled_data, -patient_status),
    y = labeled_data$patient_status,
    method = "lda",
    preProcess = c("zv", "center", "scale", "pca"),
    preProcOptions = list(pcaComp = 5)
  )
  
  lda_test_pca <- predict(
    lda_supervised$preProcess,
    newdata = dplyr::select(test_data, -patient_status)
  )
  
  lda_supervised_pred <- predict(
    lda_supervised$finalModel,
    newdata = lda_test_pca
  )$class
  
  lda_supervised_acc[i] <- confusionMatrix(
    lda_supervised_pred,
    test_data$patient_status
  )$overall["Accuracy"]
  
  
  # LDA - Self-training
  
  lda_pseudo_labels <- predict(
    lda_supervised$finalModel,
    newdata = predict(
      lda_supervised$preProcess,
      newdata = dplyr::select(unlabeled_data, -patient_status)
    )
  )$class
  
  lda_unlabeled_data <- unlabeled_data
  lda_unlabeled_data$patient_status <- lda_pseudo_labels
  
  lda_combined_data <- rbind(
    labeled_data,
    lda_unlabeled_data
  )
  
  set.seed(123)
  
  lda_self_training <- train(
    x = dplyr::select(lda_combined_data, -patient_status),
    y = lda_combined_data$patient_status,
    method = "lda",
    preProcess = c("zv", "center", "scale", "pca"),
    preProcOptions = list(pcaComp = 5)
  )
  
  lda_test_pca_self <- predict(
    lda_self_training$preProcess,
    newdata = dplyr::select(test_data, -patient_status)
  )
  
  lda_self_pred <- predict(
    lda_self_training$finalModel,
    newdata = lda_test_pca_self
  )$class
  
  lda_self_acc[i] <- confusionMatrix(
    lda_self_pred,
    test_data$patient_status
  )$overall["Accuracy"]
  
  
  # Display iteration progress
  
  cat("Iteration", i, "completed\n")
  
}



### Average accuracy and standard deviation

# Model Comparison - Mean Accuracy and Standard Deviation

comparison_10 <- data.frame(
  Algorithm = rep(
    c("Decision Tree", "Random Forest", "XGBoost", "PCA + LDA"),
    each = 2
  ),
  
  Learning = rep(
    c("Supervised", "Self-training"),
    times = 4
  ),
  
  Mean_Accuracy = c(
    mean(dt_supervised_acc),
    mean(dt_self_acc),
    mean(rf_supervised_acc),
    mean(rf_self_acc),
    mean(xgb_supervised_acc),
    mean(xgb_self_acc),
    mean(lda_supervised_acc),
    mean(lda_self_acc)
  ),
  
  SD_Accuracy = c(
    sd(dt_supervised_acc),
    sd(dt_self_acc),
    sd(rf_supervised_acc),
    sd(rf_self_acc),
    sd(xgb_supervised_acc),
    sd(xgb_self_acc),
    sd(lda_supervised_acc),
    sd(lda_self_acc)
  )
)

print(comparison_10)

#### ITERATIVE SELF LEARNING WILL BE APPLIED SOON 










