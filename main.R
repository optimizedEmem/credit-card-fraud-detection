# ************************************************
# PRACTICAL BUSINESS ANALYTICS GROUP COURSEWORK

# Ememobong Ekpenyong
# Patrick Oramah
# Judith Etiobhio
# Arjoo Gupta
# Ashwin Baiju
# Ibukun Omojola

# Department of Computer Science 
# University of Surrey
# GUILDFORD
# Surrey GU2 7XH
# *************************************************

# Kindly save all files (creditcard.csv, functions4.R) in a folder and set working directory to the  folder.

# suppress warning
options(warn=-1)

#  clears all objects in "global environment"
rm(list=ls())

# clears the console
cat("\014")  
# *************************************************

# Loads the functions4.R script that contains custom functions required to execute this script.
source("functions.R")

# Global Environment constant variables
# Uppercase is used in this script to identify the global variables

FILEPATH <- "creditcard.csv"       # Name of the filepath for the CSV document
TARGET <- "Class"                  # Target variable
THRESHOLD <- 0.7                   # Threshold for the predicted probabilities. values greater than the threshold
                                   # are converted to 1 (positive case) else converted to 0 (negative case)

HOLDOUT   <- 0.75                  # 75% of the dataset will be used for training 

NUMBER_OF_SECONDS_IN_A_DAY <- 3600
NUMBER_OF_HOURS_PER_DAY <- 24

SEED_VALUE <- 1990                 # used to achieve reproducibility

# Logistic Regression
NUMBER_OF_ITERATIONS <- 500

# Random Forest
NUMBER_OF_TREES <- 100

# K-Nearest Neigbors
NUMBER_OF_NEAREST_NEIGHBORS <- 50

# XGBoost 
NUMBER_OF_ITERS_XGB <- 200

# Multilayer Perceptron 
NUMBER_OF_HIDDEN_LAYERS <- 25
NUMBER_OF_ITERS <- 2000

# ************************************************
# Define and then load the libraries used in this project

# Library from CRAN     Version
# ************************************************

# devtools required to install older versions of pacman
install.packages("devtools")
# Install pacman 0.4.6 
packageurl <- "https://cran.r-project.org/src/contrib/Archive/pacman/pacman_0.4.6.tar.gz"
install.packages(packageurl, repos=NULL, type="source")

# pacman checks if the specified libraries are already installed, if no, installs and loads the libraries
library("pacman")
p_load(char=c("ROSE",                  # used for undersampling the training data
              "smotefamily",           # used for oversampling the training data
              "ROCit",                 # used for calculating the evaluation metrics
              "RSNNS",                 # used for implementing the MLP -  neural network model
              "corrplot",              # used for correlation plot
              "randomForest",          # used for implementing the Random Forest algorithm
              "xgboost",               # used for implementing the XGBoost algorithm
              "class",                 # used for implementing the KNN algorithm
              "e1071",                 # used for implementing the SVM algorithm
              "ggplot2",               # used for different plots and vizualization  
              "Ckmeans.1d.dp"), install=TRUE, character.only=TRUE)

 # reads the CSV document into a dataframe.
df <- read.csv(FILEPATH, stringsAsFactors=FALSE)

# Clears all plots from earlier runs
if(!is.null(dev.list())) dev.off()

# A custom function that displays correlation plot
correlation_plot(df)

# Display scatter plots showing the relationship between features
scatter_plot(df)

# Shows the distribution of fraudulent and non-fraudulent transactions per feature 
density_plot(df)

# convert time to a 24hrs period
df[, 'Hour'] <- (df[,'Time'] / NUMBER_OF_SECONDS_IN_A_DAY) %% NUMBER_OF_HOURS_PER_DAY
# drop the Time feature
df$Time <- NULL

# convert the Class feature to factor variable
df[,TARGET] <- as.factor(df[, TARGET])
# Density plot showing the distribution of the target variable within a 24 hours period
print(ggplot(df, aes(x = Hour, fill = Class, group=Class)) +  geom_density(alpha = 0.4) + 
        scale_x_continuous(limits = c(0, 24), breaks = seq(0, 24, 2)) + 
        labs(title = "Newly created Feature - Hour", x = "Hour", y = "Density", col = "Class") + 
        scale_fill_discrete(labels = c("Non Fraud", "Fraud")))


# normalize the Amount column
df[, 'norm_Amount'] <- scale(df[,'Amount'], center = TRUE, scale=TRUE)
# drop the original Amount column
df$Amount <- NULL

# A variable that stores all the input feature names
INPUT_FEATURES <- names(df)[names(df) != TARGET]

# convert the Class feature (target variable) to factor variable
df[,TARGET] <- as.factor(df[, TARGET])


# spliting the dataset into training and test dataset
# Test dataset to be used for testing the performance of the models
set.seed(SEED_VALUE)
train_ind <- sample(nrow(df), HOLDOUT * nrow(df))
train_df <- df[train_ind, ]

# test_df - unseen data to be used for validating our models
test_df <- df[-train_ind, ]

print("Class distribution of the training observations before undersampling")
print(table(train_df[, TARGET]))

print("Class distribution of the test observations")
print(table(test_df[, TARGET]))

# Undersampling the majority class of the training set to the ratio of 19:1
input_variables <- paste(names(df)[which(names(df) != TARGET)], collapse='+')
formula <- as.formula(paste(TARGET,'~',input_variables))
train_df <- ovun.sample(formula,
                              data=train_df,
                              method='under',
                              N = nrow(train_df[train_df$Class == 1, ]) * 20)$data

print("Class distribution of the training observations after undersampling")
print(table(train_df[, TARGET]))

# SMOTE is a function that performs oversampling
train_df <- SMOTE(X=train_df[,INPUT_FEATURES], target=train_df[, TARGET], K=5, dup_size = 18)$data
train_df[, TARGET] <- as.factor(train_df$class)
train_df[, 'class'] <- NULL

print("Class distribution of the training observations after undersampling and oversampling")
print(table(train_df[, TARGET]))


# Logistic Regression 
lm_model <- glm(formula, data=train_df, family = binomial('logit'), maxit=NUMBER_OF_ITERATIONS)

# Model evaluation function that return ROC AUC Score, accuracy, Specificity, Recall,F1-Score, Precision, 
# Confusion matrix,  ROC Curve          
print("Logistic Regression")
model_metrics(model=lm_model, test_df, THRESHOLD, type="lm")


# random forest
rf_model <- randomForest(formula, data=train_df, ntree=NUMBER_OF_TREES, importance=TRUE, keep.forest=TRUE)

# Feature importance plot for Random Forest
feat_imp <- as.data.frame(rf_model$importance)
feat_imp <- cbind(features = rownames(rf_model$importance), feat_imp)
feat_imp <- feat_imp[order(feat_imp$MeanDecreaseGini, decreasing=T),]
features <- feat_imp[,1]
importance <- feat_imp[,2]
print(ggplot(feat_imp, aes(x=reorder(features, importance), y=importance)) +
  geom_bar(stat="identity", fill="lightpink", colour="brown") +
  coord_flip() + theme_bw(base_size = 10) +
  labs(title="RandomForest with 100 trees", 
       subtitle="Variable importance", x="Features", y="Importance"))

# Model evaluation function that return ROC AUC Score, accuracy, Specificity, Recall,F1-Score, Precision, 
# Confusion matrix,  ROC Curve          
print("Random Forest")
model_metrics(model=rf_model, test_df, THRESHOLD, type="rf")


# XGBoost model
xgb_model <- xgboost(data=as.matrix(train_df[, INPUT_FEATURES]), 
               label=as.numeric(as.character(train_df[, TARGET])), 
               nrounds=NUMBER_OF_ITERS_XGB, verbose=0, objective='binary:logistic',
               eval_metric = 'logloss',
               max.depth = 3)

# Feature importance plot for XGBoost
mx <- xgb.importance(colnames(train_df[, INPUT_FEATURES]), model=xgb_model)
print(xgb.ggplot.importance(mx, rel_to_first = FALSE, xlab = "Relative importance") + ggtitle("XGBoost Feature Importance"))

# Model evaluation function that return ROC AUC Score, accuracy, Specificity, Recall,F1-Score, Precision, 
# Confusion matrix,  ROC Curve          
print("XGBoost")
model_metrics(model=xgb_model, test_df, THRESHOLD, type="xgb")


# KNN model
knn_preds <- knn(train=train_df[, INPUT_FEATURES], test= test_df[,INPUT_FEATURES], 
                 cl=train_df[, TARGET], k=NUMBER_OF_NEAREST_NEIGHBORS, prob=TRUE)

# Model evaluation function that return ROC AUC Score, accuracy, Specificity, Recall,F1-Score, Precision, 
# Confusion matrix,  ROC Curve          
print("K-Nearest Neigbhours")
model_metrics(model=knn_preds, test_df, THRESHOLD, type="knn")


# Multilayer Perceptron model
mlp_model <- mlp(train_df[, INPUT_FEATURES],
                 y=as.numeric(as.character(train_df[, TARGET])),
                 size=c(NUMBER_OF_HIDDEN_LAYERS),maxit=NUMBER_OF_ITERS,learnFunc="Rprop",linOut=TRUE)

# Model evaluation function that return ROC AUC Score, accuracy, Specificity, Recall,F1-Score, Precision, 
# Confusion matrix,  ROC Curve          
print("MultiLayer Perceptron")
model_metrics(model=mlp_model, test_df, THRESHOLD, type="mlp")


# SVM model
train_df[, TARGET] <- as.factor(train_df[, TARGET])
svm_model <- svm(x=train_df[, INPUT_FEATURES],
                 y=as.numeric(as.character(train_df[, TARGET])), 
                 data=train_df, probability=TRUE)

# Model evaluation function that return ROC AUC Score, accuracy, Specificity, Recall,F1-Score, Precision, 
# Confusion matrix,  ROC Curve                          
print("Support vector machine")
model_metrics(model=svm_model, test_df, THRESHOLD, type="svm")












