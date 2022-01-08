
# PRACTICAL BUSINESS ANALYTICS

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

# ************************************************
# scatter_plot() :
#
# This function returns scatterplots of different pairs of variables in the dataset
# 
# INPUT:   dataset - a dataframe
#
# OUTPUT : scatterplots of different pairs of variables
# ************************************************

scatter_plot <- function(dataset) {
  dataset$Class <- as.factor(dataset$Class)
  for (i in seq(1, ncol(dataset) - 1,2)) {
    plot <- ggplot(dataset, aes(x=dataset[,names(dataset)[i]], y=dataset[,names(dataset)[i+1]], color=Class))
    print(plot + geom_point() + labs(x=names(dataset)[i], y=names(dataset)[i+1]))
  }
}

# ************************************************
# density_plot() :
#
# This function returns the density plots of different pairs of variables in the dataset
# 
# INPUT:   dataset - a dataframe
#
# OUTPUT : density plots of different pairs of variables
# ************************************************

density_plot <- function(dataset) {
  dataset$Class <- as.factor(dataset$Class)
  for (i in seq(2,ncol(dataset),2)) {
    plot <- ggplot(dataset, aes(x=dataset[,names(dataset)[i]], group=Class, fill=Class))
    print(plot + geom_density(alpha=0.2) + labs(x=names(dataset)[i]))
  }
}

# ************************************************
# model_metrics() :
#
# This function computes the evaluation metrics for our  models.
# 
# INPUT:   model object,
#          test/validation dataframe
#          Threshold - a constant
#          type - a string - indicating the model type
#
# OUTPUT : a dataframe containing the ROC AUC Score, accuracy, Specificity, Recall,F1-Score, Precision, Confusion matrix
#          ROC Curve
# ************************************************

model_metrics <- function(type, model = NULL, test_df, threshold){
  model_types = c("lm", "rf", "xgb", "svm", "mlp", "knn")
  if (!type %in% model_types) {
    stop("Invalid entry for model type. We support ",paste(model_types, collapse=","))
  }
  
  # probabilities predicted by the model
  # targert variable corresponding to the test dataset
  test_output <- test_df[, TARGET]
  
  if (type == "knn") {
    pred_class <- model
    type  = "K-Nearest Neigbhours"
  } else {
    # input variable names
    predictors <- names(test_df)[names(test_df) != TARGET]
    
    # convert the probabilities to classes based on the defined threshold
    if(type == "rf"){
      probs <- predict(model, newdata=test_df[, INPUT_FEATURES], type='prob')[,2]
      type  = "Random Forest"
    } else if (type == "lm") {
      probs <- predict(model, newdata=test_df[, INPUT_FEATURES], type='response')
      type  = "Logistic Regression"
    } else if (type == "xgb" | type == "mlp") {
      probs <- predict(model, newdata=as.matrix(test_df[, INPUT_FEATURES]))
      if  (type == "xgb") {
        type  = "XGBoost"
      } else {
        type  = "MultiLayer Perceptron"
      }
    } else if (type == "svm") {
      probs <- predict(model, newdata=test_df[, INPUT_FEATURES], probability=TRUE)
      type  = "Support vector machine"
    } else {
      probs <- predict(model, newdata=test_df[, INPUT_FEATURES], type='prob')
    }
    pred_class <- ifelse(probs < threshold, 0, 1)
  }   
  
  metrics <- measureit(score = as.integer(pred_class), class = as.integer(test_output), 
                       measure = c("ACC", "SENS", "FSCR","PREC","SPEC"))
  metrics2 <- rocit(score = as.integer(pred_class), class = as.integer(test_output), negref = 1, 
                    method = "bin")
  
  print('CONFUSION MATRIX')
  print(table(Predictions=pred_class, Targets=test_output))
  
  print("*********************************************************************************************")

  # a dataframe containing evaluation metrics
  df <- data.frame(model_name=type, ROC_AUC_Score=metrics2$AUC, Specificity=metrics$SPEC[2],
                   Accuracy=metrics$ACC[2], Sensitivity_Recall=metrics$SENS[2], f_score=metrics$FSCR[2], 
                   Precision=metrics$PREC[2])
  
  print(df)
  
  # displays the ROC AUC Curve
  plot(metrics2, YIndex = F, values = F, col = c(2,4) )
  title(paste(type, 'ROC Curve', collapse = ' '))
}    

# ************************************************
# correlation_plot() : Returns correlation matrix
#
# INPUT: A dataset - dataframe in CSV
# 
# OUTPUT : Correlation matrix 
# 
# ************************************************

 correlation_plot <- function(dataset){
  # This returns the number of numeric and categorical variables in the dataset
  num_cols <- names(dataset)[sapply(dataset, is.numeric)]
  cat_cols <- names(dataset)[sapply(dataset, is.character)]
  print(paste("There are ", length(num_cols), " Numeric columns"))
  print(paste("There are ", length(cat_cols), " Symbolic columns"))
  
  # Outputs the number of null values in the dataset
  num_nulls = sum(is.na(dataset))
  print(paste("There are ", num_nulls, "missing values in the dataset"))
  
  # Number of fraudulant transactions
  print(paste("There are ", sum(dataset[,TARGET] == 1), "fraudulent transactions out of ", nrow(dataset), " oberservations"))
  
  
  # Displays a correlation plot for all features
  # options(repr.plot.width=25, repr.plot.height=8)
  cor_df <- cor(dataset, method='spearman')
  corrplot(cor_df, method='color')
 }
 
 
 # ************************************************
 # isFALSE : This is a base function in R but was only introduced in R 3.5 and the RStudio in the lab computer is
 #           3.4.4 hence svm() outputs an error message without this function. There is an workaround on Github which 
 #           has been mentioned in the reference below.
 # 
 # Reference: https://github.com/r-spatial/mapview/issues/177
 # 
 # ************************************************
 
 isFALSE <- function(x){
   if (getRversion() >= 3.5) {
     isFALSE(x)
   } else {
     is.logical(x) && length(x) == 1L && !is.na(x) && !x
   }
 }
 

 