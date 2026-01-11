#First we need to install our packages that will be used to help make our data modifiable.

install.packages("tidyverse")
install.packages("caret")
install.packages("randomForest")
install.packages("pROC")

#We now need to load the packages we have installed, in order to use them to prepare our data and create our prediction models, let’s start with the packages that will be used for the regression model.

library("pROC")
library(dplyr)
library(caret)

#Now load the dataset, this has a very long name which can cost a lot of time when trying to code further, so let’s change it to something shorter but still relevant - “hits”.

hits <- billboard_24years_lyrics_spotify

#Now we can start to create our Regression model, this will be based on the “hits” data, so let’s call it “hits_model” and remove the missing data.

hits_model <- hits %>%
  mutate(across(where(is.character), ~na_if(.x, "N/A"))) %>%
  na.omit()

#In order to keep only the variables relevant to us, we need to clean our data and remove any “noisy data” that could interfere with our results. So let's create a new data section, with these variables removed, named “hits_model_clean”.

hits_model_clean <- hits_model %>%
  select(
    ranking,
    danceability,
    energy,
    tempo,
    valence,
    loudness,
  )

#The ranking category is our “dependent variable” so we need to make sure that this variable is set as a numeric value…

hits_model_clean$ranking_num <- as.numeric(as.character(hits_model_clean$ranking))
hits_model_clean$ranking_num <= 20

#but also that R understands this variable is categorical, meaning its order matters.

hits_model_clean$ranking <- as.ordered(hits_model_clean$ranking)

#We can now check the order of rankings to ensure songs at each ranking appear and no data is missing.

levels(hits_model_clean$ranking)

#Currently we have a “ranking” variable, numbering songs from 1–100. We can redefine these rankings as TRUE or FALSE

hits_model_clean$hit <- hits_model_clean$ranking <= 20
hits_model_clean$hit <- factor(hits_model_clean$hit, levels = c(FALSE, TRUE))

#Now that our data has been filtered and our “dependent variable” has been determined if a song is a Hit (TRUE) or not a Hit (FALSE) we can start to build our “regression model”. Using categories available we can:
# 1. Learn what makes a hit
# 2. Predict a continuous “hit score” for new songs 
#This will create a learning pathway for our prediction model, lets call this “hit_model”


hit_model <- glm(
  hit ~ danceability + energy + tempo + valence + loudness,
  data = hits_model_clean,
  family = binomial
)

#Lets get a view of this model

summary(hit_model)

#To use this model for predictions we need to convert the coefficients to odds ratios:

exp(coef(hit_model))

#So far we can see there is a decrease in score between the “Null Deviance” (model with no predictors) and the “Residual Deviance” (model with predictors) and we have an AIC score of 524.05, this will be useful later when comparing with our “random forest” model.

#We can also see from the p-values for each of our predictors (coefficients) that valence, danceability and tempo are significant in our prediction model compared to loudness and energy.

#Now we need to train our model to make predictions


hits_model_clean$hit <- as.factor(hits_model_clean$hit)

#We should split data into train/test sets

set.seed(123)
train_index <- createDataPartition(hits_model_clean$hit, p = 0.8, list = FALSE)
train_data <- hits_model_clean[train_index, ]
test_data  <- hits_model_clean[-train_index, ]

#create our learning model using the training data, this will be called “glm_model”.

glm_model <- glm(
  hit ~ danceability + energy + tempo + valence + loudness,
  family = binomial,
  data = train_data
)

#Produce possibility prediction based on test_data

glm_probs <- predict(glm_model, newdata = test_data, type = "response")

#We can use the results to compute the performance…

roc_glm <- roc(test_data$hit, glm_probs)

#And view these by plotting an ROC curve graph, as this will help compare prediction models we will title this graph "ROC Curve: Logistic Regression vs Random Forest".

plot(roc_glm, col = "blue", lwd = 2, main = "ROC Curve: Logistic Regression vs Random Forest")
abline(a=0, b=1, lty=2, col="gray")

# Print AUC - AUC tells us how well the model discriminates hits from non-hits (1 = perfect, 0.5 = random guessing).

auc_glm <- auc(roc_glm)
print(paste("Logistic Regression AUC:", round(auc_glm, 3)))

#Now we can create our random forest prediction model in order to make a comparison, this will be our “rf_model”.

library(randomForest)
set.seed(123)
rf_model <- randomForest(
  hit ~ danceability + energy + tempo + valence + loudness,
  data = train_data,
  ntree = 500,
  importance = TRUE
)

#Using the same test data for predicting probabilities for ROC (type="prob")

rf_probs <- predict(rf_model, newdata = test_data, type = "prob")[, "TRUE"]
roc_rf <- roc(test_data$hit, rf_probs)

#Ensuring this result is added to the same plot for comparison

lines(roc_rf, col = "red", lwd = 2)
legend("bottomright", legend=c("Logistic Regression", "Random Forest"),
       col=c("blue","red"), lwd=2)

# Print AUC for random forest will be named “aur_rf” - AUC tells us how well the model discriminates hits from non-hits (1 = perfect, 0.5 = random guessing).

auc_rf <- auc(roc_rf)
print(paste("Random Forest AUC:", round(auc_rf, 3)))

#We can also double check which features are most important for predicting hits.

importance(rf_model)
varImpPlot(rf_model)
hits_model_clean <- hits_model %>%
  select(
    ranking,
    danceability,
    tempo,
    valence,
  )
hits_model_clean$ranking_num <- as.numeric(as.character(hits_model_clean$ranking))
hits_model_clean$ranking_num <= 20
hits_model_clean$ranking <- as.ordered(hits_model_clean$ranking)
levels(hits_model_clean$ranking)
hits_model_clean$hit <- hits_model_clean$ranking <= 20
hits_model_clean$hit <- factor(hits_model_clean$hit, levels = c(FALSE, TRUE))
hit_model <- glm(
  hit ~ danceability + tempo + valence,
  data = hits_model_clean,
  family = binomial
)
summary(hit_model)
exp(coef(hit_model))

#Remove the 2 insignificant prediction factors to see impact on the models accuracy.

hits_model_clean$hit <- as.factor(hits_model_clean$hit)
set.seed(123)
train_index <- createDataPartition(hits_model_clean$hit, p = 0.8, list = FALSE)
train_data <- hits_model_clean[train_index, ]
test_data  <- hits_model_clean[-train_index, ]
glm_model <- glm(
  hit ~ danceability + tempo + valence,
  family = binomial,
  data = train_data
)
glm_probs <- predict(glm_model, newdata = test_data, type = "response")
roc_glm <- roc(test_data$hit, glm_probs)
plot(roc_glm, col = "blue", lwd = 2, main = "ROC Curve: Logistic Regression vs Random Forest")
abline(a=0, b=1, lty=2, col="gray")
auc_glm <- auc(roc_glm)
print(paste("Logistic Regression AUC:", round(auc_glm, 3)))
library(randomForest)
set.seed(123)
rf_model <- randomForest(
  hit ~ danceability + tempo + valence,
  data = train_data,
  ntree = 500,
  importance = TRUE
)
rf_probs <- predict(rf_model, newdata = test_data, type = "prob")[, "TRUE"]
roc_rf <- roc(test_data$hit, rf_probs)
lines(roc_rf, col = "red", lwd = 2)
legend("bottomright", legend=c("Logistic Regression", "Random Forest"),
       col=c("blue","red"), lwd=2)
auc_rf <- auc(roc_rf)
print(paste("Random Forest AUC:", round(auc_rf, 3)))
importance(rf_model)
varImpPlot(rf_model)
