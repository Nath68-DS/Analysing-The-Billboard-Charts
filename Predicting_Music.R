install.packages("tidyverse")
install.packages("caret")
install.packages("randomForest")
install.packages("pROC")
library("pROC")
library(dplyr)
library(caret)
hits <- billboard_24years_lyrics_spotify
hits_model <- hits %>%
  mutate(across(where(is.character), ~na_if(.x, "N/A"))) %>%
  na.omit()
hits_model_clean <- hits_model %>%
  select(
    ranking,
    danceability,
    energy,
    tempo,
    valence,
    loudness,
  )
hits_model_clean$ranking_num <- as.numeric(as.character(hits_model_clean$ranking))
hits_model_clean$ranking_num <= 20
hits_model_clean$ranking <- as.ordered(hits_model_clean$ranking)
levels(hits_model_clean$ranking)
hits_model_clean$hit <- hits_model_clean$ranking <= 20
hits_model_clean$hit <- factor(hits_model_clean$hit, levels = c(FALSE, TRUE))
hit_model <- glm(
  hit ~ danceability + energy + tempo + valence + loudness,
  data = hits_model_clean,
  family = binomial
)
summary(hit_model)
exp(coef(hit_model))
hits_model_clean$hit <- as.factor(hits_model_clean$hit)
set.seed(123)
train_index <- createDataPartition(hits_model_clean$hit, p = 0.8, list = FALSE)
train_data <- hits_model_clean[train_index, ]
test_data  <- hits_model_clean[-train_index, ]
glm_model <- glm(
  hit ~ danceability + energy + tempo + valence + loudness,
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
  hit ~ danceability + energy + tempo + valence + loudness,
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
