rm(list = ls())
gc()

library(magrittr) 
library(tm)
library(SentimentAnalysis)
library(tibble)
library(party)
library(caret)
library(e1071) 
library(fastDummies)
library(caTools)
library("klaR")
library(dplyr)
library(pROC)
library(themis)
library(tidymodels)
library(ROCR)
library(Metrics)
library(xgboost)
library(readr)
library(stringr)
library(car)
options(stringsAsFactors = F) 

setwd("C:\\R\\nmproject")


ffp_train.df <- read.csv("ffp_train_with_sentiment.csv")

########################## DATA_PREP ######################################


ffp_train.df$STATUS_PANTINUM <- as.factor(ffp_train.df$STATUS_PANTINUM)
ffp_train.df$STATUS_GOLD <- as.factor(ffp_train.df$STATUS_GOLD)
ffp_train.df$STATUS_SILVER <- as.factor(ffp_train.df$STATUS_SILVER)
ffp_train.df$SERVICE_FLAG <- as.factor(ffp_train.df$SERVICE_FLAG)
ffp_train.df$CANCEL_FLAG <- as.factor(ffp_train.df$CANCEL_FLAG)
ffp_train.df$CREDIT_FLAG <- as.factor(ffp_train.df$CREDIT_FLAG)
ffp_train.df$BENEFIT_FLAG <- as.factor(ffp_train.df$BENEFIT_FLAG)
ffp_train.df$sentiment_Bad<- as.factor(ffp_train.df$sentiment_Bad)
ffp_train.df$sentiment_Not_Good<- as.factor(ffp_train.df$sentiment_Not_Good)
ffp_train.df$sentiment_Neutral<- as.factor(ffp_train.df$sentiment_Neutral)
ffp_train.df$sentiment_Good<- as.factor(ffp_train.df$sentiment_Good)
ffp_train.df$sentiment_Great<- as.factor(ffp_train.df$sentiment_Great)
ffp_train.df$BUYER_FLAG<- as.factor(ffp_train.df$BUYER_FLAG)


################################### LOGISTIC REGRESSION ####################

train_index <- createDataPartition(y=ffp_train.df$BUYER_FLAG, p=0.8, list=FALSE)
buyer_train <- ffp_train.df[train_index, ]
buyer_test <- ffp_train.df[-train_index, ]
buyer_glm0<- glm(BUYER_FLAG~., family=binomial, data=buyer_train)
summary(buyer_glm0)

hist(predict(buyer_glm0))

pred_resp <- predict(buyer_glm0,type="response")
hist(pred_resp)

table(buyer_train$BUYER_FLAG, (pred_resp > 0.5)*1, dnn=c("Truth","Predicted"))
table(buyer_train$BUYER_FLAG, (pred_resp > 0.2)*1, dnn=c("Truth","Predicted"))


pred_glm0_test<- predict(buyer_glm0, newdata = buyer_test, type="response")
pred <- prediction(pred_glm0_test, buyer_test$BUYER_FLAG)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
unlist(slot(performance(pred, "auc"), "y.values"))

########################## LOGISTIC REGRESSION WITH CROSS VALIDATION ##########

ffp_index <- createDataPartition(y=ffp_train.df$BUYER_FLAG, p=0.8, list=FALSE)
ffp_train_new <- ffp_train.df[ffp_index, ]
ffp_test_new <- ffp_train.df[-ffp_index, ]


folds <-createMultiFolds(y=ffp_train_new$BUYER_FLAG,k=4,times=20)
train <- ffp_train_new[folds[[1]],]
test <- ffp_train_new[-folds[[1]],]
model<-glm(BUYER_FLAG~., family = binomial(link=logit), data=train )
model_pre<-predict(model, type='response', newdata=test)

roc1<-roc((test$BUYER_FLAG),model_pre)
plot(roc1, 
     print.auc=T, 
     auc.polygon=T, 
     auc.polygon.col="skyblue",
     grid=c(0.1, 0.2),
     grid.col=c("green", "red"), 
     max.auc.polygon=T,
     print.thres=T)


auc_value<-as.numeric()
for(i in 1:80){
  train <- ffp_train_new [folds [[i]],]
  test <- ffp_train_new [- folds [[i]],]
  model <- glm(BUYER_FLAG~.,family=binomial(link=logit),data=train)
  model_pre<-predict(model,type='response', newdata=test)
  roc1<-roc((test$BUYER_FLAG),model_pre)
  auc_value <- append(auc_value, roc1$auc + 0)
}


summary(auc_value)
mean(auc_value) 

model_pre_test<-predict(model, type='response', newdata=ffp_test_new)
roc1<-roc((ffp_test_new$BUYER_FLAG),model_pre_test)
plot(roc1, 
     print.auc=T, 
     auc.polygon=T, 
     auc.polygon.col="skyblue",
     grid=c(0.1, 0.2),
     grid.col=c("green", "red"), 
     max.auc.polygon=T,
     print.thres=T)
roc1$auc
