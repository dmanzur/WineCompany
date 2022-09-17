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
library(ROCR)
library(Metrics)
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


################################### NAIVE BAYAS WITHOUT CROSS VALIDATION ####################

set.seed(100)

spl = sample.split(ffp_train.df, SplitRatio = 0.8)
train = subset(ffp_train.df, spl==TRUE)
test = subset(ffp_train.df, spl==FALSE)

print(dim(train)); print(dim(test))

classifier_cl <- naiveBayes(BUYER_FLAG ~ ., data = train)
classifier_cl

y_pred <- predict(classifier_cl, newdata = test)

cm <- table(test$BUYER_FLAG, y_pred)
cm
roc_ = roc(y,predict(mnb, newdata = M, type ="prob")[,2])





