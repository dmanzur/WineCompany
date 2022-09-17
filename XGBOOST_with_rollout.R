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
ffp_train.df$BUYER_FLAG<- as.numeric(as.character(ffp_train.df$BUYER_FLAG))

########################## XGBOOST CROSS VALIDATION ##########
ffp_index <- createDataPartition(y=ffp_train.df$BUYER_FLAG, p=0.8, list=FALSE)
ffp_train_new <- ffp_train.df[ffp_index, ]
ffp_validation <- ffp_train.df[-ffp_index, ]

param <- list(objective="binary:logistic")

folds <-createMultiFolds(y=ffp_train_new$BUYER_FLAG,k=4,times=20)
auc_value<-as.numeric()
for(i in 1:80){
  train <- ffp_train.df [folds [[i]],]
  test <- ffp_train.df [- folds [[i]],]
  X_train = data.matrix(train[,-22])                 
  y_train = train[,22]  
  X_test = data.matrix(test[,-22])                
  y_test = test[,22] 
  xgboost_train = xgb.DMatrix(data=X_train, label=y_train)
  xgboost_test = xgb.DMatrix(data=X_test, label=y_test)
  model <- xgboost(data = xgboost_train, max.depth=10, , nrounds=50, objective = "binary:logistic", verbose = 0)
  pred_test <- predict(model, xgboost_test, type="response")
  
  roc1<-roc((test$BUYER_FLAG),pred_test)
  auc_value <- append(auc_value, roc1$auc + 0)
}

summary(auc_value)
mean(auc_value) 


X_valid = data.matrix(ffp_validation[,-22])                
y_valid = ffp_validation[,22] 
xgboost_valid = xgb.DMatrix(data=X_valid, label=y_valid)
model_pre_valid<-predict(model, type='response', newdata=xgboost_valid)

roc1<-roc((ffp_validation$BUYER_FLAG),model_pre_valid)
plot(roc1, 
     print.auc=T, 
     auc.polygon=T, 
     auc.polygon.col="skyblue",
     grid=c(0.1, 0.2),
     grid.col=c("green", "red"), 
     max.auc.polygon=T,
     print.thres=T)
roc1$auc
ffp_validation$probs <- model_pre_valid



x<-as.numeric()
y<-as.numeric()
z<-as.null()
sums<-as.numeric()

loops <- seq(0, 1, by=0.01)

for (i in loops){
  ffp_validation[[paste0("probs",i)]] <- ifelse(ffp_validation$probs>i,ifelse(ffp_validation$BUYER_FLAG==1,54.7,-11.15),0)
  sums <-append(sums, sum(ffp_validation[[paste0("probs",i)]]))
  x<-append(x,i)
  y<-append(y,sum(ffp_validation[[paste0("probs",i)]]))
  z<-append(z,as.String(i))
}

df <- data.frame(x,y,z)

plot(df$x, df$y)








#################### RUNNING ON ROLLOUT #################

ffp_rollout.df <- read.csv("ffp_rollout_with_sentiment.csv")
ffp_rollout.df$STATUS_PANTINUM <- as.factor(ffp_rollout.df$STATUS_PANTINUM)
ffp_rollout.df$STATUS_GOLD <- as.factor(ffp_rollout.df$STATUS_GOLD)
ffp_rollout.df$STATUS_SILVER <- as.factor(ffp_rollout.df$STATUS_SILVER)
ffp_rollout.df$SERVICE_FLAG <- as.factor(ffp_rollout.df$SERVICE_FLAG)
ffp_rollout.df$CANCEL_FLAG <- as.factor(ffp_rollout.df$CANCEL_FLAG)
ffp_rollout.df$CREDIT_FLAG <- as.factor(ffp_rollout.df$CREDIT_FLAG)
ffp_rollout.df$BENEFIT_FLAG <- as.factor(ffp_rollout.df$BENEFIT_FLAG)
ffp_rollout.df$sentiment_Bad<- as.factor(ffp_rollout.df$sentiment_Bad)
ffp_rollout.df$sentiment_Not_Good<- as.factor(ffp_rollout.df$sentiment_Not_Good)
ffp_rollout.df$sentiment_Neutral<- as.factor(ffp_rollout.df$sentiment_Neutral)
ffp_rollout.df$sentiment_Good<- as.factor(ffp_rollout.df$sentiment_Good)
ffp_rollout.df$sentiment_Great<- as.factor(ffp_rollout.df$sentiment_Great)

X_rollout = data.matrix(ffp_rollout.df[,-22])                
y_rollout = ffp_rollout.df[,22] 
xgboost_test = xgb.DMatrix(data=X_rollout, label=y_rollout)
model_rollout_pre<-predict(model, type='response', newdata=xgboost_test)
summary(model_rollout_pre)

ffp_rollout.df$probs <- model_rollout_pre
ffp_rollout.df$BUYER_FLAG<-ifelse(ffp_rollout.df$probs>0.16,1,0)
ffp_rollout.df$BUYER_FLAG

ID<-as.numeric()
for(i in 50001:70000){
  ID<-append(ID,i)
}
BUYER_FLAG <- ffp_rollout.df$BUYER_FLAG
df <- data.frame(ID,BUYER_FLAG)
write.csv(df,"recommendations.csv", row.names = FALSE)

