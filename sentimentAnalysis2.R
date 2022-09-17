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
options(stringsAsFactors = F) 

setwd("C:\\R\\nmproject")

reviews_training.df <- read.csv("reviews_training.csv")
ffp_train.df <- read.csv("ffp_train.csv")


dtm <- as.DocumentTermMatrix(reviews_training.df[,-(1)],weighting=weightTf)

sentiments <- analyzeSentiment(dtm)
sentiments <- as.data.frame(sentiments[,c("SentimentLM")]) ###BEST RESULT
colnames(sentiments) <- c("sentiment")

sentiments$ID <- reviews_training.df$ID

####if  sentiments results to categorical
sentiments$sentiment <-cut(sentiments$sentiment,breaks = 5, labels = c("Bad","Not_Good","Neutral","Good","Great"))
sentiments <- dummy_cols(sentiments,select_columns = 'sentiment')
sentiments$sentiment <-NULL

ffp_train.df <- merge(ffp_train.df,sentiments, all.x = TRUE)
ffp_train.df[is.na(ffp_train.df)] <- 0
ffp_train.df$ID <- NULL

write.csv(ffp_train.df,"ffp_train_with_sentiment.csv", row.names = FALSE)

