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

reviews_rollout.df <- read.csv("reviews_rollout.csv")
ffp_rollout.df <- read.csv("ffp_rollout_X.csv")


dtm <- as.DocumentTermMatrix(reviews_rollout.df[,-(1)],weighting=weightTf)

sentiments <- analyzeSentiment(dtm)
sentiments <- as.data.frame(sentiments[,c("SentimentLM")]) ###BEST RESULT
colnames(sentiments) <- c("sentiment")

sentiments$ID <- reviews_rollout.df$ID

####if  sentiments results to categorical
sentiments$sentiment <-cut(sentiments$sentiment,breaks = 5, labels = c("Bad","Not_Good","Neutral","Good","Great"))
sentiments <- dummy_cols(sentiments,select_columns = 'sentiment')
sentiments$sentiment <-NULL

ffp_rollout.df <- merge(ffp_rollout.df,sentiments, all.x = TRUE)
ffp_rollout.df[is.na(ffp_rollout.df)] <- 0
ffp_rollout.df$ID <- NULL

write.csv(ffp_rollout.df,"ffp_rollout_with_sentiment.csv", row.names = FALSE)

