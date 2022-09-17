rm(list = ls())
gc()

library(magrittr) 
library(tm)
library(SentimentAnalysis)
library(tibble)
library(party)
library(caret)
library(e1071) 
options(stringsAsFactors = F) 

setwd("C:\\R\\nmproject")

reviews_training.df <- read.csv("reviews_training.csv")
ffp_train.df <- read.csv("ffp_train.csv")

dtm <- as.DocumentTermMatrix(reviews_training.df[,-(1)],weighting=weightTfIdf)

sentiments <- analyzeSentiment(dtm)

sentiments <- as.data.frame(sentiments[,c("SentimentLM")]) ###BEST RESULT

colnames(sentiments) <- c("sentiment")
sentiments$ID <- reviews_training.df$ID
ffp_train.df <- merge(ffp_train.df,sentiments, all.x = TRUE)
ffp_train.df$sentiment[is.na(ffp_train.df$sentiment)] <- 0
ffp_train.df$ID <- NULL