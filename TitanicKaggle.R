setwd("C:/Users/ugrayca/Documents/Pearson/Training/Kaggle/Titanic")


library(xgboost)
library(dplyr)
library(tidyr)
library(glmnet)
library(AppliedPredictiveModeling)
library(caret)
library(ipred)
library(e1071)
library(mice)
library(ggplot2)
library(klaR)

#Identify and combine train and test data sets for data cleaning & feature engineering#
test <- read.table("test.csv",sep= ",",header = T,stringsAsFactors = T)
test$source <- "Test"
test$Survived <- NA
train <- read.table("train.csv",sep= ",",header = T,stringsAsFactors = T)
train$source <- "Train"
full <- rbind (train, test)

str(train)

#count all NA's by columns
sapply(full, function(y) sum(length(which(is.na(y)))))
subset(full, is.na(Fare))
# Replace missing fare value with median fare for class/embarkment
full$Fare[1044] <- median(full[full$Pclass == '3' & full$Embarked == 'S', ]$Fare, na.rm = TRUE)

#Impute some 
# Set a random seed
set.seed(666)

# Perform mice imputation, excluding certain less-than-useful variables:
mice_mod <- mice(full[, !names(full) %in% c('PassengerId','Name','Ticket','Cabin','Family','Surname','Survived')], method='rf') 
mice_output <- complete(mice_mod)
full$Age <- mice_output$Age

#find NA's
train_missings <- full_transformed[rowSums(is.na(full_transformed)) > 0,]

full[c(62, 830), 'Embarked']
full[c(62, 830), 'Fare']
# Since their fare was $80 for 1st class, they most likely embarked from 'C'
full$Embarked[c(62, 830)] <- 'C'

str(full)

full_transformed <- full %>%
  mutate(Sex_male = ifelse(Sex == "male",1,0)) %>% #male 1 female 0
  mutate(Sex_female = ifelse(Sex == "female",1,0)) %>%
  mutate(Title = gsub('(.*, )|(\\..*)', '', Name)) %>%
  mutate(Title = case_when(Title == 'Dona'| Title =='Lady'| Title =='the Countess'|Title =='Capt'| Title =='Col'| Title =='Don'|
                             Title =='Dr'| Title =='Major'| Title =='Rev'| Title =='Sir'| Title =='Jonkheer' ~ "rare",
                           Title == "Mlle" ~ "Miss",
                           Title == "Ms" ~ "Miss",
                           Title == "Mne" ~ "Mrs",
                           Title == "Mr" ~ "Mr",
                           Title == "Mrs" ~ "Mrs",
                           Title == "Miss" ~ "Miss",
                           Title == "Master" ~ "Master")) %>%
  mutate(Embarked = case_when(Embarked == "C" ~ 1,
                              Embarked == "Q" ~ 2,
                              Embarked == "S" ~ 3)) %>% #cherbourg=1, Queenstown=2, Southampton =3, NA=NA
  mutate(Embarked_c = ifelse(Embarked == 1,1,0)) %>%
  mutate(Embarked_q = ifelse(Embarked == 2,1,0)) %>%
  mutate(Embarked_s = ifelse(Embarked == 3,1,0)) %>%
  mutate(Pclass_1 = ifelse(Pclass ==1,1,0)) %>%
  mutate(Pclass_2 = ifelse(Pclass ==2,1,0)) %>%
  mutate(Pclass_3 = ifelse(Pclass ==3,1,0)) %>%
  # mutate(Age = as.numeric(ifelse(is.na(Age),median(Age,na.rm=T),Age))) %>%
  # mutate(Fare = as.numeric(ifelse(Fare==0,median(Fare,na.rm=T),Fare))) %>%
  # mutate(Fare = scale(Fare)) %>% 
  # mutate(Age = scale(Age)) %>%
  mutate(Family_Size = SibSp+Parch+1) %>%
  mutate(FsizeD_sing = ifelse(Family_Size == 1,1,0)) %>%
  mutate(FsizeD_sm = ifelse((Family_Size < 5 & Family_Size >1),1,0)) %>%
  mutate(FsizeD_lrg = ifelse(Family_Size > 4,1,0)) %>%
  mutate(is_Alone = ifelse(Family_Size>1,0,1)) %>%
  mutate(Age_Fare = Fare*Age) %>%
  mutate(Child = ifelse(Age<18,1,0)) %>%
  mutate(Mother = ifelse((Sex == "female" & Age > 18 & Parch > 0 & Title != "Miss"),1,0)) %>%
  dplyr::select("Survived","Age","SibSp","Parch","Fare","source","Sex_male","Sex_female","Embarked_c","Embarked_q",
         "Embarked_s","Pclass_1","Pclass_2","Pclass_3","Family_Size","FsizeD_sing","FsizeD_sm","FsizeD_lrg","is_Alone",
         "Age_Fare","Child","Mother")


# Use ggplot2 to visualize the relationship between family size & survival
ggplot(full_transformed[1:891,], aes(x = Family_Size, fill = factor(Survived))) +
  geom_bar(stat='count', position='dodge') +
  scale_x_continuous(breaks=c(1:11)) +
  labs(x = 'Family Size')

#Converting the dependent variable back to categorical
full_transformed$Survived<-as.factor(full_transformed$Survived)
levels(full_transformed$Survived) <- c("Died", "Survived")



#Split Test and Training Data
train <- full_transformed[ which(full_transformed$source=='Train'),]
test <- full_transformed[ which(full_transformed$source=='Test'),]

#drop source variable#
train <- train[c(-6)]
test <- test[c(-6)]

str(train)
str(test)
#Spliting training set into two parts based on outcome: 75% and 25%
#Creates a cross validation set#
index <- createDataPartition(train$Survived, p=0.75, list=FALSE)
trainSet <- train[ index,]
cvSet <- train[-index,]

#Feature selection using rfe in caret
control <- rfeControl(functions = rfFuncs,
                      method = "repeatedcv",
                      repeats = 3,
                      verbose = FALSE)

outcomeName<-'Survived'
predictors<-names(trainSet)[!names(trainSet) %in% outcomeName]
Best_Pred_Profile <- rfe(trainSet[,predictors], trainSet[,outcomeName],
                         rfeControl = control)
# The top 5 variables (out of 8):
#   Sex_male, Sex_female, Pclass_3, Fare, Age

fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 3,
                           classProbs = TRUE,
                           # summaryFunction = twoClassSummary,
                           search = "random")


model_gbm<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm',trControl = fitControl,verbose=F)
model_rf<-train(trainSet[,predictors],trainSet[,outcomeName],method='rf',trControl = fitControl,verbose=F)
model_nnet<-train(trainSet[,predictors],trainSet[,outcomeName],method='nnet',trControl = fitControl,verbose=F)
model_glm<-train(trainSet[,predictors],trainSet[,outcomeName],method='glm',trControl = fitControl)
model_xgbTree<-train(trainSet[,predictors],trainSet[,outcomeName],method='xgbTree',trControl = fitControl,verbose=F)
model_rda<-train(trainSet[,predictors],trainSet[,outcomeName],method='rda',trControl = fitControl,verbose=F)
model_adabag<-train(trainSet[,predictors],trainSet[,outcomeName],method='AdaBoost.M1',trControl = fitControl,verbose=F)
model_treebag<-train(trainSet[,predictors],trainSet[,outcomeName],method='treebag',trControl = fitControl,verbose=F)
model_knn<-train(trainSet[,predictors],trainSet[,outcomeName],method='knn',trControl = fitControl,tuneLength=3)

summary(object=model_gbm)
varImp(object=model_rf)
varImp(object=model_nnet)
varImp(object=model_glm)
varImp(object=model_xgbTree)
varImp(object=model_rda)
varImp(object=model_adabag)
varImp(object=model_treebag)

plot(summary(object=model_gbm))
plot(varImp(object=model_rf))
plot(varImp(object=model_nnet))
plot(varImp(object=model_glm))
plot(varImp(object=model_xgbTree))

#Identify the best performing models with confusion matrices#
cvSet$pred_gbm<-predict.train(object=model_gbm,cvSet[,predictors],type="raw")
cm_gbm<-confusionMatrix(cvSet$Survived,cvSet$pred_gbm)

cvSet$pred_rf<-predict.train(object=model_rf,cvSet[,predictors],type="raw")
cm_rf<-confusionMatrix(cvSet$Survived,cvSet$pred_rf)

cvSet$pred_nnet<-predict.train(object=model_nnet,cvSet[,predictors],type="raw")
cm_nnet<-confusionMatrix(cvSet$Survived,cvSet$pred_nnet)

cvSet$pred_glm<-predict.train(object=model_glm,cvSet[,predictors],type="raw")
cm_glm<-confusionMatrix(cvSet$Survived,cvSet$pred_glm)

cvSet$pred_xgbtree<-predict.train(object=model_xgbTree,cvSet[,predictors],type="raw")
cm_xgbtree<-confusionMatrix(cvSet$Survived,cvSet$pred_xgbtree)

cvSet$pred_rda<-predict.train(object=model_rda,cvSet[,predictors],type="raw")
cm_rda<-confusionMatrix(cvSet$Survived,cvSet$pred_rda)

cvSet$pred_adabag<-predict.train(object=model_adabag,cvSet[,predictors],type="raw")
cm_adabag<-confusionMatrix(cvSet$Survived,cvSet$pred_adabag)

cvSet$pred_treebag<-predict.train(object=model_treebag,cvSet[,predictors],type="raw")
cm_treebag<-confusionMatrix(cvSet$Survived,cvSet$pred_treebag)

cvSet$pred_knn<-predict.train(object=model_knn,cvSet[,predictors],type="raw")
cm_knn<-confusionMatrix(cvSet$Survived,cvSet$pred_knn)

raw_overall<-data.frame(rbind(cm_gbm$overall,cm_rf$overall,cm_nnet$overall,cm_glm$overall,
                              cm_xgbtree$overall,cm_rda$overall,cm_adabag$overall,cm_treebag$overall,cm_knn$overall))
raw_byclass<-data.frame(rbind(cm_gbm$byClass,cm_rf$byClass,cm_nnet$byClass,cm_glm$byClass,
                              cm_xgbtree$byClass,cm_rda$byClass,cm_adabag$byClass,cm_treebag$byClass,cm_knn$byClass))
names<-matrix(c("gbm",'rf','nnet',"glm",'xgbtree','rda','adabag','treebag','knn'),nrow=9,byrow=T)
raw_table<-cbind(names,raw_overall,raw_byclass)
raw_table<-raw_table[order(-raw_table$Accuracy),]


#Probabilities#
cvSet$prob_gbm<-predict.train(object=model_gbm,cvSet[,predictors],type="prob")
cvSet$prob_rf<-predict.train(object=model_rf,cvSet[,predictors],type="prob")
cvSet$prob_nnet<-predict.train(object=model_nnet,cvSet[,predictors],type="prob")
cvSet$prob_glm<-predict.train(object=model_glm,cvSet[,predictors],type="prob")
cvSet$prob_xgbtree<-predict.train(object=model_xgbTree,cvSet[,predictors],type="prob")
cvSet$prob_rda<-predict.train(object=model_rda,cvSet[,predictors],type="prob")
cvSet$prob_adabag<-predict.train(object=model_adabag,cvSet[,predictors],type="prob")
cvSet$prob_treebag<-predict.train(object=model_treebag,cvSet[,predictors],type="prob")
cvSet$prob_knn<-predict.train(object=model_knn,cvSet[,predictors],type="prob")

#Observe Correlations of the survive predictions#
names(cvSet)
survived_probs <- as.data.frame(c(cvSet$prob_gbm[2],cvSet$prob_rf[2],cvSet$prob_nnet[2],cvSet$prob_glm[2],cvSet$prob_xgbtree[2],cvSet$prob_rda[2],cvSet$prob_adabag[2],
    cvSet$prob_treebag[2],cvSet$prob_knn[2]))
names(survived_probs) <- c('prob_gbm','prob_rf','prob_nnet',"prob_glm",'prob_xgbtree','prob_rda','prob_adabag','prob_treebag','prob_knn')
cor(survived_probs)

cvSet$pred_avg<-(cvSet$prob_gbm$Survived+cvSet$prob_rf$Survived+cvSet$prob_nnet$Survived+cvSet$prob_glm$Survived+cvSet$prob_xgbtree$Survived+
                   cvSet$prob_rda$Survived+cvSet$prob_adabag$Survived+cvSet$prob_treebag$Survived+cvSet$prob_knn$Survived)/9
#Splitting into binary classes at 0.5
cvSet$pred_avg<-as.factor(ifelse(cvSet$pred_avg>0.5,'Survived','Died'))


#The majority vote
cvSet$pred_majority<-as.factor(ifelse(as.numeric((cvSet$pred_gbm=="Survived") + (cvSet$pred_rf=="Survived") +
                                                   (cvSet$pred_nnet=="Survived") + (cvSet$pred_glm=="Survived") + (cvSet$pred_xgbtree=="Survived") +
                                                   (cvSet$pred_rda=="Survived") + (cvSet$pred_adabag=="Survived") + (cvSet$pred_treebag=="Survived") + 
                                                   (cvSet$pred_knn=="Survived"))>=5,"Survived","Died"))

prob_overall<-data.frame(rbind(cm_gbm$overall,cm_rf$overall,cm_nnet$overall,cm_glm$overall,
                              cm_xgbtree$overall,cm_rda$overall,cm_adabag$overall,cm_treebag$overall,cm_knn$overall))
prob_byclass<-data.frame(rbind(cm_gbm$byClass,cm_rf$byClass,cm_nnet$byClass,cm_glm$byClass,
                              cm_xgbtree$byClass,cm_rda$byClass,cm_adabag$byClass,cm_treebag$byClass,cm_knn$byClass))
names<-matrix(c("gbm",'rf','nnet',"glm",'xgbtree','rda','adabag','treebag','knn'),nrow=9,byrow=T)
prob_table<-cbind(names,prob_overall,prob_byclass)
prob_table<-prob_table[order(-prob_table$Accuracy),]

#Majority vs Average Prediction with all models#
confusionMatrix(cvSet$Survived,cvSet$pred_majority)
confusionMatrix(cvSet$Survived,cvSet$pred_avg)

##Reduce to the top three models, xgbtree,nnet,rf##
cvSet <- cvSet[1:21]

cvSet$pred_rf<-predict.train(object=model_rf,cvSet[,predictors],type="raw")
cvSet$pred_nnet<-predict.train(object=model_nnet,cvSet[,predictors],type="raw")
cvSet$pred_xgbtree<-predict.train(object=model_xgbTree,cvSet[,predictors],type="raw")
cvSet$pred_majority<-as.factor(ifelse(as.numeric((cvSet$pred_rf=="Survived") + (cvSet$pred_nnet=="Survived") +
                                                   (cvSet$pred_nnet=="pred_xgbtree"))>=2,"Survived","Died"))

cvSet$prob_rf<-predict.train(object=model_rf,cvSet[,predictors],type="prob")
cvSet$prob_nnet<-predict.train(object=model_nnet,cvSet[,predictors],type="prob")
cvSet$prob_xgbtree<-predict.train(object=model_xgbTree,cvSet[,predictors],type="prob")
cvSet$pred_avg<-(cvSet$prob_rf$Survived+cvSet$prob_nnet$Survived+cvSet$prob_xgbtree$Survived)/3
#Splitting into binary classes at 0.5
cvSet$pred_avg<-as.factor(ifelse(cvSet$pred_avg>0.5,'Survived','Died'))

#revisit confusion matrix using 3 models
confusionMatrix(cvSet$Survived,cvSet$pred_majority)
confusionMatrix(cvSet$Survived,cvSet$pred_avg)


##Apply to Test Data##
test$prob_rf<-predict.train(object=model_rf,test[,predictors],type="prob")
test$prob_nnet<-predict.train(object=model_nnet,test[,predictors],type="prob")
test$prob_xgbtree<-predict.train(object=model_xgbTree,test[,predictors],type="prob")
test$pred_avg<-(test$prob_rf$Survived+test$prob_nnet$Survived+test$prob_xgbtree$Survived)/3
test$pred_avg<-as.factor(ifelse(test$pred_avg>0.5,'1','0'))

test <- test[c()]
test$PassengerID <- rownames(test)
test_submission <- test[c("PassengerID","pred_avg")]
#81% Accuracy on Kaggle
write.table(test_submission,"titantic_submission_7.2.18.csv",sep=",", row.names = F)







##Stack Algorthims##
#The three best were xgbtree, nnet, rf#
#This doesn't really add a whole lot to the model, so just commented out#
# fitControl <- trainControl(
#   method = "cv",
#   number = 10,
#   savePredictions = 'final', # To save out of fold predictions for best parameter combinantions
#   classProbs = T # To save the class probabilities of the out of fold predictions
# )
# 
# model_rf<-train(trainSet[,predictors],trainSet[,outcomeName],method='rf',trControl = fitControl,verbose=F,tuneLength=3)
# model_nnet<-train(trainSet[,predictors],trainSet[,outcomeName],method='nnet',trControl = fitControl,verbose=F,tuneLength=3)
# model_xgbTree<-train(trainSet[,predictors],trainSet[,outcomeName],method='xgbTree',trControl = fitControl,verbose=F,tuneLength=3)
# 
# #Predicting the out of fold prediction probabilities for training data
# trainSet$OOF_pred_rf<-model_rf$pred$Survived[order(model_rf$pred$rowIndex)]
# trainSet$OOF_pred_nnet<-model_nnet$pred$Survived[order(model_nnet$pred$rowIndex)]
# trainSet$OOF_pred_xgbtree<-model_xgbTree$pred$Survived[order(model_xgbTree$pred$rowIndex)]
# 
# #Predicting probabilities for the test data
# cvSet_stack <- cvSet[1:21]
# cvSet_stack$OOF_pred_rf<-predict(model_rf,cvSet[predictors],type='prob')$Survived
# cvSet_stack$OOF_pred_nnet<-predict(model_nnet,cvSet[predictors],type='prob')$Survived
# cvSet_stack$OOF_pred_xgbtree<-predict(model_xgbTree,cvSet[predictors],type='prob')$Survived
# 
# #Predictors for top layer models 
# predictors_top<-c('OOF_pred_rf','OOF_pred_nnet','OOF_pred_xgbtree') 
# 
# #xgbtree as top layer model 
# model_xgbTree_top<-train(trainSet[,predictors_top],trainSet[,outcomeName],method='xgbTree',trControl=fitControl,tuneLength=3)
# 
# #predict using xgbtree top layer model
# cvSet_stack$xgbTree_stacked<-predict(model_xgbTree_top,cvSet_stack[,predictors_top])
# 
# cm_glm<-confusionMatrix(cvSet_stack$Survived,cvSet_stack$xgbTree_stacked)