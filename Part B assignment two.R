library(dplyr)
library(neuralnet)
library(caTools)
library(caret)



setwd("F:\\uOttawa\\Fundamental\\Assignment\\Assignment Two")

diabetes <- read.delim("diabetes.csv", header = TRUE, sep= ",")

#Converting all the column to numeric column and replacing the question mark sign by NA

diabetes$BMI          <-as.numeric(diabetes$BMI)
diabetes$Glucose      <-as.numeric(diabetes$Glucose)
diabetes$SkinThickness<-as.numeric(diabetes$SkinThickness)
diabetes$BloodPressure<-as.numeric(diabetes$BloodPressure)
diabetes$Insulin      <-as.numeric(diabetes$Insulin)

BMI_null <- sum(is.na(diabetes$BMI))
Glucose_null <- sum(is.na(diabetes$Glucose))
SkinThickness_null <- sum(is.na(diabetes$SkinThickness))
BloodPressure_null <- sum(is.na(diabetes$BloodPressure))
Insulin_null <- sum(is.na(diabetes$Insulin))




#chossing the beat central measure of tendency for BMI
hist(diabetes$BMI)
d <- density(diabetes$BMI,na.rm=T)
plot(d, main = "Distribution of BMI")
polygon(d, col="red", border="blue")

BMI_mean <- mean(diabetes$BMI,na.rm=T)
BMI_median <- median(diabetes$BMI,na.rm=T)


#chossing the beat central measure of tendency for Glucose
hist(diabetes$Glucose)
d2 <- density(diabetes$Glucose,na.rm=T)
plot(d2, main = "Distribution of Glucose")
polygon(d2, col="red", border="blue")

Glucose_mean <- mean(diabetes$Glucose,na.rm=T)
Glucose_median <- median(diabetes$Glucose,na.rm=T)

#chossing the beat central measure of tendency for BloodPressure
hist(diabetes$BloodPressure)
d3 <- density(diabetes$BloodPressure,na.rm=T)
plot(d3, main = "Distribution of BloodPressure")
polygon(d3, col="red", border="blue")

BloodPressure_mean <- mean(diabetes$BloodPressure,na.rm=T)
BloodPressure_median <- median(diabetes$BloodPressure,na.rm=T)

#chossing the beat central measure of tendency for Insulin
hist(diabetes$Insulin)
d4 <- density(diabetes$Insulin,na.rm=T)
plot(d4, main = "Distribution of Insulin")
polygon(d4, col="red", border="blue")

Insulin_mean <- mean(diabetes$Insulin,na.rm=T)
Insulin_median <- median(diabetes$Insulin,na.rm=T)

#chossing the beat central measure of tendency for SkinThickness
hist(diabetes$SkinThickness)
d5 <- density(diabetes$SkinThickness,na.rm=T)
plot(d5, main = "Distribution of SkinThickness")
polygon(d5, col="red", border="blue")

SkinThickness_mean <- mean(diabetes$SkinThickness,na.rm=T)
SkinThickness_median <- median(diabetes$SkinThickness,na.rm=T)



# Calculating the mean Insulin for each class
Class0 <- diabetes[(diabetes$Outcome ==0),]
Class1 <- diabetes[(diabetes$Outcome ==1),]

mean_insulin_class0 <- mean(Class0$Insulin, na.rm=T)
median_insulin_class0 <- median(Class0$Insulin, na.rm=T)

mean_insulin_class1 <- mean(Class1$Insulin, na.rm=T)
median_insulin_class1 <- median(Class1$Insulin, na.rm=T)

mean_SkinThickness_class0 <- mean(Class0$SkinThickness, na.rm=T)
median_SkinThickness_class0 <- median(Class0$SkinThickness, na.rm=T)

mean_SkinThickness_class1 <- mean(Class1$SkinThickness, na.rm=T)
median_SkinThickness_class1 <- median(Class1$SkinThickness, na.rm=T)

# updating the na values for BMI, Glucose, and Bloodpressure columns with means calculated 
diabetes$BMI[is.na(diabetes$BMI)] = BMI_mean
diabetes$Glucose[is.na(diabetes$Glucose) ]=Glucose_mean
diabetes$BloodPressure[is.na(diabetes$BloodPressure)]= BloodPressure_mean
# Updating the Na values of insulin for with the median each class

diabetes$Insulin[is.na(diabetes$Insulin) & diabetes$Outcome==0]<- median_insulin_class0

diabetes$Insulin[is.na(diabetes$Insulin) & diabetes$Outcome==1]<- median_insulin_class1

# Updating the Na values of SkinThickness for with the median each class
diabetes$SkinThickness[is.na(diabetes$SkinThickness) & diabetes$Outcome==0]<- median_SkinThickness_class0

diabetes$SkinThickness[is.na(diabetes$SkinThickness) & diabetes$Outcome==1]<- median_SkinThickness_class1



#standardization of all of the columns' values using zscore
diabetes[c(1: 8)] <- scale(diabetes[c(1: 8)])





# splitting the dataset into training and testing data
set.seed(123)
sample_data = sample.split(diabetes, SplitRatio = 0.7)
train_data <- subset(diabetes, sample_data == TRUE)
test_data <- subset(diabetes, sample_data == FALSE)


#Training Neural network model with only 2 hidden nodes 

set.seed(555)
NN <- neuralnet(Outcome~ Pregnancies+Glucose+BloodPressure+SkinThickness+Insulin+BMI+DiabetesPedigreeFunction
                +Age,train_data, hidden=2, linear.output = T )

plot(NN)

NN$result.matrix




predict_testNN = neuralnet::compute(NN, test_data[,c(1:8)])
predict_testNN <- data.frame(predict_testNN)
results <- data.frame(actual=test_data$Outcome, Prediction=predict_testNN$net.result)


results
roundedresults<-results%>% mutate(Prediction = ifelse(Prediction < 0.5, 0, 1))


roundedresults$actual <- as.factor(roundedresults$actual)

roundedresults$Prediction <- as.factor(roundedresults$Prediction)
confusionMatrix(roundedresults$actual,roundedresults$Prediction)

saveRDS(NN, "model.rds")
#=====================================================================
#trying to change the learning rate 
set.seed(555)
NN <- neuralnet(Outcome~ Pregnancies+Glucose+BloodPressure+SkinThickness+Insulin+BMI+DiabetesPedigreeFunction
                +Age,train_data, hidden=2, linear.output = T ,learningrate=.001,stepmax=10e10)

plot(NN)

NN$result.matrix




predict_testNN = neuralnet::compute(NN, test_data[,c(1:8)])
predict_testNN <- data.frame(predict_testNN)
results <- data.frame(actual=test_data$Outcome, Prediction=predict_testNN$net.result)


results
roundedresults<-results%>% mutate(Prediction = ifelse(Prediction < 0.5, 0, 1))


roundedresults$actual <- as.factor(roundedresults$actual)

roundedresults$Prediction <- as.factor(roundedresults$Prediction)
confusionMatrix(roundedresults$actual,roundedresults$Prediction)
#=========================================================================================
#changing the number of the nodes per the two hidden nodes
set.seed(555)
NN <- neuralnet(Outcome~ Pregnancies+Glucose+BloodPressure+SkinThickness+Insulin+BMI+DiabetesPedigreeFunction
                +Age,train_data, hidden=c(5,5), linear.output = T ,learningrate=.1,stepmax=10e10)

plot(NN)

NN$result.matrix




predict_testNN = neuralnet::compute(NN, test_data[,c(1:8)])
predict_testNN <- data.frame(predict_testNN)
results <- data.frame(actual=test_data$Outcome, Prediction=predict_testNN$net.result)


results
roundedresults<-results%>% mutate(Prediction = ifelse(Prediction < 0.5, 0, 1))


roundedresults$actual <- as.factor(roundedresults$actual)

roundedresults$Prediction <- as.factor(roundedresults$Prediction)
confusionMatrix(roundedresults$actual,roundedresults$Prediction)
 #========================================================================================
#changing the activation function
set.seed(555)
NN <- neuralnet(Outcome~ Pregnancies+Glucose+BloodPressure+SkinThickness+Insulin+BMI+DiabetesPedigreeFunction
                +Age,train_data, hidden=2, linear.output = T ,learningrate=.1,stepmax=10e10, act.fct = "logistic")

plot(NN)

NN$result.matrix



predict_testNN = neuralnet::compute(NN, test_data[,c(1:8)])
predict_testNN <- data.frame(predict_testNN)
results <- data.frame(actual=test_data$Outcome, Prediction=predict_testNN$net.result)


results
roundedresults<-results%>% mutate(Prediction = ifelse(Prediction < 0.5, 0, 1))


roundedresults$actual <- as.factor(roundedresults$actual)

roundedresults$Prediction <- as.factor(roundedresults$Prediction)
confusionMatrix(roundedresults$actual,roundedresults$Prediction)

#
#==============================================================================================
#changing the activation function
set.seed(555)
NN <- neuralnet(Outcome~ Pregnancies+Glucose+BloodPressure+SkinThickness+Insulin+BMI+DiabetesPedigreeFunction
                +Age,train_data, hidden=2, linear.output = T ,learningrate=.1,stepmax=10e10, act.fct = "tanh")

plot(NN)

NN$result.matrix



predict_testNN = neuralnet::compute(NN, test_data[,c(1:8)])
predict_testNN <- data.frame(predict_testNN)
results <- data.frame(actual=test_data$Outcome, Prediction=predict_testNN$net.result)


results
roundedresults<-results%>% mutate(Prediction = ifelse(Prediction < 0.5, 0, 1))


roundedresults$actual <- as.factor(roundedresults$actual)

roundedresults$Prediction <- as.factor(roundedresults$Prediction)
confusionMatrix(roundedresults$actual,roundedresults$Prediction)
#=====================================================================================================
#Calculating the accuracy on the training set




set.seed(555)
NN <- neuralnet(Outcome~ Pregnancies+Glucose+BloodPressure+SkinThickness+Insulin+BMI+DiabetesPedigreeFunction
                +Age,train_data, hidden=2, linear.output = T)

plot(NN)

NN$result.matrix




predict_trainNN = neuralnet::compute(NN, train_data[,c(1:8)])
predict_trainNN <- data.frame(predict_trainNN)
results <- data.frame(actual=train_data$Outcome, Prediction=predict_trainNN$net.result)


results
roundedresults<-results%>% mutate(Prediction = ifelse(Prediction < 0.5, 0, 1))


roundedresults$actual <- as.factor(roundedresults$actual)

roundedresults$Prediction <- as.factor(roundedresults$Prediction)
confusionMatrix(roundedresults$actual,roundedresults$Prediction)


#calculating the accuracy on the training dataset using 2 hidden layers and 5 nodes

set.seed(555)
NN <- neuralnet(Outcome~ Pregnancies+Glucose+BloodPressure+SkinThickness+Insulin+BMI+DiabetesPedigreeFunction
                +Age,train_data, hidden=c(5,5), linear.output = T)

plot(NN)

NN$result.matrix




predict_trainNN = neuralnet::compute(NN, train_data[,c(1:8)])
predict_trainNN <- data.frame(predict_trainNN)
results <- data.frame(actual=train_data$Outcome, Prediction=predict_trainNN$net.result)


results
roundedresults<-results%>% mutate(Prediction = ifelse(Prediction < 0.5, 0, 1))


roundedresults$actual <- as.factor(roundedresults$actual)

roundedresults$Prediction <- as.factor(roundedresults$Prediction)
confusionMatrix(roundedresults$actual,roundedresults$Prediction)





