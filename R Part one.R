install.packages("plyr")
install.packages("corrplot")
install.packages('rpart')
install.packages('varlmp')
install.packages("caTools")
install.packages("magrittr") 
install.packages("dplyr") 
install.packages("corrplot")

library("tidyverse")
library(plyr)
library(caret)
library(caTools)
library(corrplot)
library(rpart)
library(rpart.plot)
library(caret)
library(magrittr) 
library(dplyr) 

source("http://www.sthda.com/upload/rquery_cormat.r")
#setting the working directory to the project's path
setwd("F:\\uOttawa\\Fundamental\\Assignment\\Assignment Two")

#reading the data from the csv
Hypothyroid <-read.delim(file="hypothyroid.csv", header =TRUE, sep=",")


#rempving all values that has unkown value
Hypothyroid<-Hypothyroid[!(Hypothyroid$sex=="?"|Hypothyroid$TSH=="?"|Hypothyroid$T3=="?"|
                             Hypothyroid$TT4=="?"|Hypothyroid$T4U=="?"|
                             Hypothyroid$FTI=="?"|Hypothyroid$age=="?"),]


#Removing the class secondary_hypothyroid from the dataset
Hypothyroid <- Hypothyroid[!(Hypothyroid$Class=="secondary_hypothyroid"),]

#Dropping un important columns from the dataset

Hypothyroid <- subset(Hypothyroid, select=-c(referral_source,
                                             TBG,
                                             TBG_measured,
                                             FTI_measured,
                                             T4U_measured,
                                             TT4_measured,
                                             T3_measured,
                                             TSH_measured))


#changing the Categorical values into numerical values


Hypothyroid$Class<- factor(Hypothyroid$Class, levels = c("negative","compensated_hypothyroid",    
                                  "primary_hypothyroid"),
                                               labels = c(1, 2, 3))



unique(Hypothyroid$Class)


# chacking the type of each columns
str(Hypothyroid)
 

#converting columns type to numeric

Hypothyroid$TSH <- as.numeric(Hypothyroid$TSH)
Hypothyroid$T3 <- as.numeric(Hypothyroid$T3)
Hypothyroid$TT4 <- as.numeric(Hypothyroid$TT4)
Hypothyroid$T4U <- as.numeric(Hypothyroid$T4U)
Hypothyroid$FTI <- as.numeric(Hypothyroid$FTI)
Hypothyroid$age <- as.integer(Hypothyroid$age)



str(Hypothyroid)



#Removing columns with null values
Hypothyroid_cleaned<-Hypothyroid %>% drop_na()


str(Hypothyroid_cleaned)

# Change categorical values to numerical values

Hypothyroid_final<- data.matrix(data.frame(unclass(Hypothyroid_cleaned)))
Hypothyroid_final = data.frame(Hypothyroid_final)

#Performing attribute selection
correlationMatrix <- cor(Hypothyroid_final)
print(correlationMatrix)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=.75)
print(highlyCorrelated)
hc = sort(highlyCorrelated)
hc
reduced_Data = Hypothyroid_final[ ,-c(hc)]
print (reduced_Data)


rquery.cormat(Hypothyroid_final)
cormat<-rquery.cormat(Hypothyroid_final, graphType="heatmap")
correlationMatrix[19]
#===================================================================
Hypothyroid_final$Class<-factor(Hypothyroid_final$Class, levels = c(1, 2, 3), 
                                labels = c("negative","compensated_hypothyroid",    
                                           "primary_hypothyroid"))
set.seed(123)
sample_data = sample.split(Hypothyroid_final, SplitRatio = 0.7)
train_data <- subset(Hypothyroid_final, sample_data == TRUE)
test_data <- subset(Hypothyroid_final, sample_data == FALSE)
#===================================================================



set.seed(123) #generates a reproducible random sampling

#specify the cross-validation method
ctrl <- trainControl(method = "cv", number = 10)

#fit a decision tree model and use k-fold CV to evaluate performance
dtree_fit_gini <- train(Class~., data = Hypothyroid_final,
                        method = "rpart", parms = list(split = "gini")
                        , trControl = ctrl, tuneLength = 10)

#Step 5: Evaluate - view summary of k-fold CV               
print(dtree_fit_gini) #metrics give us an idea of how well the model performed on previously unseen data

#view final model
dtree_fit_gini$finalModel
prp(dtree_fit_gini$finalModel, box.palette = "Greens", tweak = 1.3) #view the tree using prop() function

#view predictions for each fold
dtree_fit_gini$resample

#Check accuracy
test_pred_gini <- predict(dtree_fit_gini, newdata = test_data)
confusionMatrix(test_pred_gini, test_data$Class )  #check accuracy

#==============================================================
#Training decision tree using Reduction in Variance strategy

DT_newmodel1 <- rpart(Class ~ ., data = train_data, method = "class", parms=list(split = "reduction"),
                     control = rpart.control(cp = 0,maxdepth = 5, minsplit =20))
                       
summary(DT_newmodel1)
#Plot Decision Tree
prp(DT_newmodel1, box.palette = "Greens", tweak = 1.3)
# Examine the complexity plot
printcp(DT_newmodel1)
plotcp(DT_newmodel1)
pred1 <-(predict(DT_newmodel1, newdata = test_data,type="class"))
confusionMatrix(pred1, test_data$Class)  #check accuracy
mean(pred1==test_data$Class)
opt <- which.min(DT_newmodel1$cptable[,"xerror"])
cp <- DT_newmodel1$cptable[opt, "CP"]
pruned_model1 <- prune(DT_newmodel1,cp)
prp(pruned_model1, box.palette = "Greens", tweak = 1.3)
pred1_pruned <-(predict(pruned_model1, newdata = test_data,type="class"))
confusionMatrix(pred1_pruned, test_data$Class)  #check accuracy
mean(pred1_pruned==test_data$Class)
#==============================================================
#Training a DT using Chi-Square splitting strategy

DT_newmodel2 <- rpart(Class ~ ., data = train_data, method = "class",
                      parms=list(split = "chi"),
                     control = rpart.control(cp = 0,maxdepth = 5, minsplit =20))

summary(DT_newmodel2)
#Plot Decision Tree
prp(DT_newmodel2, box.palette = "Greens", tweak = 1.3)
# Examine the complexity plot
printcp(DT_newmodel2)
plotcp(DT_newmodel2)
pred2 <-(predict(DT_newmodel2, newdata = test_data,type="class"))
confusionMatrix(pred2, test_data$Class)  #check accuracy on the training set
mean(pred2==test_data$Class)
opt <- which.min(DT_newmodel2$cptable[,"xerror"]) #Caculating the CP
cp <- DT_newmodel2$cptable[opt, "CP"]
pruned_model2 <- prune(DT_newmodel2,cp) # pruning the tree
prp(pruned_model2, box.palette = "Greens", tweak = 1.3)
pred2_pruned <-(predict(pruned_model2, newdata = test_data,type="class"))
confusionMatrix(pred2_pruned, test_data$Class)  #check accuracy on the tesing set
mean(pred2_pruned==test_data$Class)
#================================================================
#Training a DT using entropy splitting strategy

DT_newmodel3 <- rpart(Class ~ ., data = train_data, method = "class", parms=list(split = "entropy"),
                      control = rpart.control(cp = 0,maxdepth = 5, minsplit =20))

summary(DT_newmodel3)
#Plot Decision Tree
prp(DT_newmodel3, box.palette = "Greens", tweak = 1.3)
# Examine the complexity plot
printcp(DT_newmodel3)
plotcp(DT_newmodel3)
pred3 <-(predict(DT_newmodel3, newdata = test_data,type="class"))
confusionMatrix(pred3, test_data$Class)  #check accuracy
mean(pred3==test_data$Class)
opt <- which.min(DT_newmodel3$cptable[,"xerror"])
cp <- DT_newmodel3$cptable[opt, "CP"]
pruned_model3 <- prune(DT_newmodel3,cp)
prp(pruned_model3, box.palette = "Greens", tweak = 1.3)
pred3_pruned <-(predict(pruned_model3, newdata = test_data,type="class"))
confusionMatrix(pred3_pruned, test_data$Class)  #check accuracy
mean(pred3_pruned==test_data$Class)
#================================================================
# Training a DT using Information gain splitting strategy 

DT_newmodel4 <- rpart(Class ~ ., data = train_data, method = "class", parms=list(split = "information"),
                      control = rpart.control(cp = 0,maxdepth = 5, minsplit =20))

summary(DT_newmodel4)
#Plot Decision Tree
prp(DT_newmodel4, box.palette = "Greens", tweak = 1.3)
# Examine the complexity plot
printcp(DT_newmodel4)
plotcp(DT_newmodel4)
pred4 <-(predict(DT_newmodel4, newdata = test_data,type="class"))
confusionMatrix(pred4, test_data$Class)  #check accuracy
mean(pred4==test_data$Class)
opt <- which.min(DT_newmodel4$cptable[,"xerror"])
cp <- DT_newmodel4$cptable[opt, "CP"]
pruned_model4 <- prune(DT_newmodel4,cp)
prp(pruned_model4, box.palette = "Greens", tweak = 1.3)
pred4_pruned <-(predict(pruned_model4, newdata = test_data,type="class"))
confusionMatrix(pred4_pruned, test_data$Class)  #check accuracy
mean(pred4_pruned==test_data$Class)
#==================================================================
# Training a DT using gain Ratio splitting strategy 

DT_newmodel5 <- rpart(Class ~ ., data = train_data, method = "class", parms=list(split = "gain"),
                      control = rpart.control(cp = 0,maxdepth = 5, minsplit =20))

summary(DT_newmodel5)
#Plot Decision Tree
prp(DT_newmodel5, box.palette = "Greens", tweak = 1.3)
# Examine the complexity plot
printcp(DT_newmodel5)
plotcp(DT_newmodel5)
pred5 <-(predict(DT_newmodel5, newdata = test_data,type="class"))
confusionMatrix(pred5, test_data$Class)  #check accuracy
mean(pred5==test_data$Class)
opt <- which.min(DT_newmodel5$cptable[,"xerror"])
cp <- DT_newmodel5$cptable[opt, "CP"]
pruned_model5 <- prune(DT_newmodel5,cp)
prp(pruned_model5, box.palette = "Greens", tweak = 1.3)
pred5_pruned <-(predict(pruned_model5, newdata = test_data,type="class"))
confusionMatrix(pred5_pruned, test_data$Class)  #check accuracy
mean(pred5_pruned==test_data$Class)
#===================================================================
# Training a DT using Information Gini splitting strategy 

DT_newmodel6 <- rpart(Class ~ ., data = train_data, method = "class", parms=list(split = "gini"),
                      control = rpart.control(cp = 0,maxdepth = 5, minsplit =20))

summary(DT_newmodel6)
#Plot Decision Tree
prp(DT_newmodel2, box.palette = "Greens", tweak = 1.3)
# Examine the complexity plot
printcp(DT_newmodel6)
plotcp(DT_newmodel6)
pred6 <-(predict(DT_newmodel6, newdata = test_data,type="class"))
confusionMatrix(pred6, test_data$Class)  #check accuracy
mean(pred6==test_data$Class)
opt <- which.min(DT_newmodel6$cptable[,"xerror"])
cp <- DT_newmodel6$cptable[opt, "CP"]
pruned_model6 <- prune(DT_newmodel6,cp)
prp(pruned_model6, box.palette = "Greens", tweak = 1.3)
pred6_pruned <-(predict(pruned_model6, newdata = test_data,type="class"))
confusionMatrix(pred6_pruned, test_data$Class)  #check accuracy
mean(pred6_pruned==test_data$Class)

