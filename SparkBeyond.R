#####-------------------------------------- My Home Test-----------------------------################
#------------------------------------owner: Liran Ben-Zion-------------------------------------------
#------------------------------------email- bzliran@gmail.com----------------------------------------

rm(list=ls())
library(dplyr)
library(data.table)
library(mice)
library(ggplot2)
library(corrplot)
library(RCurl)
library(Hmisc)

#--------------------------------load data-------------------------------------------------------------
x <- getURL("https://raw.githubusercontent.com/Liranbz/SB/master/kc_house_data.csv")
data <- read.csv(text = x,header=TRUE, sep=",")
#data <- read.csv(file="C:\\Users\\benzionl\\Desktop\\SB\\kc_house_data.csv", header=TRUE, sep=",")
str(data)
names(data)
summary(data)

#-----------------------------Data preparation--------------------------------------------
data<-data[,-1] # ramove id col
data[,c(9:11,16)]<-lapply(data[,c(9:11,16)],as.factor) #change variables (view, condition, grade,zipcode) to factors 
data$date <- as.Date(as.Date(as.character(data$date),"%Y%m%d")) # Formatting date as date format from string
data$age <- as.numeric(format(data$date, "%Y")) - data$yr_built # Creating a variable column name 'age' of house

#Creating a variable column name 'is_basement`
data$is_basement<-0
data$is_basement[data$sqft_basement > 0]<-1
data$is_basement=factor(data$is_basement)

# Column 'rate' is created which is selling price per square feet
data$rate <- data$price/data$sqft_living

# Checking how many NA are there
missing_values_summary_table<-t(md.pattern(data, plot = TRUE))

#----------------------------corrleations-----------------------------------------
library(GGally)
numeric_data<-select_if(data, is.numeric)
cor <- rcorr(as.matrix(numeric_data))
M <- cor$r
p_mat <- cor$P
col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))
corrplot(M, method = "color", col = col(200),  
         type = "upper", order = "hclust", 
         addCoef.col = "black", # Add coefficient of correlation
         tl.col = "darkblue", tl.srt = 45, #Text label color and rotation
         # Combine with significance level
         p.mat = p_mat, sig.level = 0.01,  
         # hide correlation coefficient on the principal diagonal
         diag = FALSE 
)

#-------------Plots---------------------------
# histogram of prices
hist(data$price)
## Checking Relationship between price, bedrooms, bathrooms, sqft_living and sqft lot
plot1<-ggpairs(data=data, columns=2:6,
               mapping = aes(color = "dark red"),
               axisLabels="show")
plot1

# Sqft vs Price- 6 figures arranged in 3 rows and 2 columns
attach(mtcars)
par(mfrow=c(4,2))
plot(data$sqft_living,log(data$price), main="Scatterplot of sqft_living vs. price", xlab = "sqft_living",ylab = "Price",col="blue")
plot(data$sqft_lot,log(data$price), main="Scatterplot of sqft_lot vs. price", xlab = "sqft_lot",ylab = "Price",col="red")
plot(data$sqft_living15,log(data$price),main="Scatterplot of sqft_living15 vs. price", xlab = "sqft_living15",ylab = "Price",col="green")
plot(data$sqft_lot15,log(data$price),main="Scatterplot of sqft_lot15 vs price", xlab = "sqft_lot15",ylab = "Price",col="purple")
plot(data$sqft_above,log(data$price), main="Scatterplot of sqft_above vs price", xlab = "sqft_above",ylab = "Price",col="dark red")
plot(data$sqft_basement,log(data$price), main="Scatterplot of sqft_basement vs price", xlab = "sqft_basement",ylab = "Price",col="dark blue")

#Price vs Bedrooms
p1<-ggplot(data,aes(bedrooms,price), main="Scatterplot of Bedrooms vs. price", 
           xlab = "bedrooms",ylab = "Price",col="blue")+ xlim(1,10)
p1+geom_bar(stat = "identity")

plot(data$price~factor(data$bedrooms), main="plot of Bedrooms vs. price" )

#------------sets for train and test----
#sample data
row_sampler=function(df){
  set.seed(789)
  n_rows_data=(nrow(df))
  random_row_nums <-sample(x=1:n_rows_data,size=n_rows_data,replace = FALSE)
  return(random_row_nums)
}

Train_test_division=function(train_fraction,df){
  random_rows=row_sampler(df)
  Division_point=round(nrow(df)*train_fraction,digits = 0)
  Train_indices=random_rows[1:Division_point]
  Test_indices=random_rows[(1+Division_point):length(random_rows)]
  Train=df[Train_indices,]
  Test=df[Test_indices,]
  return(list(Train=Train,Test=Test))
}
Train_test_Data=Train_test_division(0.75,data)
Train=Train_test_Data$Train
Test=Train_test_Data$Test


#-----------------------Linear Regression models------------------------------------------------------
full_model=lm(formula = Train$price~.,data = Train)
summary(full_model)

Linear_Predictions=predict(full_model)
Test$predicted_price=predict(full_model,Test)
RMSE=RMSE(Test$price,Test$predicted_price)

#model 3
model3 <- lm(price~ sqft_living + bedrooms + bathrooms + grade + sqft_above + zipcode,data = data)
summary(model3)

#model 4-log to price
model4 <- lm(log(price)~ sqft_living + bedrooms + bathrooms + grade + sqft_above + zipcode,data = data)
summary(model4)

#model 5-log to price+vars
model5 <- lm(log(price)~ log(sqft_living) + bedrooms + bathrooms + grade + log(sqft_above) + zipcode+age+lat+long,data = data)
summary(model5)
Linear_Predictions_model5=predict(model5,Test)
RMSE=RMSE(Test$price,Linear_Predictions_model5)

#---------------------------------------Machine Learning models-------------------------
#----------------------try all ML regression models from caret library-------------
install.packages.compile.from.source = "always"
install.packages(c("feather","tidyr"), type = "both")

library(caret)
library(foreach)
library(doParallel)
gc()
setwd("C:\\Users\\benzionl\\Desktop\\SB")
#--------train control-----
trCtrl <- trainControl(
  method = "repeatedcv"
  , number = 2
  , repeats = 5
  , allowParallel = TRUE
)

# sample with 300 observations
ttt <- sample_n(data,300)
str(ttt)

# all caret models
names(getModelInfo())

#regression ML models only:
caret_models <-c("xgbDART","xgbLinear","ppr","gamLoess","cubist","glm","lm","foba","monmlp","glmStepAIC","lmStepAIC","lars2","rqnc","lars","extraTrees","glmnet","qrf","penalized","bagEarthGCV","bagEarth","xgbTree","Rborist","glmboost","M5Rules","M5","ranger","parRF","nnls","rf","RRFglobal","earth","gcvEarth","msaenet","RRF","relaxo","bstTree","leapBackward","blackboost","gbm","nodeHarvest","treebag","kknn","evtree","rpart1SE","rpart2","icr","rpart","partDSA","leapForward","leapSeq","kernelpls","pls","simpls","widekernelpls","BstLm","pcr","knn","svmRadial","svmRadialCost","xyf","svmRadialSigma","null","neuralnet","mlpWeightDecayML","mlp","rfRules","mlpWeightDecay","gaussprRadial","dnn","mlpML","rqlasso","rvmRadial","avNNet","nnet","pcaNNet","superpc","rbfDDA","svmLinear3","svmPoly","randomGLM","svmLinear2","svmLinear")

# create a log file
fname <- 'ttt.log'
cat(paste0(paste('counter','method','user','system','elapsed','mse',sep=','),'\n'), file=fname)

counter <- 0
for(current_method in caret_models) {
  counter <- counter+1
  print(paste('Trying model #',counter,'/',length(caret_models),current_method))
  tryCatch({
    registerDoSEQ() # to disable "invalid connection" error
    profiler <- system.time(model.1 <- train(form = price~., data=ttt, trControl = trCtrl, method=current_method))
    mse <- mean((predict(model.1)-ttt$price)^2)
    # write status of current method to log file
    status <- paste(counter,current_method,profiler[[1]],profiler[[2]],profiler[[3]],mse,sep=',')
    cat(paste0(status,'\n'), file=fname, append=T)
  }, error = function(err_cond) {
    print(paste('An error with model',current_method))
    cat(paste('Error with model #',counter,'/',length(caret_models),current_method,'Error',err_cond,'\n'), file=fname, append=T)
  })
}

#----------------------------------------model_xgbLinear with some features-------------------------------------------
model_xgbLinear1<-train(form = price~sqft_living + bedrooms + bathrooms + grade + sqft_above + zipcode, data=Train, trControl = trCtrl,method='xgbLinear')

summary(model_xgbLinear1) # summarizing the model
print(model_xgbLinear1)
plot(model_xgbLinear1)
varImp(object=model_xgbLinear1)
plot(varImp(object=model_xgbLinear1),main="model_xgbLinear - Variable Importance, 6 features")

#Predictions
predictions1<-predict.train(object=model_xgbLinear1,Test[,-Test$price],type="raw")
RMSE_model_xgbLinear1=RMSE(predictions1,Test$price)

#----------------------------------------model_xgbLinear- all features-------------------------------------------
model_xgbLinear<-train(form = price~., data=Train, trControl = trCtrl,method='xgbLinear')

print(model_xgbLinear)  # summarizing the model
plot(model_xgbLinear)
varImp(object=model_xgbLinear)
#Plotting Varianle importance for model_xgbLinear
#plot(varImp(object=model_xgbLinear),main="model_xgbLinear - Variable Importance")

#Predictions
predictions<-predict.train(object=model_xgbLinear,Test[,-Test$price],type="raw")
RMSE_model_xgbLinear1=RMSE(predictions,Test$price)

#----------------------------------------nnet with features--------------------------------------------
model_nnet<-train(form = price~sqft_living + bedrooms + bathrooms + grade + sqft_above + zipcode, data=Train, trControl = trCtrl,method='nnet')

summary(model_nnet) # summarizing the model
print(model_nnet)
plot(model_nnet)
varImp(object=model_nnet)
plot(varImp(object=model_nnet),main="model_nnet - Variable Importance, 6 features")

#Predictions
predictions_nnet<-predict.train(object=model_nnet,Test[,-Test$price],type="raw")
RMSE_model_nnet=RMSE(predictions_nnet,Test$price)

#----------------------------------------RF- with features-------------------------------------------

model_rf<-train(form = price~sqft_living + bedrooms + grade + sqft_above + zipcode+lat+long, data=Train, trControl = trCtrl,method='rf')

summary(model_rf)
print(model_rf)
plot(model_rf)
varImp(object=model_rf)
plot(varImp(object=model_rf),main="model_rf - Variable Importance, 6 features")

#Predictions
predictions_rf<-predict.train(object=model_rf,Test[,-Test$price],type="raw")
RMSE_model_xgbLinear1=RMSE(predictions_rf,Test$price)



#---------------------------------jointEntropy--------------------------------

a=c(1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0)
b=c(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0)
labels=c(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)

jointEntropy=function(a,b,labels){
  
#calc entropy for array a
t_a=table(a,labels)
prop1=t_a[1,]/sum(t_a[1,])
prop2=t_a[2,]/sum(t_a[2,])

H1=-(prop1[1]*log2(prop1[1]))-(prop1[2]*log2(prop1[2]))
H2=-(prop2[1]*log2(prop2[1]))-(prop2[2]*log2(prop2[2]))

entropy_a=(table(a)[1]/length(a))*H1 +(table(a)[2]/length(a))*H2

#calc entropy for array b
t_b=table(b,labels)
prop1=t_b[1,]/sum(t_b[1,])
prop2=t_b[2,]/sum(t_b[2,])

H1=-(prop1[1]*log2(prop1[1]))-(prop1[2]*log2(prop1[2]))
H2=-(prop2[1]*log2(prop2[1]))-(prop2[2]*log2(prop2[2]))

entropy_b=(table(b)[1]/length(b))*H1 +(table(b)[2]/length(b))*H2

#find the entropy value
entropy=sum(entropy_a,entropy_b)

return (abs(entropy - 0.344) < 0.01)
}


#get results from function
jointEntropy(a,b,labels)
