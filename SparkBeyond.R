#####My Home Test################
#---Liran Ben-Zion--------
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

plot(data$price~data$waterfront)      # less than 0.5% have waterfront
table(data$view)            # approax 10% has other than zero views 1,2,3,4
table(data$bedrooms)        # mostly bedrooms are between 1-6
table(data$bathrooms)      # mostly accounts for 1,1.5,1.75,2,2.25,2.5,2.5,3,3.5 total 30
table(data$condition)      # mostly 3 then 4 then 5 then 2 then 1
table(data$grade)          # mostly 5-9 out of 1-12
table(data$floors)         # mostly 1 and 2 then 1.5

#----------------------------corrleations-----------------------------------------
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
               mapping = aes(color = "dark green"),
               axisLabels="show")
plot1

# Sqft vs Price- 6 figures arranged in 3 rows and 2 columns
attach(mtcars)
par(mfrow=c(4,2))
plot(data$sqft_living,log(data$price), main="Scatterplot of sqft_living vs. price", xlab = "sqft_living",ylab = "Price",col="blue")
plot(data$sqft_lot,log(data$price), main="Scatterplot of wt vs disp", xlab = "sqft_lot",ylab = "Price",col="red")
plot(data$sqft_living15,log(data$price),main="Scatterplot of wt vs disp", xlab = "sqft_living15",ylab = "Price",col="green")
plot(data$sqft_lot15,log(data$price),main="Scatterplot of wt vs disp", xlab = "sqft_lot15",ylab = "Price",col="purple")
plot(data$sqft_above,log(data$price), main="Scatterplot of wt vs disp", xlab = "sqft_above",ylab = "Price",col="dark red")
plot(data$sqft_basement,log(data$price), main="Scatterplot of wt vs disp", xlab = "sqft_basement",ylab = "Price",col="dark blue")

#Price vs Bedrooms
p1<-ggplot(data,aes(bedrooms, log(price)), main="Scatterplot of Bedrooms vs. price", xlab = "bedrooms",ylab = "Price",col="blue")
p1+geom_bar(stat = "identity")

plot(data$price~data$bedrooms)

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


#Linear Regression-Full model
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
Linear_Predictions_model5=predict(model5)
RMSE=RMSE(data$price,Linear_Predictions_model5)

#----------------------try all caret models-------------
install.packages.compile.from.source = "always"
install.packages(c("feather","tidyr"), type = "both")

library(caret)
library(foreach)
library(doParallel)

#rm(list=ls())
gc()

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
#c('adaboost', 'AdaBoost.M1', 'amdai', 'ANFIS', 'vglmAdjCat', 'AdaBag', 'treebag', 'bagFDAGCV', 'bagFDA', 'logicBag', 'bagEarth', 'bagEarthGCV', 'bag', 'bartMachine', 'bayesglm', 'brnn', 'bridge', 'blassoAveraged', 'binda', 'ada', 'gamboost', 'glmboost', 'BstLm', 'LogitBoost', 'bstSm', 'blackboost', 'bstTree', 'J48', 'C5.0', 'rpart', 'rpart1SE', 'rpart2', 'rpartScore', 'chaid', 'cforest', 'ctree', 'ctree2', 'vglmContRatio', 'C5.0Cost', 'rpartCost', 'cubist', 'vglmCumulative', 'deepboost', 'dda', 'dwdPoly', 'dwdRadial', 'DENFIS', 'enet', 'randomGLM', 'xgbDART', 'xgbLinear', 'xgbTree', 'elm', 'RFlda', 'fda', 'FIR.DM', 'FRBCS.CHI', 'FH.GBML', 'SLAVE', 'GFS.FR.MOGUL', 'GFS.THRIFT', 'FRBCS.W', 'gaussprLinear', 'gaussprPoly', 'gaussprRadial', 'gamLoess', 'bam', 'gam', 'gamSpline', 'glm', 'glmStepAIC', 'gpls', 'GFS.LT.RS', 'glmnet', 'glmnet_h2o', 'gbm_h2o', 'protoclass', 'hda', 'hdda', 'hdrda', 'HYFIS', 'icr', 'kknn', 'knn', 'svmLinearWeights2', 'svmLinear3', 'lvq', 'lars', 'lars2', 'lssvmLinear', 'lssvmPoly', 'lssvmRadial', 'lda', 'lda2', 'stepLDA', 'dwdLinear', 'lm', 'leapBackward', 'leapForward', 'leapSeq', 'lmStepAIC', 'svmLinearWeights', 'loclda', 'logreg', 'LMT', 'Mlda', 'mda', 'manb', 'avNNet', 'M5Rules', 'M5', 'monmlp', 'mlp', 'mlpWeightDecay', 'mlpWeightDecayML', 'mlpML', 'msaenet', 'mlpSGD', 'mlpKerasDropout', 'mlpKerasDropoutCost', 'mlpKerasDecay', 'mlpKerasDecayCost', 'earth', 'gcvEarth', 'naive_bayes', 'nb', 'nbDiscrete', 'awnb', 'pam', 'glm.nb', 'mxnet', 'mxnetAdam', 'neuralnet', 'nnet', 'pcaNNet', 'rqnc', 'null', 'nnls', 'ORFlog', 'ORFpls', 'ORFridge', 'ORFsvm', 'ownn', 'polr', 'parRF', 'partDSA', 'kernelpls', 'pls', 'simpls', 'widekernelpls', 'plsRglm', 'PRIM', 'pda', 'pda2', 'PenalizedLDA', 'penalized', 'plr', 'multinom', 'ordinalNet', 'krlsPoly', 'pcr', 'ppr', 'qda', 'stepQDA', 'qrf', 'qrnn', 'rqlasso', 'krlsRadial', 'rbf', 'rbfDDA', 'rFerns', 'ordinalRF', 'ranger', 'Rborist', 'rf', 'extraTrees', 'rfRules', 'rda', 'rlda', 'regLogistic', 'RRF', 'RRFglobal', 'relaxo', 'rvmLinear', 'rvmPoly', 'rvmRadial', 'ridge', 'foba', 'Linda', 'rlm', 'rmda', 'QdaCov', 'rrlda', 'RSimca', 'rocc', 'rotationForest', 'rotationForestCp', 'JRip', 'PART', 'xyf', 'nbSearch', 'sda', 'CSimca', 'FS.HGD', 'C5.0Rules', 'C5.0Tree', 'OneR', 'sdwd', 'sparseLDA', 'smda', 'spls', 'spikeslab', 'slda', 'snn', 'dnn', 'gbm', 'SBC', 'superpc', 'svmBoundrangeString', 'svmRadialWeights', 'svmExpoString', 'svmLinear', 'svmLinear2', 'svmPoly', 'svmRadial', 'svmRadialCost', 'svmRadialSigma', 'svmSpectrumString', 'blasso', 'lasso', 'tan', 'tanSearch', 'awtan', 'evtree', 'nodeHarvest', 'vbmpRadial', 'WM', 'wsrf')

caret_models <-c("xgbDART","xgbLinear","ppr","gamLoess","cubist","glm","lm","foba","monmlp","glmStepAIC","lmStepAIC","lars2","rqnc","lars","extraTrees","glmnet","qrf","penalized","bagEarthGCV","bagEarth","xgbTree","Rborist","glmboost","M5Rules","M5","ranger","parRF","nnls","rf","RRFglobal","earth","gcvEarth","msaenet","RRF","relaxo","bstTree","leapBackward","blackboost","gbm","nodeHarvest","treebag","kknn","evtree","rpart1SE","rpart2","icr","rpart","partDSA","leapForward","leapSeq","kernelpls","pls","simpls","widekernelpls","BstLm","pcr","knn","svmRadial","svmRadialCost","xyf","svmRadialSigma","null","neuralnet","mlpWeightDecayML","mlp","rfRules","mlpWeightDecay","gaussprRadial","dnn","mlpML","rqlasso","rvmRadial","avNNet","nnet","pcaNNet","superpc","rbfDDA","svmLinear3","svmPoly","randomGLM","svmLinear2","svmLinear")
caret_models<-c("xgbLinear")

# create a log file
fname <- 'C:\\Users\\benzionl\\Desktop\\SB\\ttt.log'
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


model_xgbLinear1<-train(form = price~sqft_living + bedrooms + bathrooms + grade + sqft_above + zipcode, data=Train, trControl = trCtrl,method='xgbLinear')
print(model_xgbLinear1)
plot(model_xgbLinear1)
varImp(object=model_xgbLinear1)
model_xgbLinear<-train(form = price~., data=Train, trControl = trCtrl,method='xgbLinear')
# summarizing the model
print(model_xgbLinear)
plot(model_xgbLinear)
varImp(object=model_xgbLinear)
#Plotting Varianle importance for model_xgbLinear
plot(varImp(object=model_xgbLinear),main="model_xgbLinear - Variable Importance")

#Predictions
predictions<-predict.train(object=model_xgbLinear,Test[,-Test$price],type="raw")
table(predictions)
confusionMatrix(predictions,Test[,Test$price])


model_rf<-train(form = price~., data=Train, trControl = trCtrl,method='rf')
prediction_xgb4 = predict(model_xgb4, test)

model_nnet<-train(form = price~., data=Train, trControl = trCtrl,method='nnet')

#---------------------Entropy for feature selection-----------------
library(entropy)



