#####My Home Test################
#---Liran Ben-Zion--------
rm(list=ls())
library(dplyr)
library(data.table)
library(mice)
library(ggplot2)
library(corrplot)

data <- read.csv(file="C:\\Users\\benzionl\\Desktop\\SB\\kc_house_data.csv", header=TRUE, sep=",")
str(data)
names(data)
summary(data)

#--Data prepartion--
# ramove id col
data<-data[,-1]
#change specific variables to factors 
data[,c(9:11,17)]<-lapply(data[,c(9:11,17)],as.factor)
# Formatting date as date format from string
data$date <- as.Date(as.Date(as.character(data$date),"%Y%m%d"))
# Creating a variable column name 'age' , because year built doesn't make sense. Age at selling the home (after built ) matters
data$age <- as.numeric(format(data$date, "%Y")) - data$yr_built

# Same as done above, column 'renage', but renage has zero values which do not make sense, so making 0 values to null before
data$yr_renovated[data$yr_renovated == 0] <- NA
data$renage <- as.numeric(format(data$date, "%Y")) - data$yr_renovated
data<-data[,-15]

#Creating a variable column name 'is_basement`
data$is_basement<-0
data$is_basement[data$sqft_basement > 0]<-1
data$is_basement=factor(data$is_basement)
names(data)
# Checking how many NA are there
table(is.na(data$renage))   # only approax 5% houses are renovated)
table(data$waterfront)      # less than 0.5% have waterfront
table(data$view)            # approax 10% has other than zero views 1,2,3,4
table(data$bedrooms)        # mostly bedrooms are between 1-6
table(data$bathrooms)      # mostly accounts for 1,1.5,1.75,2,2.25,2.5,2.5,3,3.5 total 30
table(data$condition)      # mostly 3 then 4 then 5 then 2 then 1
table(data$grade)          # mostly 5-9 out of 1-12
table(data$floors)         # mostly 1 and 2 then 1.5

# Column 'rate' is created which is selling price per square feet
data$rate <- data$price/data$sqft_living

missing_values_summary_table<-t(md.pattern(data, plot = TRUE))
missing_values_summary_table<-as.data.frame(missing_values_summary_table)
missing_values_summary_table$size<-row.names(missing_values_summary_table)

#--corrleations
#----Numeric----
numeric_data<-select_if(data, is.numeric)
res <- cor(numeric_data)
round(res, 2)
corrplot(res, type = "upper", order = "hclust",tl.col = "black", tl.srt = 45)


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
Train_test_Data=Train_test_division(0.6,data)
Train=Train_test_Data$Train
Test=Train_test_Data$Test


#Linear Regression-Full model
LM_Data=select_if(data, is.numeric)
LM_Data<-as.data.frame(LM_Data)
Train_test_LM_Data=Train_test_division(0.8,LM_Data)

LM_Train=Train_test_LM_Data$Train
LM_Test=Train_test_LM_Data$Test

Linear_train=lm(formula = LM_Train$price~.,data = LM_Train)
summary(Linear_train)

Linear_Predictions=predict(Linear_train)

LM_Test$predicted_price=predict(Linear_train,LM_Test[,-c(data$price)])
RMSE=RMSE(LM_Train$price,LM_Test$predicted_price)


#model 2
model2 <- lm(price~ sqft_living + bedrooms + bathrooms + grade + sqft_above,data = data)
summary(model2)

#model 3
model3 <- lm(price~ sqft_living + bedrooms + bathrooms + grade + sqft_above + zipcode,data = data)
summary(model3)

#model 4-log to price
model4 <- lm(log(price)~ sqft_living + bedrooms + bathrooms + grade + sqft_above + zipcode,data = data)
summary(model4)

#model 5-log to price+vars
model5 <- lm(log(price)~ log(sqft_living) + bedrooms + bathrooms + grade + log(sqft_above) + zipcode,data = data)
summary(model5)
Linear_Predictions_model5=predict(model5)
RMSE=RMSE(data$price,Linear_Predictions_model5)



#------------------------Neural Network----------------------------------------
library(keras)
library(caret)
library(tensorflow)
library(tidyverse)
Train=select_if(Train, is.numeric)
y = Train[,c(2)]
x = as.matrix(Train[,-c(2)]) 
Cat_num=length(unique(Train[,11]))
# scale to [0,1]
x = as.matrix(apply(x, 2, function(x) (x-min(x))/(max(x) - min(x))))

# one hot encode classes / create DummyFeatures
levels(y) = 1:length(y)
y=as.matrix(Train[,2])
# create sequential model
model = keras_model_sequential()

# add layers
model %>%
  layer_dense(input_shape = ncol(x), units = 10, activation = "relu") %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 1, activation = "linear")
summary(model)

# add a loss function and optimizer
model %>%
  compile(
    loss = "mse",
    optimizer = "adam",
    metrics = ('mse')
  )
model
# fit model with our training data set.
fit = model %>%
  fit(
    x = x,
    y = y,
    shuffle = T,
    batch_size = 64,
    validation_split = 0.3,
    epochs = 15
  )
plot(fit)
###################
y_test = Test[,11]
x_test = Test[,-c(2,11)]
Cat_num=length(unique(Test[,11]))
# scale to [0,1]
x_test = as.matrix(apply(x_test, 2, function(x) (x-min(x))/(max(x) - min(x))))

# one hot encode classes / create DummyFeatures
levels(y_test) = 1:length(y_test)
y_test = as.matrix(Test[,11])

#6 B:
model %>% evaluate(x_test, y_test)
Predictions_vec=model %>% predict_classes(x_test)
Correct_pred=c(Predictions_vec==as.numeric(as.character(Test[,11])))
sum(Correct_pred)/length(Correct_pred)
#confmat
confmat <- confusionMatrix(Test[,11], as.factor(Predictions_vec))
confmat

