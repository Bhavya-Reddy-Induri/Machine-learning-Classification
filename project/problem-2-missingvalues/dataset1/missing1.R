library(missForest)
library(softImpute)
library(mice)
rmse <- function(error)
{
  sqrt(mean(error^2))
}

dataset = read.csv("C:\\Users\\Sai\\Desktop\\My ML\\problem-2-missingvalues\\dataset1\\MissingData1.csv",header=FALSE)
dataset[dataset==1.000e+99]<-NA
newdata <- na.omit(dataset)  
newdatamis <- prodNA(newdata, noNA = 0.04) 
#MISS FOREST####

set.seed(100)
newdatamis.imp <- missForest(newdatamis,maxiter = 10, ntree = 500, mtry=4,verbose = TRUE)

###MICE#
imputed_Data <- mice(newdatamis, m=5, maxit = 50, method = 'pmm', seed = 500)
imputed_Data1 <- mice(dataset, m=5, maxit = 50, method = 'pmm', seed = 500)
completeData1 <- complete(imputed_Data1,5)
completeData <- complete(imputed_Data,5)
err<-newdata-completeData
rmse(err)

#error
newdatamis.imp$OOBerror
dataa<-newdatamis.imp$ximp
err<-newdata-dataa
rmse(err)

# output

write.table(completeData1, "induriMissingResult1.txt", sep="\t")