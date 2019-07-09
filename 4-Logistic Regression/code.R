library("dplyr")
library("readxl")
library("reshape")
library("reshape2")


data <- read_excel("eBayAuctions.xls")


colnames(data)[colnames(data)=="Competitive?"] <- "Competitive"

names(data) <- tolower(names(data))


for(level in unique(data$category)){
  data[paste("dummy", level, sep = "_")] <- ifelse(data$category == level, 1, 0)
}

for(level in unique(data$currency)){
  data[paste("dummy", level, sep = "_")] <- ifelse(data$currency == level, 1, 0)
}

for(level in unique(data$endday)){
  data[paste("dummy", level, sep = "_")] <- ifelse(data$endday == level, 1, 0)
}

for(level in unique(data$duration)){
  data[paste("dummy", level, sep = "_")] <- ifelse(data$duration == level, 1, 0)
}

getListOfSimilarColumns<-function(pivot_table){
  l2<-list()
  j<-1
  col<-colnames(pivot_table)
  l<-list(col[1])
  k<-2
  x<-pivot_table[,1]
  for(i in 2:(dim(pivot_table)[2]-1)){
    if(abs(x-pivot_table[,i])<0.05){
      #l <- c(l,col[i])
      l[[k]]<-col[i]
      k<-k+1
    }
    else{
      x<-pivot_table[,i]
      l2[[j]]<-l
      j<-j+1
      l<-list(col[i])
      k<-2
    }
  }
  return (l2)
}

applyOr<-function(x){
  y<-0
  for(i in 1:length(x)){
    y<-as.numeric(y|x[i])
  }
  return (y)
}

mergeSimilarColumns<-function(table, data){
  temp_table<-getListOfSimilarColumns(table)
  for(i in 1:length(temp_table)){
    if(length(temp_table[[i]])>1){
      l<-c()
      newcolname<-"merged"
      for(j in 1:length(temp_table[[i]])){
        n<-paste("dummy",temp_table[[i]][[j]],sep="_")
        newcolname<-paste(newcolname,temp_table[[i]][[j]],sep="_")
        l<-c(l,n)
      }
      df<-data[,l]
      #temp<-paste("merged",i,sep = "_")
      data[newcolname]<-apply(df, 1, applyOr)
      for(j in 1:length(temp_table[[i]])){
        n<-paste("dummy",temp_table[[i]][[j]],sep="_")
        data[n]<-NULL
      }
    }
  }
  
  return (data)
}

data_melt <- melt(data, id.vars = c("category", "currency","endday","duration") , measure.vars = "competitive")

pivot_table_category<-dcast(data_melt, variable ~ category,mean)
pivot_table_category<-sort(pivot_table_category[,])
data<-mergeSimilarColumns(pivot_table_category,data)
data["category"]<-NULL

pivot_table_currency<-dcast(data_melt, variable ~ currency,mean)
pivot_table_currency<-sort(pivot_table_currency[,])
data<-mergeSimilarColumns(pivot_table_currency,data)
data["currency"]<-NULL

pivot_table_endday<-dcast(data_melt, variable ~ endday,mean)
pivot_table_endday<-sort(pivot_table_endday[,])
data<-mergeSimilarColumns(pivot_table_endday,data)
data["endday"]<-NULL

pivot_table_duration<-dcast(data_melt, variable ~ duration,mean)
pivot_table_duration<-sort(pivot_table_duration[,])
data<-mergeSimilarColumns(pivot_table_duration,data)
data["duration"]<-NULL


#create training and validation data

require('caret')
set.seed(123)

train_ind <- createDataPartition(data$competitive, p = 0.6, 
                                 list = FALSE, 
                                 times = 1)

train <- data[train_ind, ]
validation <- data[-train_ind, ]


fit.all <- glm(competitive~.,family=binomial(link='logit'),data=train)

summary(fit.all)

fit.single<-glm(competitive~dummy_EverythingElse,family=binomial(link='logit'),data=train)

summary(fit.single)

validation$pred<-predict(fit.all,validation,type='response')
validation$pred <- ifelse(validation$pred > 0.5, 1, 0)
Accuracy <- mean(validation$competitive==validation$pred)


coeff<-fit.all$coefficients

sort(abs(coeff))

fit.reduced<-glm(competitive~openprice+closeprice+`dummy_Health/Beauty`+dummy_GBP+dummy_5+dummy_Mon+`merged_Pottery/Glass_Automotive_Jewelry`+`dummy_Coins/Stamps`,family=binomial(link='logit'),data=train)

validation$predreduced<-predict(fit.reduced,validation,type='response')
validation$predreduced <- ifelse(validation$predreduced > 0.5, 1, 0)
Accuracy_reduced <- mean(validation$competitive==validation$predreduced)

anova(fit.reduced, fit.all, test='Chisq')

summary(fit.reduced)

DispersionT<-(fit.reduced$deviance/fit.reduced$df.residual)
if(as.integer(DispersionT)==1){
  print("well fitted model")
}else{
  print("dispersed model")
}
