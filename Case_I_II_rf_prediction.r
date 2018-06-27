# This code is made for event prediction using Random Forest model.
#				Prepared by University of Missouri - Columbia (May 2018)
# 1. Prepare three encoded data set: Training.csv, ValTesting.csv, Testing.csv
# 2. Case -I and Case -II are the same code
# 3. Edit your file path below





##### CLEARNING #####
rm(list=ls())						#<---- Remove all including functions
cat("\014")							#<---- Cleaning ALL Console
options(warn=-1)






################################################### PATH ###################################################
#setwd("E:/Dropbox/project/SHRP2_NDS/SHRP2_NDS_Phase_II/GUI/Younger_Drivers/6_Prediction")	# Windows OS: Incase of the directory is not the same
setwd("/Users/YohanChang/Dropbox/project/SHRP2_NDS/SHRP2_NDS_Phase_II/GUI/Younger_Drivers/6_Prediction")	# Mac OS: Incase of the directory is not the same
################################################### PATH ###################################################






(paths <- getwd())


# Load Training and Val_Testing data 
Val_Testing<-read.csv(file=sprintf("%s/ValTesting.csv",paths),  header=TRUE)	# Load validation testing set
Training<-read.csv(file=sprintf("%s/Training.csv",paths),  header=TRUE)	# Load training set
Testing<-read.csv(file=sprintf("%s/Testing.csv",paths),  header=TRUE)	# Load TRUE testing set


SoT=ncol(Training);				#------ Define size of Training dataset
SoT2=nrow(Val_Testing);				#------ Define size of Training dataset
True_Value=Val_Testing[,SoT]		#------ Save True Outcomes 

# Set a threshold value for normalization
Best_Acc=00
# Save all steops' calculations
Accuracys<-data.frame("Threshold"=0, "Accuracy"=0,"TN"=0, "FN"=0, "TP"=0, "FP"=0)



####################################### PREDICTION START with RANDOM FORESTS
library(randomForest)
set.seed(as.numeric(Sys.time()))


# Find best number of myry
tA<-tuneRF(Training[,-SoT], Training[,SoT], stepFactor=1.5, plot=FALSE, trace=FALSE, doBest=FALSE);
Found_tA=min(tA);
for (i in 1:nrow(tA)){
	if (tA[i,2]==Found_tA){set_RF=tA[i,1]}}						#------ 'set_RF' is best number of myry
remove('i','Found_tA','tA')										#------ clearning unneccery variables


#------ Find best number of trees with best myry
NofTrees=980
event.rf<-randomForest(x=Training[,-SoT], y=Training[,SoT], xtest=Val_Testing[,-SoT], ytest=Val_Testing[,SoT], 
	ntree=NofTrees, mtry=set_RF, replace=FALSE, samplesize=ceiling(.632*nrow(Training)), importance=TRUE, do.trace=FALSE, keep.forest=TRUE)

#------ Find best number of trees
Fount_trr=min(event.rf$mse);
for (i in 1:NofTrees){
	if (event.rf$mse[i]==Fount_trr){NofTrees=i}}						#------ 'set_RF' is best number of myry
remove('i','Fount_trr')										#------ clearning unneccery variables




#------ Do actual RF model with fine-tuned values
event.rf<-randomForest(x=Training[,-SoT], y=Training[,SoT], xtest=Val_Testing[,-SoT], ytest=Val_Testing[,SoT], 
	ntree=NofTrees, mtry=set_RF, replace=FALSE, samplesize=ceiling(.632*nrow(Training)), importance=TRUE, do.trace=FALSE, keep.forest=TRUE)
#------ Do prediction with the fine-tuned RF model



# Find best threshold for normalizaion from Validation Testing results using Hill climbing optimization
for (ThresholdV in seq(0.1,2,0.001)){
	test_predicted<-predict(event.rf, Val_Testing[,-SoT])

	#------ Check accuracy
	TP=0; TN=0; FP=0; FN=0; Acc=0;
	for (i in 1:SoT2){
		#---- Normalization
		if (test_predicted[i]<=ThresholdV){test_predicted[i]<-0}	#--- BASELINE
		if (test_predicted[i]>ThresholdV){test_predicted[i]<-1}	#--- CRASH

		#---- Accuracy 
		if (test_predicted[i]==True_Value[i]){Acc=Acc+1}

		#---- Confusion Matrix
		if (test_predicted[i]==0 & True_Value[i]==0){TN=TN+1}
		if (test_predicted[i]==0 & True_Value[i]==1){FN=FN+1}
		if (test_predicted[i]==1 & True_Value[i]==1){TP=TP+1}
		if (test_predicted[i]==1 & True_Value[i]==0){FP=FP+1}
	}

	sprintf("prediction %.3f %%",(Acc/SoT2)*100)
	# Keep best accuracy resutls and threshold for normalizaion
	if (((Acc/SoT2)*100) > Best_Acc){
		Best_Acc<-((Acc/SoT2)*100)
		Best_ThresholdV<-ThresholdV
	}
	Accuracys<-rbind(Accuracys, data.frame("Threshold"=ThresholdV, "Accuracy"=(Acc/SoT2)*100, "TN"=TN, "FN"=FN, "TP"=TP, "FP"=FP))
}
remove('i')

#------- Record Validation output
write.table(test_predicted, file=sprintf("%s/Output_for_Validation_Data.csv",paths), sep=",")
IMPORTAN<-round(importance(event.rf),2);
write.table(IMPORTAN,file=sprintf("%s/Variable_Importance_for_Validation_Data.csv",paths), sep=",")
#------------VALIDATION END---------------------------------------------------------------








##### Testing with Actual Values ##### 
test_predicted<-predict(event.rf, Testing[,-SoT])
TrueValue<-test_predicted
TP=0; TN=0; FP=0; FN=0; Acc=0;
SoT2=nrow(Testing);				#------ Define size of Training dataset
True_Value=Testing[,SoT]		#------ Save True Outcomes 


# Normalization for the Predicted_Results
for (i in 1:SoT2){
	#---- Normalization
	if (test_predicted[i]<=Best_ThresholdV){test_predicted[i]<-0}	#--- BASELINE
	if (test_predicted[i]>Best_ThresholdV){test_predicted[i]<-1}	#--- CRASH
	
	#---- Accuracy 
	if (test_predicted[i]==True_Value[i]){Acc=Acc+1}
	
	#---- Confusion Matrix
	if (test_predicted[i]==0 & True_Value[i]==0){TN=TN+1}
	if (test_predicted[i]==0 & True_Value[i]==1){FN=FN+1}
	if (test_predicted[i]==1 & True_Value[i]==1){TP=TP+1}
	if (test_predicted[i]==1 & True_Value[i]==0){FP=FP+1}
}


# Generating reports and exporing all results
sprintf("prediction %.3f %%",(Acc/SoT2)*100)
Testing_Accuracy<-(Acc/SoT2)*100
write.table(test_predicted, file=sprintf("%s/Calibrated_Output_for_Testing_Data.csv",paths), sep=",")
write.table(TrueValue, file=sprintf("%s/raw_Output_for_Testing_Data.csv",paths), sep=",")
IMPORTAN<-round(importance(event.rf),2);
write.table(IMPORTAN,file=sprintf("%s/Variable_Importance_for_Testing_Data.csv",paths), sep=",")



# Final Results Display
cat("\014")
sprintf("Validation accuracy = % .3f%%", Best_Acc)
sprintf("Testing accuracy = % .3f%%", Testing_Accuracy)
####################################### PREDICTION END
varImpPlot(event.rf)
