################################################################################
#                                                                              #
# Source file for Titanic: Machine Learning from Disaster Kaggle challenge     #
# This file contains every routine to manipulate, train and test the Titanic   #
# dataset.                                                                     #
#                                                                              #
# Lucas Alexandre Soares 16/07/2017                                            #
# lucassoares1793@gmail.com                                                    #
# NUsp: 9293265                                                                #
################################################################################

source("src/mlp.r")

titanic.prepare.data <- function(dataset.path = "dataset/train.csv", train = TRUE){

	# Features:
	# PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
	# Pclass is social class (1 = UPPER, 2 = MID, 3 = LOWER)
	# Name and Sex are strings, find a way to treat them as numbers.
		# Names can be parsed to get just the persons's title (mr, miss, dr, mr), this may be useful
		# Sex should probably be 0/1 for male/female	
	# Some ages are missing, how to treat them? Try to guess the value based on statistics or just 0 it for simplicity (if its not going to affect result too much)
	# SibSp is the number of siblings onboard
	# Parch is the number of parents/childs onboard
	# Ticket I believe there is not enough data to train a relation with this feature, even if extracting punctuations and letters, only 300 values are discarded and the dataset isnt large
	# Fare
	# Cabin is too sparse, first tests will just ignore it
		# NOTE: Maybe its possible to get cabin's deck for something
	# Embarked is just the port where they embarked from, probably not usefull as well

	# Feature engineering references:
	# 1 - http://www.ultravioletanalytics.com/2014/11/07/kaggle-titanic-competition-part-iv-derived-variables/
	# 2 - https://triangleinequality.wordpress.com/2013/09/08/basic-feature-engineering-with-the-titanic-data/
	# 3 - https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic/code/code

	dataset = read.csv(dataset.path, header = TRUE)

	dataset$Sex = ifelse(dataset$Sex == "male", 0, dataset$Sex)

	# Set all NA ages to median
	dataset$Age[is.na(dataset$Age)] = 0 # Defaults all NA to 0
	dataset$Age[dataset$Age == 0] = median(dataset$Age)
	dataset$Age = dataset$Age/100 # Normalize Age - rarely will be above 100

	# Add family size feature
	dataset$FamSize = dataset$SibSp + dataset$Parch + 1

	# Discretize family size since if someone is alone, there is a higher chance of death
	# Alone = -1
	# Large = 1
	# [2, 4] = 0
	dataset$CategoricFamSize[dataset$FamSize == 1] =  -1
	dataset$CategoricFamSize[dataset$FamSize < 5 & dataset$FamSize > 1] = 0
	dataset$CategoricFamSize[dataset$FamSize > 4] = 1
	
	# Add Fare per person
	# But first find where there is no fare and set it to median
	dataset$Fare[is.na(dataset$Fare)] = 0 # Defaults all NA to 0
	dataset$Fare[dataset$Fare == 0] = median(dataset$Fare)
	# Normalize fare price - prices are commonly log-normalized in economics
	dataset$Fare = log(dataset$Fare) # e^NormalizedFare returns original Fare.

	dataset$FarePerPerson = dataset$Fare/dataset$FamSize

	# Link 3 already provided de two values missing from "Embarked" so I will
	# just be using those values :)
	# The idea is to guess from where the passenger embarked based on median 
	# fare from first and third class passengers
	if(train == TRUE){
		dataset$Embarked[62] = 'C'
		dataset$Embarked[830] = 'S'
	}
	
	# Map Embarked to discrete numbers
	tmp = as.vector(dataset$Embarked)
	tmp[tmp == "C"] = 1
	tmp[tmp == "Q"] = 2
	tmp[tmp == "S"] = 3
	dataset$Embarked = as.numeric(tmp)

	# Get title from passenger names
	dataset$Title = gsub("(.*, )|(\\..*)", "", dataset$Name)

	# Dona, Lady and The Countess titles only have one appearance, wont have any
	# influence on its own
	dataset$Title[
	 dataset$Title == "Miss" | 
	 dataset$Title == "Mlle" | 
	 dataset$Title == "Ms" | 
	 dataset$Title == "Mrs" | 
	 dataset$Title == "Dona" |	
	 dataset$Title == "Lady" |
	 dataset$Title == "the Countess" |
	 dataset$Title == "Mme"] = 1 # "Miss"

	dataset$Title[dataset$Title == "Mr"] = 2 # "Mister"

	dataset$Title[
	 dataset$Title == "Dr" |
	 dataset$Title == "Master"] = 3
	
	# Special titles (only one or two ocurrences for them, except for Dr and Rev)
	dataset$Title[
	 dataset$Title == "Rev" |
	 dataset$Title == "Capt" |
	 dataset$Title == "Col" |
	 dataset$Title == "Don" |
	 dataset$Title == "Major" |
	 dataset$Title == "Sir" |
	 dataset$Title == "Jonkheer"] = 4 # "special"

	dataset$Title = as.numeric(dataset$Title)
	
	# Discard Ticket
	dataset$Ticket = NULL
	# Discard Cabin NOTE: Maybe try to use the cabin's letter and set the rest do unknown
	dataset$Cabin = NULL
	# Discard Name
	dataset$Name = NULL

	dataset$output = dataset$Survived
	dataset$Survived = NULL

	return (dataset)
}

titanic <- function(dataset.path = "dataset/train.csv", alpha=10, 
	step=0.004, threshold=1e-1){

	dataset = as.matrix(titanic.prepare.data(dataset.path))
	# TODO: Generate interation correlated variables maybe

	size = mlp.upper.hidden.size(input.size=ncol(dataset)-1, output.size=1, 
		n.samples=nrow(dataset), alpha=alpha)
	mlp = mlp.create(input.size=ncol(dataset)-2, output.size=1, hidden.size=size)
	mlp = mlp.train(mlp, dataset[,2:(ncol(dataset)-1)], dataset[,ncol(dataset)], 
		step=step, threshold=threshold)

	return (mlp)
}

titanic.test <- function(mlp, test.path = "dataset/test.csv", validation.path){

	testset = as.matrix(titanic.prepare.data(test.path, train=FALSE))
	test.validate = as.matrix(read.csv(file=validation.path, header=TRUE))

	test.size = nrow(testset)
	testset.use = testset[, 2:ncol(testset)]

	ret = list()
	ret$results = rep(0, test.size)

	for(i in 1:test.size){
		tmp = mlp.forward(mlp, testset.use[i,])
		ret$results[i] = tmp$f.output # We just want the output
	}

	ret$binary.results = titanic.discretize.results(results)
	ret$accuracy = mlp.accuracy(ret$binary.results, test.validate[,2])
	cat("Accuracy: ", ret$accuracy, "\n")

	return (ret)
}

# TODO: generalize this maybe
titanic.discretize.results <- function(results){ 
	ret = rep(0, length(results))
	ret[results >= 0.5] = TRUE
	return (ret)
}


mlp = titanic(alpha=15, step=0.008, threshold=0.1)
cat("Finished running, saving mlp dump to \"mlp-dump/mlp.dat\"...")
save(mlp, file="mlp-dump/mlp.dat")
