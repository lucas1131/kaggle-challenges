require(nnet)
# require(tseriesChaos)

# TODO
# fhard <- function(net);

# DONE
# mlp.architecture <- function(input.length = 2, hidden.length = 2, 
# 	output.length = 1, my.f = f, my.df_dnet = df_dnet);


# TODO
# time.series.prediction <- function(series, embedding.dimension, time.delay,
# 	hidden.length = 4, train.set.size = 0.9, eta = 0.05, threshold = 1e-3);

# TODO
# lorenz.recursive <- function(embedding.dimension = 3, time.delay = 5, 
# 	hidden.length = 5, train.set.size = 0.9, eta = 0.05, threshold = 0.017, 
# 	plot.points = 50);

# TODO
# logistic.recursive <- function(embedding.dimension = 2, time.delay = 1, 
# 	hidden.length = 5, train.set.size = 0.9, eta = 0.05, threshold = 1e-3, 
# 	plot.points = 50);

# Activation function
sigmoid <- function(net){ return (1/(1+exp(-net))) }

# Activation function derivative
d_sigmoid <- function(net) { return (sigmoid(net) * (1-sigmoid(net))) }

mlp.upper.hidden.size <- function(input.size, output.size, n.samples, alpha=2){
	return (floor(n.samples/alpha*(input.size + output.size)))
}

# mlp.create <- function(input.size, output.size, 
# 					hidden.size = ceiling((input.size+output.size)/2),
# 					activation.func = sigmoid, activation.df = d_sigmoid){

# 	mlp = list()
# 	mlp$layers = list()
# 	mlp$f = activation.func
# 	mlp$df = activation.df

# 	# Create input-hidden and hidden-output layers as matrix with random weights
# 	# Input size = 2   Hidden size = 3   Output size = 2
	
#     #       In1   In2   In3 (bias)             Hn1   Hn2   Hn3 (bias)
#     #     |-----|-----|-----|                 |-----|-----|-----|
#     # Hn1 |     |     |     |            Out1 |     |     |     |
#     #     |-----|-----|-----|                 |-----|-----|-----|
#     # Hn2 |     |     |     |            Out2 |     |     |     |
#     #     |-----|-----|-----|                 |-----|-----|-----|
#     # Hn3 |     |     |     |
#     #     |-----|-----|-----|

#     # Add 1 ncol so we can move the hyperplane's b (bias) parameter (a*x + b)
# 	mlp$layers$hidden = matrix(runif(min=-1, max=1, n=hidden.size*(input.size+1)),
# 					nrow=hidden.size, ncol=input.size+1)

# 	mlp$layers$output = matrix(runif(min=-1, max=1, n=output.size*(hidden.size+1)),
# 					nrow=output.size, ncol=hidden.size+1)

# 	return (mlp)
# }

mlp.create <- function(input.size, output.size, 
					hidden.size = ceiling((input.size+output.size)/2),
					activation.func = sigmoid, activation.df = d_sigmoid){

	mlp = list()
	mlp$layers = list()
	mlp$size = list()
	mlp$f = activation.func
	mlp$df = activation.df

	# Avoid recalculating layers sizes every time
	mlp$size$input = input.size
	mlp$size$hidden = hidden.size
	mlp$size$output = output.size
	
	# Create input-hidden and hidden-output layers as matrix with random weights
	# Input size = 2   Hidden size = 3   Output size = 2
	
    #       In1   In2   In3 (bias)             Hn1   Hn2   Hn3   Hn4 (bias)
    #     |-----|-----|-----|                 |-----|-----|-----|-----|
    # Hn1 |     |     |     |            Out1 |     |     |     |     |
    #     |-----|-----|-----|                 |-----|-----|-----|-----|
    # Hn2 |     |     |     |            Out2 |     |     |     |     |
    #     |-----|-----|-----|                 |-----|-----|-----|-----|
    # Hn3 |     |     |     |            
    #     |-----|-----|-----|            

	# Add 1 to ncol so we can move the hyper plane's b (bias) parameter (a*x + b)
	mlp$layers$hidden = matrix(runif(min=-1, max=1, n=hidden.size*(input.size+1)),
					nrow=hidden.size, ncol=input.size+1)

	mlp$layers$output = matrix(runif(min=-1, max=1, n=output.size*(hidden.size+1)),
					nrow=output.size, ncol=hidden.size+1)

	return (mlp)
}

mlp.forward <- function(mlp, input){
	
	input = as.matrix(input)

	fwd = list()
	
	# R is an interpreted language, there is no real need to create an entire 
	# vector just to initialize this, there is?
	# But this may be faster if R can allocate everything at once, better than
	# reallocating every time a new value is added. (I believe this is faster)

	# See Section 4 in http://www.noamross.net/blog/2014/4/16/vectorization-in-r--why.html
	# Its definely faster to create the vector first and them iterate though it

	fwd$f.hidden = rep.int(0, mlp$size$hidden)
	fwd$df.hidden = rep.int(0, mlp$size$hidden)
	fwd$f.output = rep(0, mlp$size$output)
	fwd$df.output = rep(0, mlp$size$output)

	# Forward Input -> Hidden layer
	for(i in 1:mlp$size$hidden){
		
		# Append a 1 input to the end for the bias parameter
		net.hidden = c(input, 1) %*% mlp$layers$hidden[i,]
		
		fwd$f.hidden[i] = mlp$f(net.hidden)	# Activate hidden layer perceptrons
		fwd$df.hidden[i] = mlp$df(net.hidden) # Activation derivative
	}

	# Forward Hidden -> Output layer
	for (i in 1:mlp$size$output){
		
		# Append a 1 input to the end for the bias parameter
		net.output = c(fwd$f.hidden, 1) %*% mlp$layers$output[i,]
		
		fwd$f.output[i] = mlp$f(net.output) # Activate output layer perceptrons
		fwd$df.output[i] = mlp$df(net.output) # Activation derivative
	}

	return (fwd)
}

# Train with backpropagation of errors
mlp.train <- function(mlp, train.input, train.output, step=0.1, threshold=1e-2){

	error = threshold+1; # Just to enter the loop
	train.input.size = nrow(train.input)

	train.input = as.matrix(train.input)
	train.output = as.matrix(train.output)

	# Keep training until error is below acceptable threshold
	while(error > threshold){

		error = 0

		for(i in 1:train.input.size){

			# Feed input forward
			fwd = mlp.forward(mlp, train.input[i,])

			# Calculate delta from expected and achieved outputs
			delta = train.output[i,] - fwd$f.output
			error = error + sum(delta^2) # Squared error

			# Feed result backwards (backpropagation)
			# Calculate output layer delta and update its weights
			delta.output = as.vector(delta * fwd$df.output)

			mlp$layers$output = mlp$layers$output + 
				step*(delta.output %*% t(c(as.vector(fwd$f.hidden), 1)))

			# Calculate hidden layer delta and update its weights
			delta.hidden = fwd$df.hidden * 
				(delta.output %*% mlp$layers$output[,1:mlp$size$hidden])

			mlp$layers$hidden = mlp$layers$hidden +
				step*(as.vector(delta.hidden) %*% t(c(train.input[i,], 1)))
		}

		# Normalize the error
		error = error/train.input.size
	    cat("Average squared error: ", error, "\n") # Faster than print
	}

	return (mlp)
}

mlp.titanic.prepare.data <- function(dataset.path = "dataset/train.csv", train = TRUE){

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
	
	# Discard PassengerId because this is just and id 
	# dataset$PassengerId = NULL
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

mlp.titanic <- function(dataset.path = "dataset/train.csv"){

	dataset.path = "dataset/train.csv"
	dataset = mlp.titanic.prepare.data(dataset.path)
	# TODO: Generate interation correlated variables maybe

	size = mlp.upper.hidden.size(input.size=ncol(dataset)-1, output.size=1, 
		n.samples=nrow(dataset), alpha=5)
	mlp = mlp.create(input.size=ncol(dataset)-1, output.size=1, hidden.size=size)
	mlp = mlp.train(mlp, dataset[2:ncol(dataset)-1], dataset$output, step=0.01)

	return (mlp)
}
