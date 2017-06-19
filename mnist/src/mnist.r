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

require(nnet)
source("src/mlp.r")

mnist.pca <- function(dataset){

	data.pca = summary(prcomp(dataset))
	size = length(dataset)
	
	trunc = tcrossprod(data.pca$x, data.pca$rotation)

	for(i in 1:size){
		# Find i where the cumulative importance is >= 95%
		if(data.pca$importance[3*i] >= 0.95){
			break
		}
	}
	
	# Truncate to only the PC's whose cumulative importance is >= 95%
	trunc = data.pca$x[,1:i]

	return (trunc)
}

mnist.prepare.data <- function(dataset.path = "dataset/train.csv"){

	# dataset = read.csv(dataset.path, header = TRUE)
	dataset = read.csv("dataset/train.csv", header = TRUE)

	dataset[,2:ncol(dataset)] = dataset[,2:ncol(dataset)]/255
	
	# dataset.pca = mnist.pca(dataset)

	labels = dataset$label
	# dataset$label = NULL	# remove label from dataset for PCA

	return (dataset)
}	

mnist.prepare.output <- function(raw.output){

	output = matrix(0, nrow=length(raw.output), ncol=10)

	for(i in 1:length(raw.output))
		output[i, raw.output[i]+1] = 1

	return (output)
}

mnist <- function(dataset.path = "dataset/train.csv", alpha=10, 
	step=0.004, threshold=1e-1){

	dataset = as.matrix(mnist.prepare.data(dataset.path))
	output = mnist.prepare.output(dataset[,1])

	nfeatures = ncol(dataset)-1
	samples = nrow(dataset)
	output.size = ncol(output)

	hsize = mlp.upper.hidden.size(nfeatures, output.size, samples, 2)

	mlp = mlp.create(nfeatures, output.size, hsize)
	mlp = mlp.train(mlp, dataset[,2:ncol(dataset)], output, step=step, threshold=threshold)

	return (mlp)
}

mnist.test <- function(mlp, test.path = "dataset/test.csv", validation.path){

	testset = as.matrix(mnist.prepare.data(test.path, train=FALSE))
	test.validate = as.matrix(read.csv(file=validation.path, header=TRUE))

	test.size = nrow(testset)
	testset.use = testset[, 2:ncol(testset)]

	ret = list()
	ret$results = rep(0, test.size)

	for(i in 1:test.size){
		tmp = mlp.forward(mlp, testset.use[i,])
		ret$results[i] = tmp$f.output # We just want the output
	}

	ret$binary.results = mnist.discretize.results(results)
	ret$accuracy = mlp.accuracy(ret$binary.results, test.validate[,2])
	cat("Accuracy: ", ret$accuracy, "\n")

	return (ret)
}

mnist.discretize.results <- function(results){ 
	ret = rep(0, length(results))
	ret[results >= 0.5] = TRUE
	return (ret)
}


# mlp = mnist(alpha=15, step=0.008, threshold=0.1)
# cat("Finished running, saving mlp dump to \"mlp-dump/mlp.dat\"...")
# save(mlp, file="mlp-dump/mlp.dat")
