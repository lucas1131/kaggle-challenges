################################################################################
#                                                                              #
# Source file for Digit Recognition: MNIST dataset                             #
# This file contains every routine to manipulate, train and test the MNIST     #
# dataset.                                                                     #
#                                                                              #
# Lucas Alexandre Soares 16/07/2017                                            #
# lucassoares1793@gmail.com                                                    #
# NUsp: 9293265                                                                #
################################################################################

require(nnet)
source("src/mlp.r")

mnist.pca <- function(dataset.path="dataset/train.csv", alpha=10, 
	step=0.004, threshold=1e-1){

	cat("Reading and preparing dataset...\n")
	dataset = as.matrix(mnist.prepare.data(dataset.path))
	output = mnist.prepare.output(dataset[,1])

	cat("Done!\nApplying PCA...\n")
	data.pca = summary(prcomp(dataset[,2:ncol(dataset)]))
	cat("Done!\n")
	size = length(dataset)
	
	trunc = tcrossprod(data.pca$x, data.pca$rotation)

	cat("Finding relevant PC's...\n")
	for(i in 1:size){
		# Find i where the cumulative importance is >= 95%
		if(data.pca$importance[3*i] >= 0.95){
			break
		}
	}

	mlp$pcs = i

	# Truncate to only the PC's whose cumulative importance is >= 95%
	trunc = as.matrix(data.pca$x[,1:i])

	cat("Done!\nEstimating hidden layer size...\n")
	nfeatures = i
	samples = nrow(trunc)
	output.size = ncol(output)

	hsize = mlp.upper.hidden.size(nfeatures, output.size, samples, alpha)

	cat("Done!\nCreating mlp...\n")
	mlp = mlp.create(nfeatures, output.size, hsize)
	cat("Done!\nStart training!\n")
	mlp = mlp.train(mlp, trunc, output, step=step, threshold=threshold)

	return (mlp)
}

mnist.prepare.data <- function(dataset.path = "dataset/train.csv"){

	# dataset = read.csv(dataset.path, header = TRUE)
	dataset = read.csv(dataset.path, header = TRUE)

	# Normalize data
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

mnist <- function(dataset.path="dataset/train.csv", alpha=2, 
	step=0.004, threshold=1e-1){

	dataset.path = "dataset/train.csv"
	dataset = as.matrix(mnist.prepare.data(dataset.path))
	output = mnist.prepare.output(dataset[,1])

	nfeatures = ncol(dataset)-1
	samples = nrow(dataset)
	output.size = ncol(output)

	hsize = mlp.upper.hidden.size(nfeatures, output.size, samples, alpha)

	mlp = mlp.create(nfeatures, output.size, hsize)
	mlp = mlp.train(mlp, dataset[,2:ncol(dataset)], output, step=step, threshold=threshold)

	return (mlp)
}

mnist.test <- function(mlp, test.path="dataset/test.csv", 
						output.path="results/result1.csv"){

	testset = as.matrix(mnist.prepare.data(test.path))
	test.size = nrow(testset)

	Label = rep(-1, test.size)

	for(i in 1:test.size){
		# Transform results index back to labels
		Label[i] = which.max(mlp.forward(mlp, testset[i,])$f.output)-1
	}
	# ImageId = 
	write.csv(cbind(seq(1:length(Label)), Label), file=output.path, 
		row.names=FALSE, col.names=c("ImageId", "Label"))
}


mnist.pca.test <- function(mlp, test.path="dataset/test.csv", 
						output.path="results/pca-result.csv"){

	testset = as.matrix(mnist.prepare.data(test.path))

	test.pca = summary(prcomp(testset))
	
	trunc = tcrossprod(test.pca$x, test.pca$rotation)
	# trunc = as.matrix(test.pca$x[,1:mlp$pcs])
	trunc = as.matrix(test.pca$x[,1:154])

	test.size = nrow(testset)

	for(i in 1:test.size){
		
		# Test
		fwd = mlp.forward(mlp, trunc[i,])
		
		# Transform results index back to labels
		Label[i] = which.max(fwd$f.output)-1
	}
	# ImageId = 
	write.csv(cbind(seq(1:length(Label)), Label), file=output.path, 
		row.names=FALSE, col.names=c("ImageId", "Label"))
}
