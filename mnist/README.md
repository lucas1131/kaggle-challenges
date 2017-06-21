MNIST dataset from Kaggle challenges, running with MLP (expandable in the future)

# Functions
	`mnist(dataset.path, alpha, step, threshold)`
	`mnist.pca(dataset.path, alpha, step, threshold)`

	Run the dataset with a model and then return the trained model.
	(PCA version not working yet :[ )

	`mnist.prepare.data(dataset.path = "dataset/train.csv")`
	
	Prepare dataset by doing all features engineering, imputation, normalization. Return the dataset ready to feed the model.
	
	`mnist.prepare.output(raw.output)`

	Expands the output from single value with the corresponding number to a vector of 1's and 0's to feed the model.

	`mnist.test(mlp, test.path, output.path)`
	`mnist.pca.test(mlp, test.path, output.path)`

	Run the test set with a the model and create a .csv file with the results.
	(PCA version not working yet :[ )
