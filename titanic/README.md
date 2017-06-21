Titanic dataset from Kaggle challenges, running with MLP (expandable in the future)

Functions

```titanic(dataset.path, alpha, step, threshold)```

Run the dataset with a model and then return the trained model.

```titanic.prepare.data(dataset.path)```

Prepare dataset by doing all features engineering, imputation, normalization. Return the dataset ready to feed the model.

```titanic.discretize.results(results)```

Discretizes results to be boolean values by applying hard threshold. True if value >= 0.5, false otherwise.

```titanic.test(mlp, test.path, output.path)```

Run the test set with a the model and create a .csv file with the results.
