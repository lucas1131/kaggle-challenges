################################################################################
#                                                                              #
# Source file for Multilayer Percetron                                         #
#                                                                              #
# Lucas Alexandre Soares 14/07/2017                                            #
# lucassoares1793@gmail.com                                                    #
# NUsp: 9293265                                                                #
################################################################################

# require(gmatrix)
# require(Rcpp)
# source("src/cpp-routines.rcpp")

# Activation function
sigmoid <- function(net){ return (1/(1+exp(-net))) }

# Activation function derivative
d_sigmoid <- function(net) { 
	tmp = sigmoid(net)
	return (tmp * (1-tmp))
}

mlp.accuracy <- function(mlp.results, test.validation){	
	return (sum(mlp.results == test.validation)/length(test.validation))
}

# Estimates the upper bound for the hidden layer size that wont overfit the mlp
mlp.upper.hidden.size <- function(input.size, output.size, n.samples, alpha=2){
	return (floor( n.samples/(alpha*(input.size + output.size)) ))
}

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
	mlp$layers$hidden = matrix(runif(min=-0.5, max=0.5, n=hidden.size*(input.size+1)),
					nrow=hidden.size, ncol=input.size+1)

	mlp$layers$output = matrix(runif(min=-0.5, max=0.5, n=output.size*(hidden.size+1)),
					nrow=output.size, ncol=hidden.size+1)

	return (mlp)
}

# Feed input forwad and get output
mlp.forward <- function(mlp, input){
	
	fwd = list()
	
	fwd$f.hidden = rep.int(0, mlp$size$hidden)
	fwd$df.hidden = rep.int(0, mlp$size$hidden)
	fwd$f.output = rep(0, mlp$size$output)
	fwd$df.output = rep(0, mlp$size$output)

	net.hidden = rep(0, mlp$size$hidden)
	net.output = rep(0, mlp$size$output)

	# Append a 1 input to the end for the bias parameter
	tmp = as.vector(c(input, 1))

	# Forward Input -> Hidden layer
	for(i in 1:mlp$size$hidden)
		net.hidden[i] = tmp %*% mlp$layers$hidden[i,]
	
	fwd$f.hidden = mlp$f(net.hidden)	# Activate hidden layer perceptrons
	fwd$df.hidden = mlp$df(net.hidden)	# Activation derivative

	# Append a 1 input to the end for the bias parameter
	tmp = c(fwd$f.hidden, 1)

	# Forward Hidden -> Output layer
	for (i in 1:mlp$size$output)
		net.output[i] = tmp %*% mlp$layers$output[i,]

	fwd$f.output = mlp$f(net.output) # Activate output layer perceptrons
	fwd$df.output = mlp$df(net.output) # Activation derivative

	return (fwd)
}

# Train with backpropagation of errors
mlp.train <- function(mlp, train.input, train.output, step=0.1, threshold=1e-2){

	old.error = 100
	error = threshold+1 # Just to enter the loop
	train.input.size = nrow(train.input)

	# Keep training until error is below acceptable threshold
	while(error > threshold){

		error = 0
		t = proc.time()
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
				step*(tcrossprod(delta.output, c(as.vector(fwd$f.hidden), 1)))

			# Calculate hidden layer delta and update its weights
			delta.hidden = fwd$df.hidden * 
				(delta.output %*% mlp$layers$output[,1:mlp$size$hidden])

			mlp$layers$hidden = mlp$layers$hidden +
				step*(tcrossprod(as.vector(delta.hidden), c(train.input[i,], 1)))
		}

		# Normalize the error
		error = error/train.input.size

		# Try to make an adaptative step so mlp wont get stuck
		if(error > old.error){
			step = step - (step/10) # Keep subtracting by 10% of the actual step
			cat("Error got bigger, reducing step by 10%\nNew step:", step, "\n")
		}

		old.error = error

		cat("Time: ", proc.time() - t, "\n")
	    cat("Average squared error: ", error, "\n") # Faster than print
	}

	return (mlp)
}

# debug(mlp.forward)
# debug(mlp.train)