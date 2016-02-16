from layers import convolutional, activation, maxpool, linear



def get_model(X, batch_size, image_dimension):

	input_shape = (batch_size, 3, image_dimension, image_dimension)
	all_parameters = []

	#############################################
	# a first convolution with 32 (3, 3) filters
	output, params, output_shape = convolutional(X, input_shape, 32, (3, 3))
	all_parameters += params

	# maxpool with size=(2, 2)
	output, params, output_shape = maxpool(output, output_shape, (2, 2))

	# relu activation
	output, params, output_shape = activation(output, output_shape, 'relu')

	#############################################
	# a second convolution with 32 (3, 3) filters
	output, params, output_shape = convolutional(output, output_shape, 32, (3, 3))
	all_parameters += params

	# maxpool with size=(2, 2)
	output, params, output_shape = maxpool(output, output_shape, (2, 2))

	# relu activation
	output, params, output_shape = activation(output, output_shape, 'relu')
	
	#############################################
	# a third convolution with 32 (3, 3) filters
	output, params, output_shape = convolutional(output, output_shape, 32, (3, 3))
	all_parameters += params

	# maxpool with size=(2, 2)
	output, params, output_shape = maxpool(output, output_shape, (2, 2))

	# relu activation
	output, params, output_shape = activation(output, output_shape, 'relu')

	#############################################
	# MLP first layer

	output = output.flatten(2)
	
	output, params, output_shape = linear(output, (output_shape[0], output_shape[1]*output_shape[2]*output_shape[3]), 500)
	all_parameters += params

	output, params, output_shape = activation(output, output_shape, 'relu')

	#############################################
	# MLP second layer

	output, params, output_shape = linear(output, output_shape, 1)
	all_parameters += params

	output, params, output_shape = activation(output, output_shape, 'sigmoid')

	#
	return output, all_parameters
