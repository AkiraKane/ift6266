from layers import convolutional, activation, maxpool, linear



def get_model(X, batch_size, image_dimension):

	input_shape = (batch_size, 3, image_dimension, image_dimension)
	all_parameters = []

	#############################################
	# a first block with 2 convolutions of 32 (3, 3) filters
	output, output_test, params, output_shape = convolutional(X, X, input_shape, 32, (3, 3))
	all_parameters += params
	output, output_test, params, output_shape = activation(output, output_test, output_shape, 'relu')
	output, output_test, params, output_shape = convolutional(output, output_test, output_shape, 32, (3, 3))
	all_parameters += params
	output, output_test, params, output_shape = activation(output, output_test, output_shape, 'relu')
	
	# maxpool with size=(2, 2)
	output, output_test, params, output_shape = maxpool(output, output_test, output_shape, (2, 2))

	#############################################
	# a 2nd block with 3 convolutions of 64 (3, 3) filters
	output, output_test, params, output_shape = convolutional(output, output_test, output_shape, 64, (3, 3))
	all_parameters += params
	output, output_test, params, output_shape = activation(output, output_test, output_shape, 'relu')
	output, output_test, params, output_shape = convolutional(output, output_test, output_shape, 64, (3, 3))
	all_parameters += params
	output, output_test, params, output_shape = activation(output, output_test, output_shape, 'relu')
	output, output_test, params, output_shape = convolutional(output, output_test, output_shape, 64, (3, 3))
	all_parameters += params
	output, output_test, params, output_shape = activation(output, output_test, output_shape, 'relu')
	
	# maxpool with size=(2, 2)
	output, output_test, params, output_shape = maxpool(output, output_test, output_shape, (2, 2))

	#############################################
	# a 3rd block with 4 convolutions of 128 (3, 3) filters
	output, output_test, params, output_shape = convolutional(output, output_test, output_shape, 128, (3, 3))
	all_parameters += params
	output, output_test, params, output_shape = activation(output, output_test, output_shape, 'relu')
	output, output_test, params, output_shape = convolutional(output, output_test, output_shape, 128, (3, 3))
	all_parameters += params
	output, output_test, params, output_shape = activation(output, output_test, output_shape, 'relu')
	output, output_test, params, output_shape = convolutional(output, output_test, output_shape, 128, (3, 3))
	all_parameters += params
	output, output_test, params, output_shape = activation(output, output_test, output_shape, 'relu')
	output, output_test, params, output_shape = convolutional(output, output_test, output_shape, 128, (3, 3))
	all_parameters += params
	output, output_test, params, output_shape = activation(output, output_test, output_shape, 'relu')
	
	# maxpool with size=(2, 2)
	output, output_test, params, output_shape = maxpool(output, output_test, output_shape, (2, 2))

	#############################################
	# MLP first layer

	output = output.flatten(2)
	output_test = output_test.flatten(2)
	
	output, output_test, params, output_shape = linear(output, output_test, (output_shape[0], output_shape[1]*output_shape[2]*output_shape[3]), 500)
	all_parameters += params

	output, output_test, params, output_shape = activation(output, output_test, output_shape, 'relu')

	#############################################
	# MLP second layer

	output, output_test, params, output_shape = linear(output, output_test, output_shape, 1)
	all_parameters += params

	output, output_test, params, output_shape = activation(output, output_test, output_shape, 'sigmoid')

	#
	return output, output_test, all_parameters
