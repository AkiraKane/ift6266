from fuel.streams import ServerDataStream

import theano
import sys
from theano import tensor
from theano import config
import numpy
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from theano.tensor.nnet import relu
import datetime

from layers import convolutional, activation
import socket

from blocks.algorithms import GradientDescent, Adam, Scale, Momentum
from blocks.model import Model
from blocks.extensions import Printing, Timing
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
from blocks.graph import ComputationGraph, apply_batch_normalization, get_batch_normalization_updates
from blocks.main_loop import MainLoop
from blocks_extras.extensions.plot import Plot
from blocks.extensions.saveload import Checkpoint

def run(model_name, port_train, port_valid):

	running_on_laptop = socket.gethostname() == 'yop'

	X = tensor.tensor4('image_features', dtype='float32')
	T = tensor.matrix('targets', dtype='float32')

	image_border_size = (100, 100)

	if running_on_laptop:
		host_plot = 'http://localhost:5006'
		batch_size = 10
	else:
		host_plot = 'http://localhost:5006'
		# host_plot = 'http://hades.calculquebec.ca:5042'
		batch_size = 256

	prediction, prediction_test = get_model(X, batch_size, image_border_size)

	alpha = 0.1

	cg = ComputationGraph([prediction])
	cg = apply_batch_normalization(cg)
	pop_updates = get_batch_normalization_updates(cg)
	extra_updates = [(p, m * alpha + p * (1 - alpha)) for p, m in pop_updates]

	prediction = cg.outputs[0]
	model = Model(prediction_test)
    
    # trick !
	# T2 = T * 0.8 + 0.1
	## loss and validation error
	loss = tensor.nnet.binary_crossentropy(prediction, T).mean()
	loss_test = tensor.nnet.binary_crossentropy(prediction_test, T).mean()
	error = tensor.gt(tensor.abs_(prediction_test - T), 0.5).mean(dtype='float32')
	error.name = 'error'
	loss.name = 'loss'
	loss_test.name = 'loss_test'

	algorithm = GradientDescent(cost=loss, parameters=cg.parameters,
	                            # step_rule=Adam(0.01))
	                            # step_rule=Scale(0.01))
	                            step_rule=Momentum(learning_rate=0.01, momentum=0.9))

	algorithm.add_updates(extra_updates)
	

	train_stream = ServerDataStream(('image_features','targets'), False, port=port_train)
	valid_stream = ServerDataStream(('image_features','targets'), False, port=port_valid)

	extensions = [
		Timing(),
		TrainingDataMonitoring([loss, error], after_epoch=True),
		DataStreamMonitoring(variables=[error], data_stream=valid_stream, prefix="valid"),
		Plot('%s %s @ %s' % (model_name, datetime.datetime.now(), socket.gethostname()), channels=[['loss'], ['error', 'valid_error']], after_epoch=True, server_url=host_plot),
		Printing(),
		Checkpoint('/tmp/train_bn2')
	]

	main_loop = MainLoop(data_stream=train_stream, algorithm=algorithm,
	                     extensions=extensions, model=model)
	main_loop.run()


if __name__ == "__main__":

	if len(sys.argv) < 2:
		print('Usage: python train.py path_to_model.py [port_train] [port_valid]')
		exit()

	# prepare path for import
	path = sys.argv[1]
	if path[-3:] == '.py':
		path = path[:-3]
	path = path.replace('/','.')

	if len(sys.argv) > 3:
		port_train = int(sys.argv[2])
		port_valid = int(sys.argv[3])
	else:
		port_train = 5557
		port_valid = 5558
	
	# import right model
	get_model = __import__(path, globals(), locals(), ['get_model']).get_model

	# run the training
	run(path.split('.')[-1], port_train, port_valid)
