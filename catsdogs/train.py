from fuel.streams import ServerDataStream
# Create the Theano MLP
import theano
from theano import tensor
from theano import config
import numpy
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from theano.tensor.nnet import relu
import datetime

from layers import convolutional, activation
import socket

X = tensor.tensor4('image_features', dtype='float32')
T = tensor.matrix('targets', dtype='float32')

batch_size = 100
image_border_size = 100

## getting model

# from models.simple_convolutional import get_model
from models.conv_3_layers import get_model
prediction, all_parameters = get_model(X, batch_size, image_border_size)

## loss and validation error
loss = tensor.nnet.binary_crossentropy(prediction, T).mean()
error = tensor.gt(tensor.abs_(prediction - T), 0.5).mean(dtype='float32')
error.name = 'error'

if socket.gethostname() == 'yop':
	host_plot = 'http://localhost:5006'
else:
	host_plot = 'http://hades.calculquebec.ca:5042'

# Use Blocks to train this network
from blocks.algorithms import GradientDescent, Adam
from blocks.extensions import Printing, ProgressBar
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
from blocks.main_loop import MainLoop
from blocks_extras.extensions.plot import Plot
from blocks.extensions.saveload import Checkpoint

algorithm = GradientDescent(cost=loss, parameters=all_parameters,
                            step_rule=Adam())


# We want to monitor the cost as we train
loss.name = 'loss'

train_stream = ServerDataStream(('image_features','targets'), False)
valid_stream = ServerDataStream(('image_features','targets'), False, port=5558)

extensions = [
	TrainingDataMonitoring([loss], after_epoch=True),
	DataStreamMonitoring(variables=[loss, error], data_stream=valid_stream, prefix="valid"),
	Plot('Training %s @ %s' % (datetime.datetime.now(), socket.gethostname()), channels=[['loss', 'valid_loss'], ['valid_error']], after_epoch=True, server_url=host_plot),
	Printing(),
	Checkpoint('train2')
]

main_loop = MainLoop(data_stream=train_stream, algorithm=algorithm,
                     extensions=extensions)
main_loop.run()
