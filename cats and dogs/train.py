from fuel.streams import ServerDataStream
# Create the Theano MLP
import theano
from theano import tensor
from theano import config
import numpy
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from theano.tensor.nnet import relu

X = tensor.tensor4('image_features', dtype='float32')
T = tensor.matrix('targets', dtype='float32')

batch_size = 100
image_border_size = 100

conv1_size = 5
filter1_shape = (32, 3, conv1_size, conv1_size)
poolsize1 = (2, 2)
input1_shape = (batch_size, 3, image_border_size, image_border_size)
filters1 = theano.shared(
    numpy.random.uniform(low=-0.01, high=0.01, size=filter1_shape).astype(numpy.float32), 'filter1')
bias1 = theano.shared(numpy.zeros((filter1_shape[0], )).astype(numpy.float32))
conv1_out_size = (image_border_size - conv1_size + 1)/poolsize1[0]

conv2_size = 5
filter2_shape = (32, filter1_shape[0], conv2_size, conv2_size)
poolsize2 = (2, 2)
input2_shape = (batch_size, filter1_shape[0], conv1_out_size, conv1_out_size)
filters2 = theano.shared(
    numpy.random.uniform(low=-0.01, high=0.01, size=filter2_shape).astype(numpy.float32), 'filter2')
bias2 = theano.shared(numpy.zeros((filter2_shape[0], )).astype(numpy.float32))
conv2_out_size = (conv1_out_size - conv2_size + 1)/poolsize2[0]

W1 = theano.shared(
	numpy.random.uniform(low=-0.01, high=0.01, size=(conv2_out_size**2*filter2_shape[0], 500)).astype(numpy.float32), 'W1')
b1 = theano.shared(numpy.zeros(500).astype(numpy.float32))
W2 = theano.shared(
	numpy.random.uniform(low=-0.01, high=0.01, size=(500, 1)).astype(numpy.float32), 'W2')
b2 = theano.shared(numpy.zeros(1).astype(numpy.float32))

params = [filters1, bias1, filters2, bias2, W1, b1, W2, b2]

conv1 = conv2d(input=X, filters=filters1, filter_shape=filter1_shape, image_shape=input1_shape)
pooled1 = max_pool_2d(input=conv1, ds=poolsize1, ignore_border=True)
out1 = relu(pooled1 + bias1.dimshuffle('x', 0, 'x', 'x'))

conv2 = conv2d(input=out1, filters=filters2, filter_shape=filter2_shape, image_shape=input2_shape)
pooled2 = max_pool_2d(input=conv2, ds=poolsize2, ignore_border=True)
out2 = relu(pooled2 + bias2.dimshuffle('x', 0, 'x', 'x'))

flattened = out2.flatten(2)
out3 = relu(tensor.dot(flattened, W1) + b1)
prediction = tensor.nnet.sigmoid(tensor.dot(out3, W2) + b2)

loss = tensor.nnet.binary_crossentropy(prediction, T).mean()
error = tensor.gt(tensor.abs_(prediction - T), 0.5).mean()

# Use Blocks to train this network
from blocks.algorithms import GradientDescent, Adam
from blocks.extensions import Printing, ProgressBar
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
from blocks.main_loop import MainLoop
from blocks_extras.extensions.plot import Plot
from blocks.extensions.saveload import Checkpoint

algorithm = GradientDescent(cost=loss, parameters=params,
                            step_rule=Adam())


# We want to monitor the cost as we train
loss.name = 'loss'

train_stream = ServerDataStream(('image_features','targets'), False)
valid_stream = ServerDataStream(('image_features','targets'), False, port=5558)

extensions = [
	TrainingDataMonitoring([loss, error], after_epoch=True),
	DataStreamMonitoring(variables=[error], data_stream=valid_stream, prefix="valid"),
	Plot('Plotting example', channels=[['loss'], ['error', 'valid_error']], after_epoch=True, server_url='http://localhost:8088'),
	Checkpoint('train')
]

main_loop = MainLoop(data_stream=data_stream, algorithm=algorithm,
                     extensions=extensions)
main_loop.run()
