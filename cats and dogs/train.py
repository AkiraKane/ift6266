from fuel.streams import ServerDataStream
data_stream = ServerDataStream(('image_features','targets'), False)

# Create the Theano MLP
import theano
from theano import tensor
import numpy

X = tensor.matrix('image_features')
T = tensor.matrix('targets')

W = theano.shared(
    numpy.random.uniform(low=-0.01, high=0.01, size=(3072, 500)), 'W')
b = theano.shared(numpy.zeros(500))
V = theano.shared(
    numpy.random.uniform(low=-0.01, high=0.01, size=(500, 1)), 'V')
c = theano.shared(numpy.zeros(1))
params = [W, b, V, c]

H = tensor.nnet.sigmoid(tensor.dot(X, W) + b)
Y = tensor.nnet.sigmoid(tensor.dot(H, V) + c)

loss = tensor.nnet.binary_crossentropy(Y, T).mean()

# Use Blocks to train this network
from blocks.algorithms import GradientDescent, Scale
from blocks.extensions import Printing
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.main_loop import MainLoop

algorithm = GradientDescent(cost=loss, parameters=params,
                            step_rule=Scale(learning_rate=0.1))

# We want to monitor the cost as we train
loss.name = 'loss'
extensions = [TrainingDataMonitoring([loss]),
              Printing(every_n_batches=1)]

main_loop = MainLoop(data_stream=data_stream, algorithm=algorithm,
                     extensions=extensions)
main_loop.run()