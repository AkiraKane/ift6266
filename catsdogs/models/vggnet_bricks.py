from blocks.bricks import Rectifier, Logistic, FeedforwardSequence, Initializable, MLP
from blocks.bricks.conv import Convolutional, ConvolutionalSequence, Flattener, MaxPooling
from blocks.graph import ComputationGraph
from blocks.initialization import Constant, Uniform
import numpy

from theano import tensor

class VGGNet(FeedforwardSequence, Initializable):

    def __init__(self, image_dimension, **kwargs):

        layers = []
        
        #############################################
        # a first block with 2 convolutions of 32 (3, 3) filters
        layers.append(Convolutional((3, 3), 32))
        layers.append(Convolutional((3, 3), 32))

        # maxpool with size=(2, 2)
        layers.append(MaxPooling((2, 2)))

        #############################################
        # a 2nd block with 3 convolutions of 64 (3, 3) filters
        layers.append(Convolutional((3, 3), 64))
        layers.append(Convolutional((3, 3), 64))
        layers.append(Convolutional((3, 3), 64))
        
        # maxpool with size=(2, 2)
        layers.append(MaxPooling((2, 2)))

        #############################################
        # a 3rd block with 4 convolutions of 128 (3, 3) filters
        layers.append(Convolutional((3, 3), 128))
        layers.append(Convolutional((3, 3), 128))
        layers.append(Convolutional((3, 3), 128))
        layers.append(Convolutional((3, 3), 128))
        
        # maxpool with size=(2, 2)
        layers.append(MaxPooling((2, 2)))

        self.conv_sequence = ConvolutionalSequence(layers, 3, image_size=image_dimension)

        flattener = Flattener()

        self.top_mlp = MLP(activations=[Rectifier(), Logistic()], dims=[500, 1])

        application_methods = [self.conv_sequence.apply, flattener.apply, self.top_mlp.apply]

        super(VGGNet, self).__init__(application_methods, biases_init=Constant(0), weights_init=Uniform(width=.2), **kwargs)


    def _push_allocation_config(self):
        self.conv_sequence._push_allocation_config()
        conv_out_dim = self.conv_sequence.get_dim('output')
        
        self.top_mlp.dims = [numpy.prod(conv_out_dim)] + self.top_mlp.dims



def get_model(X, batch_size, image_dimension):

    vgg = VGGNet(image_dimension)

    vgg.push_initialization_config()
    vgg.initialize()

    output = vgg.apply(X)

    output_test1 = vgg.apply(X[:,:,:image_dimension[0],:image_dimension[1]])
    output_test2 = vgg.apply(X[:,:,-image_dimension[0]:,-image_dimension[1]:])

    output_test = tensor.switch(tensor.ge(tensor.abs_(output_test1-0.5), tensor.abs_(output_test2-0.5)), output_test1, output_test2)

    return output, output_test2
