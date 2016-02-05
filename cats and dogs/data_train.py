# Let's load and process the dataset
from fuel.datasets.dogs_vs_cats import DogsVsCats
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.server import start_server
from fuel.transformers.image import RandomFixedSizeCrop, MinimumImageDimensions, DownscaleMinDimension
from fuel.transformers import Flatten, ScaleAndShift, Cast
import socket
import numpy

# Load the training set
if socket.gethostname() == 'yop':
	sub = slice(0, 1500)
else:
	sub = slice(0, 15000)
train = DogsVsCats(('train',), subset=sub)

# We now create a "stream" over the dataset which will return shuffled batches
# of size 128. Using the DataStream.default_stream constructor will turn our
# 8-bit images into floating-point decimals in [0, 1].
stream = DataStream.default_stream(
    train,
    iteration_scheme=SequentialScheme(train.num_examples, 100)
)

upscaled_stream = MinimumImageDimensions(stream, (100, 100), which_sources=('image_features',))
downscaled_stream = DownscaleMinDimension(upscaled_stream, 100, which_sources=('image_features',))

# Our images are of different sizes, so we'll use a Fuel transformer
# to take random crops of size (32 x 32) from each image
cropped_stream = RandomFixedSizeCrop(
    downscaled_stream, (100, 100), which_sources=('image_features',))

# We'll use a simple MLP, so we need to flatten the images
# from (channel, width, height) to simply (features,)
float_stream = ScaleAndShift(cropped_stream, 1./255, 0, which_sources=('image_features',))
float32_stream = Cast(float_stream, numpy.float32, which_sources=('image_features',))

start_server(float32_stream)
