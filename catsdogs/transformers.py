from fuel.transformers import ExpectsAxisLabels, SourcewiseTransformer
from PIL import Image
import math
import numpy
import random

class FixedSizeCrops(SourcewiseTransformer, ExpectsAxisLabels):
    """Randomly crop images to a fixed window size.
    Parameters
    ----------
    data_stream : :class:`AbstractDataStream`
        The data stream to wrap.
    window_shape : tuple
        The `(height, width)` tuple representing the size of the output
        window.
    """
    def __init__(self, data_stream, window_shape, **kwargs):
        self.window_shape = window_shape
        self.warned_axis_labels = False
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        super(FixedSizeCrops, self).__init__(data_stream, **kwargs)

    def transform_source_batch(self, source, source_name):
        return [self.transform_source_example(im, source_name)
                    for im in source]

    def transform_source_example(self, example, source_name):
        windowed_height, windowed_width = self.window_shape

        output = numpy.zeros((example.shape[0], windowed_height, windowed_width*2))
        output[:,:,:windowed_width] = example[:,:windowed_height,:windowed_width]
        output[:,:,-windowed_width:] = example[:,-windowed_height:,-windowed_width:]

        return output

class RandomHorizontalFlip(SourcewiseTransformer):
    """

    """
    def __init__(self, data_stream, **kwargs):
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        super(RandomHorizontalFlip, self).__init__(data_stream, **kwargs)

    def transform_source_batch(self, batch, source_name):
        return [self._example_transform(im, source_name) for im in batch]

    def transform_source_example(self, example, source_name):
        return self._example_transform(example, source_name)

    def _example_transform(self, example, source_name):
        flip = random.randint(0, 1)*2-1
        return example[:,:,::flip]

class DownscaleMinDimension(SourcewiseTransformer, ExpectsAxisLabels):
    """Resize (lists of) images so that their shortest dimension is of a given size.
    Parameters
    ----------
    data_stream : instance of :class:`AbstractDataStream`
        The data stream to wrap.
    min_dimension_size : int
        The desired length of the smallest dimension.
    resample : str, optional
        Resampling filter for PIL to use to upsample any images requiring
        it. Options include 'nearest' (default), 'bilinear', and 'bicubic'.
        See the PIL documentation for more detailed information.
    Notes
    -----
    This transformer expects stream sources returning individual images,
    represented as 2- or 3-dimensional arrays, or lists of the same.
    The format of the stream is unaltered.
    """
    def __init__(self, data_stream, min_dimension_size, resample='nearest',
                 **kwargs):
        self.min_dimension_size = min_dimension_size
        try:
            self.resample = getattr(Image, resample.upper())
        except AttributeError:
            raise ValueError("unknown resampling filter '{}'".format(resample))
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(DownscaleMinDimension, self).__init__(data_stream, **kwargs)

    def transform_source_batch(self, batch, source_name):
        self.verify_axis_labels(('batch', 'channel', 'height', 'width'),
                                self.data_stream.axis_labels[source_name],
                                source_name)
        return [self._example_transform(im, source_name) for im in batch]

    def transform_source_example(self, example, source_name):
        self.verify_axis_labels(('channel', 'height', 'width'),
                                self.data_stream.axis_labels[source_name],
                                source_name)
        return self._example_transform(example, source_name)

    def _example_transform(self, example, _):
        if example.ndim > 3 or example.ndim < 2:
            raise NotImplementedError
        original_height, original_width = example.shape[-2:]
        if True or original_height > self.min_dimension_size and original_width > self.min_dimension_size:
            dt = example.dtype
            # If we're dealing with a colour image, swap around the axes
            # to be in the format that PIL needs.
            if example.ndim == 3:
                im = example.transpose(1, 2, 0)
            else:
                im = example

            im = Image.fromarray(im)
            width, height = im.size
            multiplier = max(float(self.min_dimension_size) / width, float(self.min_dimension_size) / height)
            width = int(math.ceil(width * multiplier))
            height = int(math.ceil(height * multiplier))
            im = numpy.array(im.resize((width, height))).astype(dt)
            # If necessary, undo the axis swap from earlier.
            if im.ndim == 3:
                example = im.transpose(2, 0, 1)
            else:
                example = im
        return example
