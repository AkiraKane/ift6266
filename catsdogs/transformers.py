from fuel.transformers import ExpectsAxisLabels, SourcewiseTransformer
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
