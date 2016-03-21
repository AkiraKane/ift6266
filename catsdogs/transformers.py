from fuel.transformers import ExpectsAxisLabels, SourcewiseTransformer
import numpy

class FixedSizeCrops(SourcewiseTransformer, ExpectsAxisLabels):
    """Randomly crop images to a fixed window size.
    Parameters
    ----------
    data_stream : :class:`AbstractDataStream`
        The data stream to wrap.
    window_shape : tuple
        The `(height, width)` tuple representing the size of the output
        window.
    Notes
    -----
    This transformer expects to act on stream sources which provide one of
     * Single images represented as 3-dimensional ndarrays, with layout
       `(channel, height, width)`.
     * Batches of images represented as lists of 3-dimensional ndarrays,
       possibly of different shapes (i.e. images of differing
       heights/widths).
     * Batches of images represented as 4-dimensional ndarrays, with
       layout `(batch, channel, height, width)`.
    The format of the stream will be un-altered, i.e. if lists are
    yielded by `data_stream` then lists will be yielded by this
    transformer.
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