import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell


class RecurrentNeuralNetwork(LayerRNNCell):
    """
    The implementation is based on official tensorflow source code (rnn_cell_impl.py)

    RecurrentNeuralNetwork inherit LayerRNNCell while LayerRNNCell inherit RNNCell and RNNCell inherits Layer


    RNNCell is the fundamental abstract class of rnn. It defines many abstract functions which need concrete
    implementation in its children class
    abstract functions contains output_size and state_size. meanwhile, the build function doesn't have function body

    LayerRNNCell almost does nothing other than implements the __call__ function, which, make the instance of this
    class callable


    """
    def __init__(self):
        return 1

    def build(self, _):
        return 1

    @property
    def output_size(self):
        return 1

    @property
    def state_size(self):
        return 1

    def call(self, inputs, **kwargs):
        return 1


class BidirectionalRnn(RecurrentNeuralNetwork):
    pass


class RnnWithAttention(RecurrentNeuralNetwork):
    pass


class BidirectionalRnnWithAttention(RnnWithAttention):
    pass


def main():
    pass


if __name__ == "__main__":
    main()