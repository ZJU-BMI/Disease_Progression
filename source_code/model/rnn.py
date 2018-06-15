import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell


class RecurrentNeuralNetwork(LayerRNNCell):
    """
    The implementation is based on official tensorflow source code (rnn_cell_impl.py)

    RecurrentNeuralNetwork inherit LayerRNNCell while LayerRNNCell inherit RNNCell and RNNCell inherits Layer

    Layer defines the basic API of a hidden layer of neural network. However, it also leaves some unimplemented
    functions, such as 'call' function

    RNNCell is the fundamental abstract class of rnn. it defines abstract structure of a RNN Layer, which insists of
    several abstract function. In detail, it does things listed below
    1. it overrides of function 'call', actually defines the output of this structure. the 'call' function is a
    auxiliary function affiliated to __call__, which is defined in 'Layer'. the 'call' decides the output of a layer.
    RNN is different from CNN and DNN, the output of a rnn layer should contain both output and the next state.
    2. it defines how to set the zero state of RNN
    3. it defines some abstract functions that subclass must implemented
    Note, the term 'Cell' in Tensorflow actually means an array of unit rather than a scalar.

    LayerRNNCell almost does nothing. Actually I don't understand the function of this class.
    I think the name 'LayerRNNCell' means a hidden layer consists of RNN Cell, the term RNN Cell means the cell
    output not only the 'output', but also output the 'next state'

    LayerRNN have several subclasses, such as BasicRNNCell, LSTMRNNCell, GRURNNCell etc., note all objects of such
    classes are hidden layers, not a real cell. such classes actually just implemented some abstract functions.
    to be concrete, 2 functions.
    call: set the math detail of a layer, input of this function are data and previous state while the output are new
    state and output.
    Note, most RNN don't distinguish the difference between new state and output, they treat them as same thing

    build: set the parameter matrices of layer, i.e., weight matrix and bias.

    """
    def __init__(self, network_type, state_num, layer_length, activation):
        self.network_type = network_type
        self.state_num = state_num
        self.layer_length = layer_length

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