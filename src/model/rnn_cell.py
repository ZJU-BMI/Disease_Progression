from tensorflow.contrib.rnn import LayerRNNCell
from tensorflow.contrib.rnn.ops import gen_lstm_ops
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell_impl


# 参考Tensorflow源代码,对BasicLSTM进行了部分改进，如果以后顺利，可以在修改Fused/Block RNN的基础上加快速度
class BlockLSTMChangeForgetGate(LayerRNNCell):
    # change the way to calculate the forget gate
    def __init__(self, num_units, forget_bias=1.0, cell_clip=None, use_peephole=False, reuse=None,
                 name='lstm__cell_change_f_gate'):
        super(BlockLSTMChangeForgetGate, self).__init__(_reuse=reuse, name=name)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._use_peephole = use_peephole
        self._cell_clip = cell_clip if cell_clip is not None else -1
        self._names = {
            "W": "kernel",
            "b": "bias",
            "wci": "w_i_diag",
            "wcf": "w_f_diag",
            "wco": "w_o_diag",
            "scope": "lstm_cell"
        }
        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=2)

        self._kernel = None
        self._bias = None

        # peephole
        self._w_i_diag = None
        self._w_f_diag = None
        self._w_o_diag = None

    @property
    def state_size(self):
        return rnn_cell_impl.LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def build(self, input_shape):
        input_size = input_shape[2].value
        self._kernel = self.add_variable(
            "kernel", [input_size + self._num_units, self._num_units * 4])
        self._bias = self.add_variable(
            "bias", [self._num_units * 4],
            initializer=init_ops.constant_initializer(0.0))
        if self._use_peephole:
            self._w_i_diag = self.add_variable("w_i_diag", [self._num_units])
            self._w_f_diag = self.add_variable("w_f_diag", [self._num_units])
            self._w_o_diag = self.add_variable("w_o_diag", [self._num_units])

        self.built = True

    def _call_cell(self, inputs, initial_cell_state=None, initial_output=None, dtype=None, sequence_length=None):

        inputs_shape = inputs.get_shape().with_rank(3)
        time_len = inputs_shape[0].value
        if time_len is None:
            time_len = array_ops.shape(inputs)[0]

        if self._use_peephole:
            wci = self._w_i_diag
            wco = self._w_o_diag
            wcf = self._w_f_diag
        else:
            wci = wcf = wco = array_ops.zeros([self._num_units], dtype=dtype)

        if sequence_length is None:
            max_seq_len = math_ops.to_int64(time_len)
        else:
            max_seq_len = math_ops.to_int64(math_ops.reduce_max(sequence_length))

        _, cs, _, _, _, _, h = gen_lstm_ops.block_lstm(
            seq_len_max=max_seq_len,
            x=inputs,
            cs_prev=initial_cell_state,
            h_prev=initial_output,
            w=self._kernel,
            wci=wci,
            wcf=wcf,
            wco=wco,
            b=self._bias,
            forget_bias=self._forget_bias,
            cell_clip=self._cell_clip,
            use_peephole=self._use_peephole)
        return cs, h


class BlockLSTMChangeInput(LayerRNNCell):
    pass
