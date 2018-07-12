from tensorflow.contrib.rnn import LayerRNNCell


# 参考Tensorflow源代码,对BasicLSTM进行了部分改进，如果以后顺利，可以在修改Fused/Block RNN的基础上加快速度
class BlockLSTM1(LayerRNNCell):
    """
    change the way calculate input, using peephole connection
    """
    def __init__(self, num_units, forget_bias=1.0, cell_clip=None, use_peephole=False, reuse=None,
                 name='lstm__cell_change_f_gate'):
        pass


class BlockLSTM2(LayerRNNCell):
    """
    directly using time information to decide value of forget gate
    """
