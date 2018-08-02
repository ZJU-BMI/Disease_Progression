# coding=utf-8
import tensorflow as tf

import rnn_config as config
from neural_network import intensity, attention_mechanism, revised_rnn, prediction


class ProposedModel(object):
    def __init__(self, model_config):
        self.model_config = model_config
        self.loss_ratio = model_config.c_r_ratio

        # key node
        self.c_loss = None
        self.r_loss = None
        self.loss = None
        self.c_pred_list = None
        self.r_pred_list = None
        self.merged_summary = None
        self.placeholder_x = None
        self.placeholder_t = None

    def __call__(self, **kwargs):
        """
        build the computational graph of model
        :param kwargs:
        :return:
        """
        placeholder_x = kwargs['placeholder_x']
        placeholder_t = kwargs['placeholder_t']
        mutual_intensity_placeholder = kwargs['mutual_intensity']

        self.placeholder_x = placeholder_x
        self.placeholder_t = placeholder_t

        model_config = self.model_config
        # component define
        revise_gru_rnn = revised_rnn.RevisedRNN(model_configuration=model_config)

        attention_model = \
            attention_mechanism.HawkesBasedAttentionLayer(model_configuration=model_config,
                                                          mutual_intensity_placeholder=mutual_intensity_placeholder)
        attention_layer = prediction.AttentionMixLayer(model_configuration=model_config,
                                                       mutual_intensity=mutual_intensity_placeholder,
                                                       revise_rnn=revise_gru_rnn, attention=attention_model)
        prediction_layer = prediction.PredictionLayer(model_configuration=model_config)

        # model construct
        mix_state_list = attention_layer(input_x=placeholder_x, input_t=placeholder_t)
        c_loss, r_loss, c_pred_list, r_pred_list, c_label, r_label = \
            prediction_layer(mix_hidden_state_list=mix_state_list, input_x=placeholder_x, input_t=placeholder_t)
        prediction.performance_summary(input_x=c_label, input_t=r_label, c_pred=c_pred_list,
                                       r_pred=r_pred_list, threshold=model_config.threshold)

        merged_summary = tf.summary.merge_all()

        self.c_loss = c_loss
        self.r_loss = r_loss
        self.c_pred_list = c_pred_list
        self.r_pred_list = r_pred_list
        self.merged_summary = merged_summary
        with tf.name_scope('loss_sum'):
            self.loss = c_loss + self.loss_ratio * r_loss
        mi = mutual_intensity_placeholder
        return self.loss, c_pred_list, r_pred_list, merged_summary, mi


def unit_test():
    _, model_config = config.validate_configuration_set()
    intensity_obj = intensity.Intensity(model_config)
    base_intensity = intensity_obj.base_intensity
    mutual_intensity = intensity_obj.mutual_intensity
    placeholder_x = tf.placeholder('float64', [model_config.max_time_stamp, model_config.batch_size,
                                               model_config.input_x_depth])
    placeholder_t = tf.placeholder('float64', [model_config.max_time_stamp, model_config.batch_size,
                                               model_config.input_t_depth])
    proposed_model = ProposedModel(model_config)
    _ = proposed_model(placeholder_x=placeholder_x, placeholder_t=placeholder_t,
                       mutual_intensity=mutual_intensity, base_intensity=base_intensity)


if __name__ == "__main__":
    unit_test()
