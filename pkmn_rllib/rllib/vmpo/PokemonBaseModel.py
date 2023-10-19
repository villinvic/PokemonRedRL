from typing import List

import numpy as np
from ray.rllib import SampleBatch
from ray.rllib.models import ModelV2
from ray.rllib.models.tf import TFModelV2
import tensorflow as tf
from ray.rllib.utils import override

class PokemonBaseModel(TFModelV2):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):

        # learns to play as one character, against many characters

        self.num_outputs = action_space.n
        self.fcnet_size = model_config.get("fcnet_size", 64)

        super(PokemonBaseModel, self).__init__(
            obs_space, action_space, self.num_outputs, model_config, name
        )

        img_input = tf.keras.layers.Input(shape=obs_space.shape, name="img_input",
                                                 dtype=tf.float32)

        print(self.model_config)

        filters = self.model_config["conv_filters"]

        last_layer = img_input

        for i, (out_size, kernel, stride) in enumerate(filters, 1):
            last_layer = tf.keras.layers.Conv2D(
                out_size,
                kernel,
                strides=stride
                if isinstance(stride, (list, tuple))
                else (stride, stride),
                activation="relu",
                padding="valid" if i < len(filters) else "valid",
                data_format="channels_last",
                name="conv{}".format(i),
            )(last_layer)

        post_cnn = tf.keras.layers.Flatten()(last_layer)

        fc1 = tf.keras.layers.Dense(
            self.fcnet_size,
            name="fc1",
            activation="relu",
        )(post_cnn)

        fc2 = tf.keras.layers.Dense(
            self.fcnet_size,
            name="fc2",
            activation="relu",
        )(fc1)

        action_logits = tf.keras.layers.Dense(
            self.num_outputs,
            name="action_logits",
            activation=None,
        )(fc2)

        # Value Function

        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            # kernel_initializer=tf.random_normal_initializer(0., 1.),
            bias_initializer=tf.zeros_initializer(),
        )(fc2)

        self.base_model = tf.keras.Model(
            [img_input],
            [action_logits, value_out])

    def forward(self, input_dict, state, seq_lens):

        img_input = tf.cast(input_dict[SampleBatch.OBS], tf.float32) / 255.

        context, self._value_out = self.base_model(
            [img_input]
        )
        return tf.reshape(context, [-1, self.num_outputs]), state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])