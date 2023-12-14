from typing import List, Dict, Union

import numpy as np
from ray.rllib import SampleBatch
from ray.rllib.models.tf import TFModelV2
import tensorflow as tf
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils.typing import TensorType
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.tf.tf_action_dist import Categorical


class PokemonICMModel(TFModelV2):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):

        self.num_outputs = action_space.n
        self.fcnet_size = model_config.get("fcnet_size")
        self.icm_beta = model_config.get("icm_beta", 0.2)
        self.curiosity_reward_scale = model_config.get("icm_eta", 0.1)
        self.icm_lambda = model_config.get("icm_lambda", 0.1)
        self.learner_bound = model_config["learner_bound"]



        #self.flag_embedding_size = model_config.get("flag_embedding_size")

        super(PokemonICMModel, self).__init__(
            obs_space, action_space, self.num_outputs, model_config, name
        )

        self.view_requirements[SampleBatch.NEXT_OBS] = ViewRequirement(
            SampleBatch.OBS, shift=1, space=self.obs_space, used_for_training=True, used_for_compute_actions=False
        )

        screen_input = tf.keras.layers.Input(shape=obs_space["screen"].shape, name="screen_input",
                                                 dtype=tf.float32)
        stats_input = tf.keras.layers.Input(shape=obs_space["stats"].shape, name="stats_input",
                                                 dtype=tf.float32)

        filters = self.model_config["conv_filters"]

        last_layer = screen_input

        for i, (out_size, kernel, stride, padding) in enumerate(filters, 1):
            last_layer = tf.keras.layers.Conv2D(
                out_size,
                kernel,
                strides=stride
                if isinstance(stride, (list, tuple))
                else (stride, stride),
                activation="relu",
                padding=padding,
                data_format="channels_last",
                name="conv{}".format(i),
            )(last_layer)

        post_cnn = tf.keras.layers.Flatten()(last_layer)

        concat_features = tf.keras.layers.Concatenate(axis=-1, name="screen_and_stats_pre_embedding")(
            [post_cnn, stats_input,
             ]
        )

        fc1 = tf.keras.layers.Dense(
            self.fcnet_size,
            name="fc1",
            activation="relu",
        )(concat_features)

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

        # ICM

        self.base_model = tf.keras.Model(
            [screen_input, stats_input],

            [action_logits, value_out]
        )

        if self.learner_bound:

            action_input = tf.keras.layers.Input(shape=(1,), name="actions", dtype=tf.int32)
            action_one_hot = tf.one_hot(action_input, depth=self.num_outputs, dtype=tf.float32)[:, 0]
            curr_screen_input = tf.keras.layers.Input(shape=obs_space["screen"].shape, name="curr_screen_input",
                                                      dtype=tf.float32)
            next_screen_input = tf.keras.layers.Input(shape=obs_space["screen"].shape, name="next_screen_input",
                                                      dtype=tf.float32)
            next_stats_input = tf.keras.layers.Input(shape=obs_space["stats"].shape, name="next_stats_input",
                                                     dtype=tf.float32)

            curr_state_embedding_input = tf.keras.layers.Input(shape=(320,), name="curr_state_embedding_input",
                                                     dtype=tf.float32)


            cnn_layers = ([tf.keras.layers.Conv2D(
                    out_size,
                    kernel,
                    strides=stride
                    if isinstance(stride, (list, tuple))
                    else (stride, stride),
                    activation="relu",
                    padding=padding,
                    data_format="channels_last",
                    name="ICM_conv{}".format(i),
                ) for i, (out_size, kernel, stride, padding) in enumerate(filters, 1)]
                + [tf.keras.layers.Flatten()])

            state_embedding_concat = tf.keras.layers.Concatenate(axis=-1, name="ICM_state_embedding_concat")

            state_embedding_fc = tf.keras.layers.Dense(
                self.fcnet_size,
                name="ICM_state_embedding_fc",
                activation="relu",
            )

            last_layer_curr = curr_screen_input
            last_layer_next = next_screen_input

            for cnn_layer in cnn_layers:
                last_layer_curr = cnn_layer(last_layer_curr)
                last_layer_next = cnn_layer(last_layer_next)

            curr_state_pre_f1 = state_embedding_concat(
                [last_layer_curr, stats_input]
            )
            next_state_pre_f1 = state_embedding_concat(
                [last_layer_next, next_stats_input]
            )
            curr_state_embedding = state_embedding_fc(curr_state_pre_f1)
            next_state_embedding = state_embedding_fc(next_state_pre_f1)


            action_prediction_input = tf.keras.layers.Concatenate(axis=-1, name="ICM_action_prediction_input")(
                [curr_state_embedding, next_state_embedding]
            )

            action_prediction_fc1 = tf.keras.layers.Dense(
                self.fcnet_size,
                name="ICM_action_prediction_fc1",
                activation="relu",
                #kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )(action_prediction_input)

            action_prediction_logits = tf.keras.layers.Dense(
                self.num_outputs,
                name="ICM_action_logits",
                activation=None,
                #kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )(action_prediction_fc1)

            state_prediction_input = tf.keras.layers.Concatenate(axis=-1, name="ICM_state_prediction_input")(
            [curr_state_embedding_input, action_one_hot,
             ]
            )

            state_prediction_fc1 = tf.keras.layers.Dense(
                self.fcnet_size,
                name="ICM_state_prediction_fc1",
                activation="relu",
                #kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )(state_prediction_input)

            state_prediction_out = tf.keras.layers.Dense(
                320,
                name="ICM_state_prediction_fc2",
                activation=None,
                #kernel_initializer=tf.random_normal_initializer(0, 0.01)
            )(state_prediction_fc1)

            self.icm_forward_model = tf.keras.Model(
                [curr_state_embedding_input, action_input],
                [state_prediction_out]
            )

            self.icm_prediction_model = tf.keras.Model(
            [curr_screen_input, next_screen_input, stats_input, next_stats_input],
            [action_prediction_logits, curr_state_embedding, next_state_embedding]

            )

    def forward(self, input_dict, state, seq_lens):

        self.screen_input = tf.cast(input_dict[SampleBatch.OBS]["screen"], tf.float32) / 255.
        self.stats_inputs = input_dict[SampleBatch.OBS]["stats"]
        allowed_actions = tf.cast(input_dict[SampleBatch.OBS]["allowed_actions"], tf.float32)

        self.actions = input_dict[SampleBatch.ACTIONS]
        next_screen_input = tf.cast(input_dict[SampleBatch.NEXT_OBS]["screen"], tf.float32) / 255.
        next_stats_inputs = input_dict[SampleBatch.NEXT_OBS]["stats"]

        action_logits, self._value_out = self.base_model(
            [self.screen_input, self.stats_inputs]
        )
        allowed_action_logits = action_logits + tf.maximum(tf.math.log(allowed_actions), tf.float32.min)


        self.delta_screen = tf.reduce_sum(tf.math.square(next_screen_input - self.screen_input))


        if self.learner_bound:

            action_prediction_logits, curr_state_embedding, icm_next_state_embedding = self.icm_prediction_model(
                [self.screen_input, next_screen_input, self.stats_inputs, next_stats_inputs]
            )

            #allowed_action_prediction_logits = action_prediction_logits + tf.maximum(tf.math.log(allowed_actions), tf.float32.min)

            self.icm_next_state_embedding = tf.stop_gradient(icm_next_state_embedding)
            self.icm_state_predictions = self.icm_forward_model(
                [tf.stop_gradient(curr_state_embedding), self.actions]
            )
            self.icm_action_predictions = action_prediction_logits

        return allowed_action_logits, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def state_prediction_loss(self):
        return tf.reduce_sum(tf.math.square(self.icm_next_state_embedding - self.icm_state_predictions), axis=-1) * 0.5

    def action_prediction_loss(self):

        action_dist = (
            Categorical(self.icm_action_predictions, self)
        )
        # Neg log(p); p=probability of observed action given the inverse-NN
        # predicted action distribution.
        return -action_dist.logp(tf.convert_to_tensor(self.actions))
        #return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(self.actions), logits=self.icm_action_predictions)


    def metrics(self) -> Dict[str, TensorType]:

        return {
            "delta_screen": self.delta_screen
        }






