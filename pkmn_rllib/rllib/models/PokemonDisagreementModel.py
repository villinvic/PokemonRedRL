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


class PokemonDisagreementMModel(TFModelV2):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):

        self.num_outputs = action_space.n
        self.fcnet_size = model_config.get("fcnet_size")
        self.learner_bound = model_config["learner_bound"]

        self.n_models = model_config.get("n_disagreement_models", 3)
        self.intrinsic_reward_scale = model_config.get("intrinsic_reward_scale", 0.01)
        self.intrinsic_reward_ratio = model_config.get("intrinsic_reward_scale", 0.5)
        self.forward_loss_ratio = model_config.get("forward_loss_ratio", 0.)


        self.state_embedding_size = model_config.get("state_embedding_size", 512)



        #self.flag_embedding_size = model_config.get("flag_embedding_size")

        super(PokemonDisagreementMModel, self).__init__(
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
                activation="elu",
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
            activation="elu",
        )(concat_features)

        fc2 = tf.keras.layers.Dense(
            self.fcnet_size,
            name="fc2",
            activation="elu",
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
            [screen_input, stats_input],

            [action_logits, value_out]
        )

        if self.learner_bound:

            action_input = tf.keras.layers.Input(shape=(1,), name="actions", dtype=tf.int32)
            action_one_hot = tf.one_hot(action_input, depth=self.num_outputs, dtype=tf.float32)[:, 0]
            curr_screen_input = tf.keras.layers.Input(shape=(36*3,) + obs_space["screen"].shape[1:], name="curr_screen_input", #(36,) + obs_space["screen"].shape[1:]
                                                      dtype=tf.float32)
            next_screen_input = tf.keras.layers.Input(shape=(36*3,) + obs_space["screen"].shape[1:], name="next_screen_input",
                                                      dtype=tf.float32)
            next_stats_input = tf.keras.layers.Input(shape=obs_space["stats"].shape, name="next_stats_input",
                                                     dtype=tf.float32)

            curr_state_embedding_input = tf.keras.layers.Input(shape=(self.state_embedding_size,), name="curr_state_embedding_input",
                                                     dtype=tf.float32)


            cnn_layers = ([tf.keras.layers.Conv2D(
                    out_size,
                    kernel,
                    strides=stride
                    if isinstance(stride, (list, tuple))
                    else (stride, stride),
                    activation="elu",
                    padding=padding,
                    data_format="channels_last",
                    name="ICM_conv{}".format(i),
                ) for i, (out_size, kernel, stride, padding) in enumerate(filters, 1)]
                + [tf.keras.layers.Flatten()])

            state_embedding_concat = tf.keras.layers.Concatenate(axis=-1, name="ICM_state_embedding_concat")

            # state_pre_embedding_fc = tf.keras.layers.Dense(
            #     self.fcnet_size,
            #     name="ICM_state_pre_embedding_fc",
            #     activation="elu",
            # )

            state_embedding_fc = tf.keras.layers.Dense(
                self.state_embedding_size,
                name="ICM_state_embedding_fc",
                activation="elu",
            )

            # def layernorm(x):
            #     m, v = tf.nn.moments(x, -1, keepdims=True)
            #     return (x - m) / (tf.math.sqrt(v) + 1e-8)

            # stats_normalization_layer = tf.keras.layers.BatchNormalization(
            #     momentum=0.98,
            #     name="ICM_stats_normalization_layer"
            # )
            # screen_normalization_layer = tf.keras.layers.BatchNormalization(
            #     momentum=0.98,
            #     name="ICM_screen_normalization_layer"
            # )

            last_layer_curr = curr_screen_input#screen_normalization_layer(curr_screen_input, training=True)
            last_layer_next = next_screen_input#screen_normalization_layer(next_screen_input, training=False)

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

            # action_prediction_input = tf.keras.layers.Concatenate(axis=-1, name=f"ICM_action_prediction_input")(
            #     [curr_state_embedding, next_state_embedding]
            # )

            # action_prediction_fc = tf.keras.layers.Dense(
            #     self.state_embedding_size,
            #     name="ICM_action_prediction_fc",
            #     activation="elu",
            # )(action_prediction_input)
            #
            # action_prediction_logits = tf.keras.layers.Dense(
            #     self.num_outputs,
            #     name="ICM_action_prediction_logits",
            #     activation=None,
            # )(action_prediction_fc)

            self.state_embedding_model = tf.keras.Model(
                [curr_screen_input, stats_input, next_screen_input, next_stats_input],
                [
                    #action_prediction_logits,
                    curr_state_embedding, next_state_embedding]
            )

            self.disagreement_models = []

            action_concat_layer = tf.keras.layers.Concatenate(axis=-1, name=f"ICM_embed_action_concat")

            def concat_action(embed):
                return action_concat_layer([embed, action_one_hot])

            for i in range(self.n_models):

                features = tf.keras.layers.Dense(self.state_embedding_size, activation="elu",
                                                 name=f"ICM_pre_res_{i}"
                                                 )(concat_action(curr_state_embedding_input))
                def residual(x, idx):

                    res = tf.keras.layers.Dense(self.state_embedding_size, activation="elu",
                                                name=f"ICM_res_{i}{idx}1"
                                                )(concat_action(x))
                    res = tf.keras.layers.Dense(self.state_embedding_size, activation=None,
                                                name=f"ICM_res_{i}{idx}2"
                                                )(concat_action(res))
                    return x + res

                for j in range(4):
                    features = residual(features, j)

                state_prediction_out = tf.keras.layers.Dense(
                    self.state_embedding_size,
                    name=f"ICM_prediction_out_{i}",
                    activation=None,
                )(features)

                self.disagreement_models.append(tf.keras.Model(
                    [curr_state_embedding_input, action_input],
                    [state_prediction_out]
                ))

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

        if self.learner_bound:

            self.clipped_curr_screen = self.screen_input[:, :36*3]
            self.clipped_next_screen = next_screen_input[:, :36*3]


            curr_state_embedding, next_state_embedding = self.state_embedding_model(
                [self.clipped_curr_screen, self.stats_inputs, self.clipped_next_screen, next_stats_inputs]
            )

            self.curr_state_embedding = tf.stop_gradient(curr_state_embedding)
            self.next_state_embedding = tf.stop_gradient(next_state_embedding)
            self.delta_image = tf.reduce_mean(tf.reduce_mean(tf.math.square(self.clipped_curr_screen - self.clipped_next_screen)[:, :, :, 0], axis=-1),
                                              axis=-1)

            self.predicted_state_embeddings = [
                model([self.curr_state_embedding, self.actions]) for model in self.disagreement_models
            ]

        return allowed_action_logits, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def state_prediction_loss(self):

        loss = sum([
            tf.reduce_sum(tf.math.square(
                tf.nn.dropout(self.next_state_embedding - predicted_state_embedding,
                              rate=0.2
                             )
            ), axis=-1)
            for predicted_state_embedding
            in self.predicted_state_embeddings
        ])

        return loss

    def compute_intrinsic_rewards(self):
        return tf.reduce_mean(tf.math.reduce_variance(self.predicted_state_embeddings, axis=0), axis=-1)

    def embedding_distance(self):
        return self.delta_image

    # def action_prediction_loss(self):
    #
    #     action_dist = (
    #         Categorical(self.action_prediction_logits, self)
    #     )
    #     # Neg log(p); p=probability of observed action given the inverse-NN
    #     # predicted action distribution.
    #     return -action_dist.logp(tf.convert_to_tensor(self.actions))

    def metrics(self) -> Dict[str, TensorType]:

        return {
            #"delta_screen": self.delta_screen
        }






