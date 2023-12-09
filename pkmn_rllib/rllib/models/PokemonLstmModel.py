from typing import List, Dict, Union

import numpy as np
from ray.rllib import SampleBatch
from ray.rllib.models.tf import TFModelV2
import tensorflow as tf
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils.typing import TensorType
from ray.rllib.models.modelv2 import restore_original_dimensions

class PokemonLstmModel(TFModelV2):

    N_MAPS = 248

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):

        # learns to play as one character, against many characters

        self.num_outputs = action_space.n
        self.fcnet_size = model_config.get("fcnet_size")
        self.lstm_size = model_config.get("lstm_size")
        #self.flag_embedding_size = model_config.get("flag_embedding_size")

        super(PokemonLstmModel, self).__init__(
            obs_space, action_space, self.num_outputs, model_config, name
        )

        self.view_requirements[SampleBatch.PREV_ACTIONS] = ViewRequirement(
            SampleBatch.ACTIONS, space=self.action_space, shift=-1
        )
        self.view_requirements[SampleBatch.PREV_REWARDS] = ViewRequirement(
            SampleBatch.REWARDS, shift=-1
        )
        self.view_requirements[SampleBatch.NEXT_OBS] = ViewRequirement(
            SampleBatch.OBS, shift=1, space=self.obs_space, used_for_training=True, used_for_compute_actions=False
        )

        screen_input = tf.keras.layers.Input(shape=obs_space["screen"].shape, name="screen_input",
                                                 dtype=tf.float32)
        stats_input = tf.keras.layers.Input(shape=obs_space["stats"].shape, name="stats_input",
                                                 dtype=tf.float32)
        # flags_input = tf.keras.layers.Input(shape=obs_space["flags"].shape, name="flags_input",
        #                                          dtype=tf.float32)

        previous_action_input = tf.keras.layers.Input(shape=(1,), name="prev_actions", dtype=tf.int32)
        prev_action_one_hot = tf.one_hot(previous_action_input, depth=self.num_outputs, dtype=tf.float32)[:, 0]

        previous_reward_input = tf.keras.layers.Input(shape=(1,), name="prev_rewards", dtype=tf.float32)

        action_input = tf.keras.layers.Input(shape=(1,), name="actions", dtype=tf.int32)
        action_one_hot = tf.one_hot(action_input, depth=self.num_outputs, dtype=tf.float32)[:, 0]

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

        # flags_embedding = tf.keras.layers.Dense(
        #     self.flag_embedding_size,
        #     name="flags_dense_embedding",
        #     activation="tanh",
        # )(flags_input)

        concat_features = tf.keras.layers.Concatenate(axis=-1, name="screen_and_stats_pre_embedding")(
            [post_cnn, stats_input,
            # flags_embedding
             ]
        )

        # pre_lstm_prediction_input = tf.keras.layers.Concatenate(axis=-1, name="pre_lstm_prediction_input")(
        #     [post_cnn, action_one_hot]
        # )

        fc1 = tf.keras.layers.Dense(
            self.fcnet_size,
            name="fc1",
            activation="relu",
        )(concat_features)

        moved_per_actions = tf.keras.layers.Dense(
            self.num_outputs,
            name="moved_logits",
            activation=None,
        )(fc1)

        moved_logits = tf.reduce_sum(moved_per_actions * action_one_hot, axis=-1)

        # fc2 = tf.keras.layers.Dense(
        #     self.fcnet_size,
        #     name="fc2",
        #     activation="relu",
        # )(fc1)

        # fc3 = tf.keras.layers.Dense(
        #     self.fcnet_size,
        #     name="fc3",
        #     activation="relu",
        # )(fc2)

        state_in_h = tf.keras.layers.Input(shape=(self.lstm_size,), name="h")
        state_in_c = tf.keras.layers.Input(shape=(self.lstm_size,), name="c")
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)

        clipped_reward = tf.clip_by_value(previous_reward_input, -1., 1.)
        lstm_input = tf.keras.layers.Concatenate(axis=-1, name="lstm_input")(
            [fc1, clipped_reward, prev_action_one_hot])

        timed_input = add_time_dimension(
            padded_inputs=lstm_input, seq_lens=seq_in, framework="tf"
        )

        # Preprocess observation with a hidden layer and send to LSTM cell
        lstm_out, state_h, state_c = tf.keras.layers.LSTM(
            self.lstm_size, return_sequences=True, return_state=True, name="pi_lstm",
        )(
            inputs=timed_input,
            mask=tf.sequence_mask(seq_in),
            initial_state=[state_in_h, state_in_c],
        )

        lstm_out = tf.reshape(lstm_out, [-1, self.lstm_size])

        concat_post_lstm = tf.keras.layers.Concatenate(axis=-1, name="concat_post_lstm")(
            [lstm_out, moved_per_actions])

        fc_post_lstm = tf.keras.layers.Dense(
            self.fcnet_size,
            name="fc_post_lstm",
            activation="relu",
        )(concat_post_lstm)

        fc_post_lstm_prediction = tf.keras.layers.Dense(
            self.fcnet_size,
            name="fc_post_lstm_prediction",
            activation="relu",
        )(concat_post_lstm)


        action_logits = tf.keras.layers.Dense(
            self.num_outputs,
            name="action_logits",
            activation=None,
        )(fc_post_lstm)

        # Value Function

        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            # kernel_initializer=tf.random_normal_initializer(0., 1.),
            bias_initializer=tf.zeros_initializer(),
        )(fc_post_lstm)

        # Prediction

        # prediction_input = tf.keras.layers.Concatenate(axis=-1, name="prediction_input")(
        #     [lstm_out]
        # )

        reward_prediction_logits = tf.keras.layers.Dense(
            3,
            name="reward_logits",
            activation=None,
        )(fc_post_lstm_prediction)

        map_logits = tf.keras.layers.Dense(
            self.N_MAPS,
            name="map_logits",
            activation=None,
        )(fc_post_lstm_prediction)


        self.base_model = tf.keras.Model(
            [screen_input, stats_input,
             #flags_input,
             previous_reward_input, previous_action_input, action_input,
             seq_in, state_in_h, state_in_c],

            [action_logits, value_out, map_logits, moved_logits, reward_prediction_logits, state_h, state_c]
        )

    def forward(self, input_dict, state, seq_lens):

        screen_input = tf.cast(input_dict[SampleBatch.OBS]["screen"], tf.float32) / 255.
        stat_inputs = input_dict[SampleBatch.OBS]["stats"]
        #flags_inputs = input_dict[SampleBatch.OBS]["flags"]
        self.map_ids = tf.cast(input_dict[SampleBatch.OBS]["coordinates"], tf.int32)
        self.moved = tf.cast(input_dict[SampleBatch.NEXT_OBS]["moved"], tf.int32)
        self.rewards = tf.squeeze(input_dict[SampleBatch.REWARDS])
        prev_reward = input_dict[SampleBatch.PREV_REWARDS]
        prev_action = input_dict[SampleBatch.PREV_ACTIONS]
        action = input_dict[SampleBatch.ACTIONS]


        context, self._value_out, map_logits, moved_logits, reward_logits, h, c = self.base_model(
            [screen_input, stat_inputs,
             #flags_inputs,
             prev_reward, prev_action, action,
             seq_lens] + state
        )

        self.map_logits = tf.reshape(map_logits, [-1, self.N_MAPS])
        self.moved_logits = tf.reshape(moved_logits, [-1])
        self.reward_logits = tf.reshape(reward_logits, [-1, 3])


        return tf.reshape(context, [-1, self.num_outputs]), [h, c]

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def get_initial_state(self) -> List[np.ndarray]:

        return [
            np.zeros(self.lstm_size, np.float32),
            np.zeros(self.lstm_size, np.float32)
        ]

    def custom_loss(
        self, policy_loss: TensorType, loss_inputs: Dict[str, TensorType]
    ) -> Union[List[TensorType], TensorType]:

        map_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(self.map_ids), logits=self.map_logits)
        moved_loss = tf.losses.binary_crossentropy(y_true=tf.squeeze(self.moved), y_pred=self.moved_logits, from_logits=True)

        reward_classes = tf.where(self.rewards < 0, 1, tf.where(self.rewards > 0, 2, 0)) # tf.where(self.rewards <= 0, 0, 1)
        num_non_zero_rewards = tf.reduce_sum(tf.cast(reward_classes > 0, tf.int32))

        zero_indices = tf.where(tf.equal(reward_classes, 0))

        shuffled_zero_indices = tf.random.shuffle(zero_indices)[:num_non_zero_rewards]

        non_zero_indices = tf.where(reward_classes > 0)

        all_indices = tf.concat([shuffled_zero_indices, non_zero_indices], axis=0)

        labels = tf.gather_nd(reward_classes, all_indices)
        values = tf.gather_nd(self.reward_logits, all_indices)

        #reward_loss = tf.losses.binary_crossentropy(y_true=tf.squeeze(labels), y_pred=tf.squeeze(values), from_logits=True)
        reward_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(labels), logits=values)
        reward_loss = tf.where(tf.math.is_nan(reward_loss), tf.zeros_like(reward_loss), reward_loss)

        self.moved_loss_mean = tf.reduce_mean(moved_loss)
        self.moved_loss_max = tf.reduce_max(moved_loss)

        self.map_loss_mean = tf.reduce_mean(map_loss)
        self.map_loss_max = tf.reduce_max(map_loss)

        self.reward_loss_mean = tf.reduce_mean(reward_loss)
        self.reward_loss_max = tf.reduce_max(reward_loss)

        prediction_loss = self.moved_loss_mean + self.map_loss_mean + self.reward_loss_mean

        return policy_loss + prediction_loss * 0.33

    def metrics(self):

        return {
            "map_loss_max"   : self.map_loss_max,
            "map_loss_mean": self.map_loss_mean,

            "moved_loss_max" : self.moved_loss_max,
            "moved_loss_mean": self.moved_loss_mean,

            "reward_loss_max" : self.reward_loss_max,
            "reward_loss_mean": self.reward_loss_mean,
        }




