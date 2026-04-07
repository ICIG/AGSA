#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import abc

import torch
import torch.nn as nn
from torchsummary import summary

from av_nav.common.utils import CategoricalNet
from av_nav.rl.models.rnn_state_encoder import RNNStateEncoder
from av_nav.rl.models.visual_cnn import VisualCNN
from av_nav.rl.models.audio_cnn import AudioCNN
import torch.nn.functional as F
import math
from types import SimpleNamespace

attn_config = SimpleNamespace(
    hidden_size=768,
    num_attention_heads=12,
    attention_probs_dropout_prob=0.1,
    hidden_dropout_prob=0.1
)

DUAL_GOAL_DELIMITER = ','

class Attention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (self.hidden_size, self.num_attention_heads))

        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if ctx_dim is None:
            ctx_dim = self.hidden_size

        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(self.attention_probs_dropout_prob)

        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.norm = nn.LayerNorm(self.hidden_size)


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        output = self.out_proj(context_layer)
        output = self.norm(output + hidden_states)  # Residual connection + LayerNorm
        return output, attention_scores
    

class MultiLayerCrossModalEncoder(nn.Module):
    def __init__(self, config, num_layers=8):
        super().__init__()
        self.layers = nn.ModuleList([Attention(config) for _ in range(num_layers)])

    def forward(self, query, context, attention_mask=None):
        x = query
        for layer in self.layers:
            x, _ = layer(x, context, attention_mask)
        return x

    

class Policy(nn.Module):
    def __init__(self, net, dim_actions):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )
        self.critic = CriticHead(self.net.output_size)

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        # print('Features: ', features.cpu().numpy())
        distribution = self.action_distribution(features)
        # print('Distribution: ', distribution.logits.cpu().numpy())
        value = self.critic(features)
        # print('Value: ', value.item())

        if deterministic:
            action = distribution.mode()
            # print('Deterministic action: ', action.item())
        else:
            action = distribution.sample()
            # print('Sample action: ', action.item())

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class PointNavBaselinePolicy(Policy):
    def __init__(
        self,
        observation_space,
        action_space,
        goal_sensor_uuid,
        hidden_size=512,
        extra_rgb=False
    ):
        super().__init__(
            PointNavBaselineNet(
                observation_space=observation_space,
                hidden_size=hidden_size,
                goal_sensor_uuid=goal_sensor_uuid,
                extra_rgb=extra_rgb
            ),
            action_space.n,
        )


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass


class PointNavBaselineNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self, observation_space, hidden_size, goal_sensor_uuid, extra_rgb=False,config=None):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid
        self._hidden_size = hidden_size
        self._audiogoal = False
        self._pointgoal = False
        self._n_pointgoal = 0


        self.cross_modal_encoder = MultiLayerCrossModalEncoder(config or attn_config, num_layers=8)
        self.instr_aug_linear = nn.Linear((config or attn_config).hidden_size, 1)
        self.instr_ori_linear = nn.Linear((config or attn_config).hidden_size, 1)
        self.instr_sigmoid = nn.Sigmoid()

        if DUAL_GOAL_DELIMITER in self.goal_sensor_uuid:
            goal1_uuid, goal2_uuid = self.goal_sensor_uuid.split(DUAL_GOAL_DELIMITER)
            self._audiogoal = self._pointgoal = True
            self._n_pointgoal = observation_space.spaces[goal1_uuid].shape[0]
        else:
            if 'pointgoal_with_gps_compass' == self.goal_sensor_uuid:
                self._pointgoal = True
                self._n_pointgoal = observation_space.spaces[self.goal_sensor_uuid].shape[0]
            else:
                self._audiogoal = True

        self.visual_encoder = VisualCNN(observation_space, hidden_size, extra_rgb)
        if self._audiogoal:
            if 'audiogoal' in self.goal_sensor_uuid:
                audiogoal_sensor = 'audiogoal'
            elif 'spectrogram' in self.goal_sensor_uuid:
                audiogoal_sensor = 'spectrogram'
            self.audio_encoder = AudioCNN(observation_space, hidden_size, audiogoal_sensor)

        #rnn_input_size = (0 if self.is_blind else self._hidden_size) + \
                         #(self._n_pointgoal if self._pointgoal else 0) + (self._hidden_size if self._audiogoal else 0)
        self.fusion_feat_dim = 768
        self.goal_feat_dim = self._n_pointgoal if self._pointgoal else 0
        rnn_input_size = self.fusion_feat_dim + self.goal_feat_dim
        
        self.state_encoder = RNNStateEncoder(rnn_input_size, self._hidden_size)

        if 'rgb' in observation_space.spaces and not extra_rgb:
            rgb_shape = observation_space.spaces['rgb'].shape
            summary(self.visual_encoder.cnn, (rgb_shape[2], rgb_shape[0], rgb_shape[1]), device='cpu')
        if 'depth' in observation_space.spaces:
            depth_shape = observation_space.spaces['depth'].shape
            summary(self.visual_encoder.cnn, (depth_shape[2], depth_shape[0], depth_shape[1]), device='cpu')
        if self._audiogoal:
            audio_shape = observation_space.spaces[audiogoal_sensor].shape
            summary(self.audio_encoder.cnn, (audio_shape[2], audio_shape[0], audio_shape[1]), device='cpu')

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        x = []

        if self._pointgoal:
            goal_feat = observations[self.goal_sensor_uuid.split(DUAL_GOAL_DELIMITER)[0]]
            x.append(goal_feat)
        else:
            goal_feat = None

        audio_feat = self.audio_encoder(observations)   # [B, D]
        visual_feat = self.visual_encoder(observations) if not self.is_blind else None

        # === 构建 context ===
        if visual_feat is not None:
            x.append(visual_feat)
            context = torch.cat(x, dim=1).unsqueeze(1)  # [B, 1, D_total]
            if context.size(-1) != self._hidden_size:
                self.context_proj = getattr(self, 'context_proj', nn.Linear(context.size(-1), self._hidden_size).to(context.device))
                context = self.context_proj(context)
        else:
            context = audio_feat.unsqueeze(1)  # [B, 1, D]

        attn_hidden_size = self.cross_modal_encoder.layers[0].hidden_size

        if audio_feat.size(-1) != attn_hidden_size:
            if not hasattr(self, 'audio_proj'):
                self.audio_proj = nn.Linear(audio_feat.size(-1), attn_hidden_size).to(audio_feat.device)
            audio_feat = self.audio_proj(audio_feat)

        audio_feat_exp = audio_feat.unsqueeze(1)

        if context.size(-1) != attn_hidden_size:
            if not hasattr(self, 'context_proj'):
                self.context_proj = nn.Linear(context.size(-1), attn_hidden_size).to(context.device)
            context = self.context_proj(context)

        audio_aug = self.cross_modal_encoder(audio_feat_exp, context)
        audio_aug = audio_aug.squeeze(1)  # [B, D]
        audio_aug = audio_aug.squeeze(1)  # [B, D]
        
        aug_linear_weight = self.instr_aug_linear(audio_aug)
        ori_linear_weight = self.instr_ori_linear(audio_feat)
        aug_weight = self.instr_sigmoid(aug_linear_weight + ori_linear_weight)
        
        fusion_feat = aug_weight * audio_aug + (1 - aug_weight) * audio_feat

        if goal_feat is not None:
            rnn_input = torch.cat([goal_feat, fusion_feat], dim=1)
        else:
            rnn_input = fusion_feat

        x2, rnn_hidden_states1 = self.state_encoder(rnn_input, rnn_hidden_states, masks)

        assert not torch.isnan(x2).any().item()
        return x2, rnn_hidden_states1
