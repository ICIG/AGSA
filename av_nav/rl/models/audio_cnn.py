import numpy as np
import torch
import torch.nn as nn
from av_nav.common.utils import Flatten


class CrossChannelAttention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(CrossChannelAttention, self).__init__()
        self.input_half = input_dim // 2
        self.query_proj = nn.Conv2d(self.input_half, attention_dim, kernel_size=1)
        self.key_proj = nn.Conv2d(self.input_half, attention_dim, kernel_size=1)
        self.value_proj = nn.Conv2d(self.input_half, attention_dim, kernel_size=1)
        self.output_proj = nn.Conv2d(attention_dim, self.input_half, kernel_size=1)

    def attend(self, query_source, key_value_source):
        query = self.query_proj(query_source)
        key = self.key_proj(key_value_source)
        value = self.value_proj(key_value_source)

        B, C, H, W = query.size()
        query = query.view(B, C, -1)         # [B, C, HW]
        key = key.view(B, C, -1)             # [B, C, HW]
        value = value.view(B, C, -1)         # [B, C, HW]

        attention_scores = torch.bmm(query.transpose(1, 2), key)  # [B, HW, HW]
        attention_scores = attention_scores / (C ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)

        attended = torch.bmm(attention_weights, value.transpose(1, 2))  # [B, HW, C]
        attended = attended.transpose(1, 2).view(B, C, H, W)
        return self.output_proj(attended)

    def forward(self, x):
        c_half = x.size(1) // 2
        x_left = x[:, :c_half, :, :]
        x_right = x[:, c_half:, :, :]

        # Cross attention: each side attends to the other
        left_out = self.attend(x_left, x_right) + x_left  # Residual
        right_out = self.attend(x_right, x_left) + x_right  # Residual

        output = torch.cat([left_out, right_out], dim=1)
        return output

class AudioCNN(nn.Module):
    def __init__(self, observation_space, output_size, audiogoal_sensor):
        super().__init__()
        self._n_input_audio = observation_space.spaces[audiogoal_sensor].shape[2]
        self._audiogoal_sensor = audiogoal_sensor

        cnn_dims = np.array(
            observation_space.spaces[audiogoal_sensor].shape[:2], dtype=np.float32
        )

        if cnn_dims[0] < 30 or cnn_dims[1] < 30:
            self._cnn_layers_kernel_size = [(5, 5), (3, 3), (3, 3)]
            self._cnn_layers_stride = [(2, 2), (2, 2), (1, 1)]
        else:
            self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]
            self._cnn_layers_stride = [(4, 4), (2, 2), (1, 1)]

        for kernel_size, stride in zip(
            self._cnn_layers_kernel_size, self._cnn_layers_stride
        ):
            cnn_dims = self._conv_output_dim(
                dimension=cnn_dims,
                padding=np.array([0, 0], dtype=np.float32),
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array(kernel_size, dtype=np.float32),
                stride=np.array(stride, dtype=np.float32),
            )

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=self._n_input_audio,
                out_channels=32,
                kernel_size=self._cnn_layers_kernel_size[0],
                stride=self._cnn_layers_stride[0],
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=self._cnn_layers_kernel_size[1],
                stride=self._cnn_layers_stride[1],
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=self._cnn_layers_kernel_size[2],
                stride=self._cnn_layers_stride[2],
            ),
            nn.ReLU(True),
        )

        self.attn = CrossChannelAttention(input_dim=64, attention_dim=64)

        self.linear_head = nn.Sequential(
            Flatten(),
            nn.Linear(64 * cnn_dims[0] * cnn_dims[1], output_size),  # 保持64通道
            nn.ReLU(True),
        )

        self.layer_init()

    def _conv_output_dim(self, dimension, padding, dilation, kernel_size, stride):
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(
                    np.floor(
                        (
                            (
                                dimension[i]
                                + 2 * padding[i]
                                - dilation[i] * (kernel_size[i] - 1)
                                - 1
                            )
                            / stride[i]
                        )
                        + 1
                    )
                )
            )
        return tuple(out_dimension)

    def layer_init(self):
        for layer in list(self.cnn) + list(self.linear_head):
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, observations):
        audio_observations = observations[self._audiogoal_sensor]
        audio_observations = audio_observations.permute(0, 3, 1, 2)  # [B, C, H, W]

        features = self.cnn(audio_observations)
        features = self.attn(features)
        return self.linear_head(features)

