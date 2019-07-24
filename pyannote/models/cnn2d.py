from typing import List

import torch
import torch.nn as nn
from torch import FloatTensor


class GatedCNN(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels * 2,
                              kernel_size=(3, 3),
                              stride=1,
                              padding=1,
                              bias=True)
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels * 2)
        self.sig = nn.Sigmoid()

    def forward(self, x: FloatTensor):
        x = self.conv(x)
        x = self.batchnorm(x)
        A, B = torch.split(x, self.out_channels, dim=1)
        x = self.sig(A) * B

        return x


class PooledGCNNBlock(nn.Module):

    def __init__(self, dim_channels: int,
                 freq_pooling: int,
                 input_channel_dim: int = None):
        super().__init__()
        if input_channel_dim is None:
            first_conv = GatedCNN(dim_channels, dim_channels)
        else:
            first_conv = GatedCNN(input_channel_dim, dim_channels)
        second_conv = GatedCNN(dim_channels, dim_channels)
        self.gated_cnns = nn.Sequential(first_conv, second_conv)
        self.pooling = nn.MaxPool2d(kernel_size=(1, freq_pooling))

    def forward(self, x: FloatTensor):
        x = self.gated_cnns(x)
        x = self.pooling(x)

        return x


class GatedBiGRU(nn.Module):

    def __init__(self, input_size: int,
                 hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru_sig, self.gru_lin = (nn.GRU(input_size=input_size,
                                             hidden_size=hidden_size,
                                             bidirectional=True,
                                             batch_first=True) for _ in range(2))
        self.sig = nn.Sigmoid()

    def forward(self, x: FloatTensor):
        # no need to initialize the hidden vectors for the GRU, if not specified they
        # are intialized by default
        output_sig, _ = self.gru_sig(x)
        avg_output_sig = (output_sig[:, :, self.hidden_size:]
                          + output_sig[:, :, :self.hidden_size])
        output_lin, _ = self.gru_lin(x)
        avg_output_lin = (output_lin[:, :, self.hidden_size:]
                          + output_lin[:, :, :self.hidden_size])
        output = self.sig(avg_output_sig) * avg_output_lin

        return output


class DeviNet(nn.Module):
    """2D Gated convolutional blocks, followed by a gated Bi-GRU
    and a fully connected layer (for classification)

        Parameters
        ----------
        conv_blocks : `int`
            Number of gated convolutional blocks.
        conv_channels : `int`
            Number of channels in the convolutional blocks
        layers_pooling : `List[int]`
            Pooling kernel size for the frequency dimension on each convolutional block.
            WARNING: the product(layers_pooling) * final_pooling should be equal to the
            number of filter-banks in the input tensor.
        gru_cells : `int`
            Number of GRU (hidden) cells in the gated bi-GRU layer.
        linear_layers: `List[int]`
            Hidden dimensions for each linear layer of the fully connected layer.
        n_classes: `int`
            Number of output classes, then used for detection.

        Usage
        -----
        Same as `torch.nn.Conv1d`

        Reference
        ---------
        Yong Xu, Qiuqiang Kong, Wenwu Wang, Mark D. Plumbley.
        "Large-scale weakly supervised audio classification using
        gated convolutional neural network". DCASE 2017. https://arxiv.org/abs/1710.00343
        """

    def __init__(self, conv_blocks: int = 3,
                 conv_channels: int = 128,
                 layers_pooling: List[int] = None,
                 final_pooling: int = 4,
                 gru_cells: int = 128,
                 linear_layers: List[int] = None,
                 n_classes: int = 4):
        super().__init__()
        self.total_freq_pooling = torch.tensor(layers_pooling).prod() * final_pooling
        self.gru_cells = gru_cells
        gcnns_list = [PooledGCNNBlock(conv_channels, layers_pooling[0], input_channel_dim=1)]
        for i in range(1, conv_blocks):
            gcnns_list.append(PooledGCNNBlock(conv_channels, layers_pooling[i]))
        self.pooled_gcnns = nn.Sequential(*gcnns_list)
        self.final_conv = nn.Conv2d(in_channels=conv_channels,
                                    out_channels=conv_channels * 2,
                                    kernel_size=(3, 3),
                                    stride=2,
                                    padding=1,
                                    bias=True)
        self.final_pool = nn.MaxPool2d(kernel_size=(1, final_pooling))
        self.gated_bigru = GatedBiGRU(conv_channels * 2, gru_cells)

        fc_layers = []
        input_dim = conv_channels
        for hidden_size in linear_layers:
            fc_layers.append(nn.Linear(input_dim, hidden_size,bias=True))
            fc_layers.append(nn.Tanh())
            input_dim = hidden_size
        final_layer = nn.Linear(input_dim, n_classes,
                                bias=True)
        self.fc_layers = nn.Sequential(*(fc_layers + [final_layer]))
        self.final_activation = nn.Sigmoid()

    def forward(self, x: FloatTensor):

        if x.size(2) != self.total_freq_pooling:
            raise ValueError("The frequency dimension (dim %i) is of size %i, not matching "
                             "the total pooling dimensionality reduction %i"
                             % (2, x.size(2), self.total_freq_pooling))

        x = x.unsqueeze(1)  # adding fake channel dimension

        # going through all the convolutions
        x = self.pooled_gcnns(x)
        x = self.final_conv(x)
        x = self.final_pool(x)

        # reducing the "empty" frequency dimension
        x = x.squeeze(3)
        # switching from (N,C,T) to (N,T,C) in preparation for the RNN layer
        x = x.transpose(1, 2).contiguous()

        # going through the gater BiGRU
        x = self.gated_bigru(x)

        # fully connected layers
        x = self.fc_layers(x)

        # sigmoid activation
        x = self.final_activation(x)

        return x