# pyannote-devinet-model

A CNN-RNN model for speaker detection. Mostly a PyTorch reimplementation of the model 
described in [Large-scale weakly supervised audio classification using gated convolutional neural network](https://arxiv.org/abs/1710.00343), 
and a implemented [here](https://github.com/yongxuUSTC/dcase2017_task4_cvssp) in Keras.

## Installation

This library's been made to be imported and used inside a
 [pyannote-audio](https://github.com/pyannote/pyannote-audio) environment.
 
Once this is done (and the environment activated), you can install it through pip:

``` 
pip install git+git://github.com/jsalt-coml/pyannote-devinet-model.git
``` 

## Pyannote Configuration

To use the model contained in this library, this is the kind of configuration you
should add your `config.yml` file:

```yaml
feature_extraction:
  name: pyannote.models.DeviNet
  params:
    conv_blocks: 3
    conv_channels: 128
    layers_pooling: [2, 2, 2 ,2]
    final_pooling: 4
    dropout: 0.3
    recurrent: [128]
    gated_rnn: True
    linear_layers: [32, 32]
```

This corresponds to the following model structure (Pytorch reprensation):

```
DeviNet(
  (pooled_gcnns): Sequential(
    (0): PooledGCNNBlock(
      (gated_cnns): Sequential(
        (0): GatedCNN(
          (conv): Conv2d(1, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (batchnorm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (sig): Sigmoid()
        )
        (1): GatedCNN(
          (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (batchnorm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (sig): Sigmoid()
        )
      )
      (pooling): MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0, dilation=1, ceil_mode=False)
    )
    (1): PooledGCNNBlock(
      (gated_cnns): Sequential(
        (0): GatedCNN(
          (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (batchnorm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (sig): Sigmoid()
        )
        (1): GatedCNN(
          (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (batchnorm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (sig): Sigmoid()
        )
      )
      (pooling): MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0, dilation=1, ceil_mode=False)
    )
    (2): PooledGCNNBlock(
      (gated_cnns): Sequential(
        (0): GatedCNN(
          (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (batchnorm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (sig): Sigmoid()
        )
        (1): GatedCNN(
          (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (batchnorm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (sig): Sigmoid()
        )
      )
      (pooling): MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0, dilation=1, ceil_mode=False)
    )
  )
  (final_conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (final_pool): MaxPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0, dilation=1, ceil_mode=False)
  (gated_bigru): GatedBiGRU(
    (gru_sig): GRU(256, 128, batch_first=True, bidirectional=True)
    (gru_lin): GRU(256, 128, batch_first=True, bidirectional=True)
    (sig): Sigmoid()
  )
  (fc_layers): Sequential(
    (0): Linear(in_features=128, out_features=32, bias=True)
    (1): Tanh()
    (2): Linear(in_features=32, out_features=32, bias=True)
    (3): Tanh()
    (4): Linear(in_features=32, out_features=4, bias=True)
  )
  (final_activation): Sigmoid()
)
```