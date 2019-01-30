# keras-CycleGAN

This is a keras implementation of [CycleGAN](https://arxiv.org/pdf/1703.10593.pdf). The original PyTorch implementation by the author can be found [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). This implementation is mostly based on theirs.

## Prerequisites

- Python 3
- Tensorflow
- Keras

## Getting Started

There is a sample usage in `CycleGAN/cyclegan.py`. For simplest use, all you need is to create a CycleGAN instance, compile it using optimizing parameters and loss weights, then feed it data and start training. It's almost the same with a typical Keras model.

Please refer to the source code in `CycleGAN/cyclegan.py` if you would like to have a precise control of the model structure and training process.

## TODO-list

- [x] Contrust model
- [ ] Add sample results
- [ ] Refactor to provide simpler usage and control

## Acknowledgements

The code is mainly inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
