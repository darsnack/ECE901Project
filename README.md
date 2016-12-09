# ECE901Project
A repository containing code and reports for UW-Madison ECE901: Large-Scale Machine Learning

## Setup
Make sure you are running Python 3.5.

Also, run the following commands in conda environment to update TF to the latest version.
```shell
(tensorflow) $ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.0rc0-py3-none-any.whl
(tensorflow) $ pip install --ignore-installed --upgrade $TF_BINARY_URL
```

## Structure
Hardware
 - _VGGNet-16_: Contains the Vivado project for FPGA implementation.

LaTeXStyleFiles: Contains LaTeX style files that may need to be installed on your system to compile LaTeX source.

PaperPresentation: Contains LaTeX source for paper presentation assignment.

ProjectProposal: Contains LaTeX source for project proposal assignment.

ProposalPresentation: Contains LaTeX source for project proposal presentation.

Tensorflow
 - _TFMechanics101Tutorial_: Contains source code for TF tutorial (https://www.tensorflow.org/versions/master/tutorials/mnist/tf/index.html)
  - _fully_connected_feed.py_: Run this using `python fully_connected_feed.py` to train the network.
  - _input_data.py_: Just for reference. The training code pulls this file in via `import`.
  - _mnist.py_: Just for reference. The training code pulls this file in via `import`.
 - _TFCNNTutorial_: Contains source code for TF CNN tutorial (https://www.tensorflow.org/versions/master/tutorials/deep_cnn/index.html)
  - _cifar10.py_: Just for reference. The training code pulls this file in via `import`.
  - _cifar10_input.py_: Just for reference. The model code pulls this file in via `import`.
  - _cifar10_train.py_: Run this using `python cifar10_train.py` to train the network.
 - _TwoLayerCNN_: Contains source code for CPU implementation of custom two layer CNN.
  - _model.py_: Contains the model related functions like `inference()`, `loss()`, and `train()`.
