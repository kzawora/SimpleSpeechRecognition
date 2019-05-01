# SimpleSpeechRecognition

Overview
============
A simple speech recognition application for classifying spoken digits. Mostly based on [pannous/tensorflow-speech-recognition](https://github.com/pannous/tensorflow-speech-recognition).
Includes scripts for training and inference (which is also used for testing/benchmarking).

Models
============
Already trained models are included in `models/` directory.
Models were trained on following `global_variables.py` config:
```py
learning_rate = 0.001
width = 20
height = 10 
classes = 10
training_iters = 200 
training_batch_size = 256
validation_batch_size = 1024
epochs = 20
```
Therefore the training took 4000 steps (`training_iters * epochs`), resulting in following accuracy:

|    type   | Vanilla RNN |  LSTM  |   GRU  |
|:---------:|:-----------:|:------:|:------:|
| accuracy* |    35.88%   | 91.97% | 95.31% |

*accuracy was calculated using `inference.py` script on 6400 samples

Logs
============
TensorBoard logs from training included models are also included in this repo, just in case.

Dependencies
============
Dependencies are included in `requirements.txt` (with tensorflow-gpu by default)
Project was developed and runs successfully on tensorflow-gpu 1.13.0 with CUDA 10.0 and Python 3.7.3 on Manjaro Budgie 18.0.4 and following dependencies versions:
* numpy>=1.16.3
* librosa>=0.6.3
* tensorflow-gpu>=1.13.0
* tensorboard>=1.13.1
* tflearn>=0.3.2
Older configurations are untested.
