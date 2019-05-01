#!/bin/bash
rm -rf /tmp/tflearn_logs/
/home/konrad/anaconda3/envs/myspeech/bin/python /home/konrad/dev/repo/training.py gru
/home/konrad/anaconda3/envs/myspeech/bin/python /home/konrad/dev/repo/training.py lstm
/home/konrad/anaconda3/envs/myspeech/bin/python /home/konrad/dev/repo/training.py rnn

