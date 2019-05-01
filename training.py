from __future__ import division, print_function, absolute_import
import sys
import tflearn
import batch_generator
import global_variables as gv

if len(sys.argv) < 2 or sys.argv[1] == 'lstm':
    mode = 'lstm'
elif sys.argv[1] == 'gru':
    mode = 'gru'
elif sys.argv[1] == 'rnn':
    mode = 'rnn'
else:
    raise ValueError("Incorrect argument provided.")

# Setting batches
validation_batch = batch_generator.mfcc_batch_generator(gv.validation_batch_size, gv.height)
x, y, z = next(validation_batch)
testX, testY = x, yitem

training_batch = batch_generator.mfcc_batch_generator(gv.training_batch_size, gv.height, exclude=z)

# Network building
net = tflearn.input_data([None, gv.width, gv.height])
if mode == 'gru':
    net = tflearn.gru(net, 128, dropout=0.8)
elif mode == 'rnn':
    net = tflearn.simple_rnn(net, 128, dropout=0.8)
else:
    net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, gv.classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=gv.learning_rate, loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=gv.tensorboard_verbosity)
for times in range(gv.training_iters):
    X, Y, Z = next(training_batch)
    trainX, trainY = X, Y
    model.fit(trainX, trainY, n_epoch=gv.epochs, validation_set=(testX, testY),
              validation_batch_size=gv.validation_batch_size,
              show_metric=True,
              batch_size=gv.training_batch_size, run_id=mode + '_training')
    if times % 10 == 0:
        model.save(gv.models_path + 'tflearn.' + mode + '.model.checkpoint')

model.save(gv.models_path + 'tflearn.' + mode + '.model')
