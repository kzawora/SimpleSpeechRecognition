from __future__ import division, print_function, absolute_import
import tflearn
import batch_generator
import global_variables as gv

# Setting batches
validation_batch = batch_generator.mfcc_batch_generator(gv.validation_batch_size, gv.height)
x, y, z = next(validation_batch)
testX, testY = x, y

training_batch = batch_generator.mfcc_batch_generator(gv.training_batch_size, gv.height, exclude=z)

# Network building
net_rnn = tflearn.input_data([None, gv.width, gv.height])
net_rnn = tflearn.simple_rnn(net_rnn, 128, dropout=0.8)
net_rnn = tflearn.fully_connected(net_rnn, gv.classes, activation='softmax')
net_rnn = tflearn.regression(net_rnn, optimizer='adam', learning_rate=gv.learning_rate, loss='categorical_crossentropy')

# Training
model_rnn = tflearn.DNN(net_rnn, tensorboard_verbose=gv.tensorboard_verbosity)

for times in range(gv.training_iters):
    X, Y, Z = next(training_batch)
    trainX, trainY = X, Y
    model_rnn.fit(trainX, trainY, n_epoch=gv.epochs, validation_set=(testX, testY),
                  validation_batch_size=gv.validation_batch_size,
                  show_metric=True,
                  batch_size=gv.training_batch_size, run_id='rnn')
    if times % 10 == 0:
        model_rnn.save("tflearn.rnn.model.checkpoint")

model_rnn.save(gv.working_directory + 'tflearn.rnn.model')
