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
net = tflearn.input_data([None, gv.width, gv.height])
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
              batch_size=gv.training_batch_size)
    if times % 10 == 0:
        model.save("tflearn.lstm.model.checkpoint")

model.save(gv.working_directory + 'tflearn.lstm.model')
