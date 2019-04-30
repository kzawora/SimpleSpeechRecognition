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
net_gru = tflearn.input_data([None, gv.width, gv.height])
net_gru = tflearn.gru(net_gru, 128, dropout=0.8)
net_gru = tflearn.fully_connected(net_gru, gv.classes, activation='softmax')
net_gru = tflearn.regression(net_gru, optimizer='adam', learning_rate=gv.learning_rate, loss='categorical_crossentropy')

# Training
model_gru = tflearn.DNN(net_gru, tensorboard_verbose=gv.tensorboard_verbosity)

for times in range(gv.training_iters):
    X, Y, Z = next(training_batch)
    trainX, trainY = X, Y
    model_gru.fit(trainX, trainY, n_epoch=gv.epochs, validation_set=(testX, testY),
                  validation_batch_size=gv.validation_batch_size,
                  show_metric=True,
                  batch_size=gv.training_batch_size, run_id='gru')
    if times % 10 == 0:
        model_gru.save(gv.models_path + "tflearn.gru.model.checkpoint")

model_gru.save(gv.models_path + 'tflearn.gru.model')
