from __future__ import division, print_function, absolute_import
import sys
import tflearn
import batch_generator
import numpy as np
import global_variables as gv

if len(sys.argv) < 2 or sys.argv[1] == 'lstm':
    mode = 'lstm'
elif sys.argv[1] == 'gru':
    mode = 'gru'
elif sys.argv[1] == 'rnn':
    mode = 'rnn'
else:
    raise ValueError("Incorrect argument provided.")

batch = batch_generator.mfcc_batch_generator(gv.inference_batch_size, gv.height)

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
model.load(gv.models_path + 'tflearn.' + mode + '.model')

total_accuracy = 0
for i in range(gv.batch_iterations):
    X, Y, Z = next(batch)
    y = model.predict(X)
    print("PROCESSING BATCH", i)
    batch_accuracy = 0
    for idx, b in enumerate(Y):
        print("Predicting file:", Z[idx])
        b = b.tolist()
        b = b.index(max(b))
        a = np.array(y[idx]).tolist()
        result = a.index(max(a))
        print("result:\t", result)
        print("actual:\t", b)
        print("Printing probabilities:")
        for cnt, elem in enumerate(a):
            print("\t", cnt, ":\t", "%.2f%%" % (100 * elem))
            cnt += 1
        if b != result:
            print("INCORRECT result with %.2f%%" % (100 * max(a)), "probability!")
        else:
            print("CORRECT result with %.2f%%" % (100 * max(a)), "probability!")
        if result == b:
            batch_accuracy += 1
        print("")
    print("Batch accuracy: %.2f%%" % ((batch_accuracy / gv.inference_batch_size) * 100))
    total_accuracy += batch_accuracy

print("Total files predicted:", gv.inference_batch_size * gv.batch_iterations)
print("Total files predicted correctly:", total_accuracy)
print("Total accuracy: %.2f%%" % ((total_accuracy / (gv.inference_batch_size * gv.batch_iterations)) * 100))
