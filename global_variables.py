# general
learning_rate = 0.001
width = 20  # mfcc features
height = 10  # (max) length of utterance
classes = 10  # digits
tensorboard_verbosity = 3

# training
training_iters = 100  # steps
training_batch_size = 256
validation_batch_size = 1024
epochs = 20

# inference
batch_iterations = 100
inference_batch_size = 64

# paths
working_directory = '/home/konrad/dev/myspeech/'
data_path = working_directory + 'data/'
pcm_path = data_path + 'spoken_numbers_pcm/'
training_path = data_path + 'training/'
validation_path = data_path + 'validation/'
inference_path = data_path + 'inference/'
