import sys
import datetime
import numpy as np
import keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint

import data
import model

# backend configuration
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
keras.backend.tensorflow_backend.set_session(sess)
###

source_language = sys.argv[1]
target_language = sys.argv[2]
source_file = sys.argv[3]
target_file = sys.argv[4]
num_sentences = int(sys.argv[5])
source_max_len = int(sys.argv[6])
target_max_len = int(sys.argv[7])

source_chars = data.load_chars(source_language)
target_chars = data.load_chars(target_language)

source_data = data.load_source(source_file, num_sentences, source_max_len, source_chars)
target_data = data.load_target(target_file, num_sentences, target_max_len, target_chars)

training_model = model.sequence_training_model(source_max_len, len(source_chars), target_max_len, len(target_chars))
training_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint(filepath=source_language+'-'+target_language+'_'+datetime.datetime.now().isoformat()+'_{epoch:03d}_{acc:.4f}.hdf5')

if len(sys.argv) > 8:
    weights_file = sys.argv[8]
    training_model.load_weights(weights_file, by_name=True)

training_model.fit(x=[source_data, target_data[:,:-1]], y=np.expand_dims(target_data[:,1:], axis=-1), batch_size=100, epochs=5, callbacks=[checkpoint])
