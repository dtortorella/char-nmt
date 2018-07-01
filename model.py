import keras
from keras import Model
from keras.layers import *

# language sequences constants
SOURCE_SEQ_LEN = 2000
SOURCE_NUM_CHARS = 1000
TARGET_SEQ_LEN = 2000
TARGET_NUM_CHARS = 200

# source sentence (fixed length) input, embedding, and sequence of hidden states
source_input = Input(shape=(SOURCE_SEQ_LEN,), dtype='int32')
source_embedding = Embedding(SOURCE_NUM_CHARS, 64, name='source_embedding')(source_input)
source_hidden_states = Bidirectional(LSTM(16, return_sequences=True, name='source_hidden_states'))(source_embedding)

# target sequence (variable length) input, embedding, and hidden state
target_input = Input(shape=(None,), dtype='int32')
target_embedding = Embedding(TARGET_NUM_CHARS, 64, name='target_embedding')(target_input)
target_hidden_state = LSTM(16, return_sequences=False, name='target_hidden_state')(target_embedding)

# attention mechanism (Bahdanau et al, 2015)
repeat_target_hidden_state = RepeatVector(SOURCE_SEQ_LEN)(target_hidden_state)
concatenate_source_target_hidden = Concatenate()([source_hidden_states, repeat_target_hidden_state])
attention_pre_energies = TimeDistributed(Dense(16, activation='tanh'), name='attention_pre_energies')(concatenate_source_target_hidden)
attention_energies = TimeDistributed(Dense(1, activation='linear'), name='attention_energies')(attention_pre_energies)
attention_energies = Reshape((SOURCE_SEQ_LEN,))(attention_energies)
attention_weights = Activation('softmax')(attention_energies)
permuted_source_hidden_states = Permute((2,1))(source_hidden_states)
context_vector = Dot(axes=(-1,-1))([attention_weights, permuted_source_hidden_states])

# character generator network
concatenate_target_and_context = Concatenate()([target_hidden_state, context_vector])
next_character_distribution = Dense(TARGET_NUM_CHARS, activation='softmax', name='generator_distribution')(concatenate_target_and_context)

# next character generator model
next_character_model = Model([source_input, target_input], next_character_distribution)
next_character_model.compile(optimizer='adam', loss='categorical_crossentropy')