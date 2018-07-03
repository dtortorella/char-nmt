import keras
from keras import Model
from keras.layers import *

# language sequences constants
SOURCE_SEQ_LEN = 2000
SOURCE_NUM_CHARS = 1000
TARGET_SEQ_LEN = 2000
TARGET_NUM_CHARS = 200

def next_character_model():
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
    context_vector = Dot(axes=(1,1))([attention_weights, source_hidden_states])

    # character generator network
    concatenate_target_and_context = Concatenate()([target_hidden_state, context_vector])
    next_character_distribution = Dense(TARGET_NUM_CHARS, activation='softmax', name='generator_distribution')(concatenate_target_and_context)

    # next character generator model
    return Model([source_input, target_input], next_character_distribution)

def sequence_training_model():
    # source sentence (fixed length) input, embedding, and sequence of hidden states
    source_input = Input(shape=(SOURCE_SEQ_LEN,), dtype='int32')
    source_embedding = Embedding(SOURCE_NUM_CHARS, 64, name='source_embedding')(source_input)
    source_hidden_states = Bidirectional(LSTM(16, return_sequences=True, name='source_hidden_states'))(source_embedding)

    # target sequence (variable length) input, embedding, and hidden state
    target_input = Input(shape=(None,), dtype='int32')
    target_embedding = Embedding(TARGET_NUM_CHARS, 64, name='target_embedding')(target_input)
    target_hidden_state_sequence = LSTM(16, return_sequences=True, name='target_hidden_state')(target_embedding)

    # attention mechanism (Bahdanau et al, 2015)
    target_hidden_state_sequence = Reshape((TARGET_SEQ_LEN, 16))(target_hidden_state_sequence)
    repeat_target_hidden_state_sequence = TimeDistributed(RepeatVector(SOURCE_SEQ_LEN))(target_hidden_state_sequence) # (Ty, Tx, #s)
    source_hidden_states = Reshape((SOURCE_SEQ_LEN, 32))(source_hidden_states)
    repeat_source_hidden_states = TimeDistributed(RepeatVector(TARGET_SEQ_LEN))(source_hidden_states) # (Tx, Ty, #h)

    permuted_repeat_source_hidden_states = Permute((2, 1, 3))(repeat_source_hidden_states) # (Ty, Tx, #h)
    concatenate_source_target_hidden_matrix = Concatenate()([permuted_repeat_source_hidden_states, repeat_target_hidden_state_sequence]) # (Ty, Tx, #h+#s)
    attention_pre_energies = TimeDistributed(TimeDistributed(Dense(16, activation='tanh')), name='attention_pre_energies')(concatenate_source_target_hidden_matrix) # (Ty, Tx, 16)
    attention_energies = TimeDistributed(TimeDistributed(Dense(1, activation='linear')), name='attention_energies')(attention_pre_energies) # (Ty, Tx, 1)
    attention_energies = Reshape((TARGET_SEQ_LEN, SOURCE_SEQ_LEN))(attention_energies) # (Ty, Tx)
    attention_weights_sequence = TimeDistributed(Activation('softmax'))(attention_energies) # (Ty, Tx)
    context_vector_sequence = Dot(axes=(2, 2))([attention_weights_sequence, permuted_repeat_source_hidden_states]) # (Ty, #h)

    # character generator network
    concatenate_target_and_context = Concatenate()([target_hidden_state_sequence, context_vector_sequence])
    next_character_distribution = TimeDistributed(Dense(TARGET_NUM_CHARS, activation='softmax'), name='generator_distribution')(concatenate_target_and_context)

    # next character generator model
    return Model([source_input, target_input], next_character_distribution)
    #next_character_model.compile(optimizer='adam', loss='categorical_crossentropy')
