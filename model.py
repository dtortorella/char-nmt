import keras
from keras import Model
from keras.layers import *

# model parameters
SOURCE_EMBEDDING = 32
SOURCE_STATE = 64
SOURCE_STATE_HALF = 32
TARGET_EMBEDDING = 32
TARGET_STATE = 64
ATTENTION_ENERGY = 32

def next_character_model(SOURCE_SEQ_LEN, SOURCE_NUM_CHARS, TARGET_NUM_CHARS):
    # source sentence (fixed length) input, embedding, and sequence of hidden states
    source_input = Input(shape=(SOURCE_SEQ_LEN,), dtype='int32')
    source_embedding = Embedding(SOURCE_NUM_CHARS, SOURCE_EMBEDDING, name='source_embedding')(source_input)
    source_hidden_states = Bidirectional(LSTM(SOURCE_STATE_HALF, return_sequences=True, name='source_hidden_states'))(source_embedding)

    # target sequence (variable length) input, embedding, and hidden state
    target_input = Input(shape=(None,), dtype='int32')
    target_embedding = Embedding(TARGET_NUM_CHARS, TARGET_EMBEDDING, name='target_embedding')(target_input)
    target_hidden_state = LSTM(TARGET_STATE, return_sequences=False, name='target_hidden_state')(target_embedding)

    # attention mechanism (Bahdanau et al, 2015)
    repeat_target_hidden_state = RepeatVector(SOURCE_SEQ_LEN)(target_hidden_state)
    concatenate_source_target_hidden = Concatenate()([source_hidden_states, repeat_target_hidden_state])

    attention_pre_energies1 = TimeDistributed(Dense(ATTENTION_ENERGY, activation='tanh'), name='attention_pre_energies1')(concatenate_source_target_hidden)
    attention_energies1 = TimeDistributed(Dense(1, activation='linear'), name='attention_energies1')(attention_pre_energies1)
    attention_energies1 = Reshape((SOURCE_SEQ_LEN,))(attention_energies1)
    attention_weights1 = Activation('softmax')(attention_energies1)
    context_vector1 = Dot(axes=(1,1))([attention_weights1, source_hidden_states])

    attention_pre_energies2 = TimeDistributed(Dense(ATTENTION_ENERGY, activation='tanh'), name='attention_pre_energies2')(concatenate_source_target_hidden)
    attention_energies2 = TimeDistributed(Dense(1, activation='linear'), name='attention_energies2')(attention_pre_energies2)
    attention_energies2 = Reshape((SOURCE_SEQ_LEN,))(attention_energies2)
    attention_weights2 = Activation('softmax')(attention_energies2)
    context_vector2 = Dot(axes=(1,1))([attention_weights2, source_hidden_states])

    attention_pre_energies3 = TimeDistributed(Dense(ATTENTION_ENERGY, activation='tanh'), name='attention_pre_energies3')(concatenate_source_target_hidden)
    attention_energies3 = TimeDistributed(Dense(1, activation='linear'), name='attention_energies3')(attention_pre_energies3)
    attention_energies3 = Reshape((SOURCE_SEQ_LEN,))(attention_energies3)
    attention_weights3 = Activation('softmax')(attention_energies3)
    context_vector3 = Dot(axes=(1,1))([attention_weights3, source_hidden_states])

    # character generator network
    concatenate_target_and_context = Concatenate()([target_hidden_state, context_vector1, context_vector2, context_vector3])
    next_character_distribution = Dense(TARGET_NUM_CHARS, activation='softmax', name='generator_distribution')(concatenate_target_and_context)

    # next character generator model
    return Model([source_input, target_input], next_character_distribution)

def sequence_training_model(SOURCE_SEQ_LEN, SOURCE_NUM_CHARS, TARGET_SEQ_LEN, TARGET_NUM_CHARS):
    # source sentence (fixed length) input, embedding, and sequence of hidden states
    source_input = Input(shape=(SOURCE_SEQ_LEN,), dtype='int32')
    source_embedding = Embedding(SOURCE_NUM_CHARS, SOURCE_EMBEDDING, name='source_embedding')(source_input)
    source_hidden_states = Bidirectional(LSTM(SOURCE_STATE_HALF, return_sequences=True, name='source_hidden_states'))(source_embedding)

    # target sequence (fixed length) input, embedding, and hidden state
    target_input = Input(shape=(TARGET_SEQ_LEN,), dtype='int32')
    target_embedding = Embedding(TARGET_NUM_CHARS, TARGET_EMBEDDING, name='target_embedding')(target_input)
    target_hidden_state_sequence = LSTM(TARGET_STATE, return_sequences=True, name='target_hidden_state')(target_embedding)

    # attention mechanism (Bahdanau et al, 2015)
    target_hidden_state_sequence = Reshape((TARGET_SEQ_LEN, TARGET_STATE))(target_hidden_state_sequence)
    repeat_target_hidden_state_sequence = TimeDistributed(RepeatVector(SOURCE_SEQ_LEN))(target_hidden_state_sequence) # (Ty, Tx, #s)
    source_hidden_states = Reshape((SOURCE_SEQ_LEN, SOURCE_STATE))(source_hidden_states)
    repeat_source_hidden_states = TimeDistributed(RepeatVector(TARGET_SEQ_LEN))(source_hidden_states) # (Tx, Ty, #h)

    permuted_repeat_source_hidden_states = Permute((2, 1, 3))(repeat_source_hidden_states) # (Ty, Tx, #h)
    concatenate_source_target_hidden_matrix = Concatenate()([permuted_repeat_source_hidden_states, repeat_target_hidden_state_sequence]) # (Ty, Tx, #h+#s)
    
    attention_pre_energies1 = TimeDistributed(TimeDistributed(Dense(ATTENTION_ENERGY, activation='tanh')), name='attention_pre_energies1')(concatenate_source_target_hidden_matrix) # (Ty, Tx, 16)
    attention_energies1 = TimeDistributed(TimeDistributed(Dense(1, activation='linear')), name='attention_energies1')(attention_pre_energies1) # (Ty, Tx, 1)
    attention_energies1 = Reshape((TARGET_SEQ_LEN, SOURCE_SEQ_LEN))(attention_energies1) # (Ty, Tx)
    attention_weights_sequence1 = TimeDistributed(Activation('softmax'))(attention_energies1) # (Ty, Tx)
    context_vector_sequence1 = Dot(axes=(2,1))([attention_weights_sequence1, source_hidden_states]) # (Tx, Tx) * (Tx, #h) -> (Ty, #h)

    attention_pre_energies2 = TimeDistributed(TimeDistributed(Dense(ATTENTION_ENERGY, activation='tanh')), name='attention_pre_energies2')(concatenate_source_target_hidden_matrix) # (Ty, Tx, 16)
    attention_energies2 = TimeDistributed(TimeDistributed(Dense(1, activation='linear')), name='attention_energies2')(attention_pre_energies2) # (Ty, Tx, 1)
    attention_energies2 = Reshape((TARGET_SEQ_LEN, SOURCE_SEQ_LEN))(attention_energies2) # (Ty, Tx)
    attention_weights_sequence2 = TimeDistributed(Activation('softmax'))(attention_energies2) # (Ty, Tx)
    context_vector_sequence2 = Dot(axes=(2,1))([attention_weights_sequence2, source_hidden_states]) # (Tx, Tx) * (Tx, #h) -> (Ty, #h)

    attention_pre_energies3 = TimeDistributed(TimeDistributed(Dense(ATTENTION_ENERGY, activation='tanh')), name='attention_pre_energies3')(concatenate_source_target_hidden_matrix) # (Ty, Tx, 16)
    attention_energies3 = TimeDistributed(TimeDistributed(Dense(1, activation='linear')), name='attention_energies3')(attention_pre_energies3) # (Ty, Tx, 1)
    attention_energies3 = Reshape((TARGET_SEQ_LEN, SOURCE_SEQ_LEN))(attention_energies3) # (Ty, Tx)
    attention_weights_sequence3 = TimeDistributed(Activation('softmax'))(attention_energies3) # (Ty, Tx)
    context_vector_sequence3 = Dot(axes=(2,1))([attention_weights_sequence3, source_hidden_states]) # (Tx, Tx) * (Tx, #h) -> (Ty, #h)

    # character generator network
    concatenate_target_and_context = Concatenate()([target_hidden_state_sequence, context_vector_sequence1, context_vector_sequence2, context_vector_sequence3])
    next_character_distribution = TimeDistributed(Dense(TARGET_NUM_CHARS, activation='softmax'), name='generator_distribution')(concatenate_target_and_context)

    # next character generator model
    return Model([source_input, target_input], next_character_distribution)
    #next_character_model.compile(optimizer='adam', loss='categorical_crossentropy')
