import keras
from keras.preprocessing.text import Tokenizer
import numpy as np

#path names to data files
path_dev_ar = '/media/storage/university/sp18/hlt/project/data/testsets/devset/UNv1.0.devset.ar'
path_dev_en = '/media/storage/university/sp18/hlt/project/data/testsets/devset/UNv1.0.devset.en'
path_dev_es = '/media/storage/university/sp18/hlt/project/data/testsets/devset/UNv1.0.devset.es'
path_dev_fr = '/media/storage/university/sp18/hlt/project/data/testsets/devset/UNv1.0.devset.fr'
path_dev_ru = '/media/storage/university/sp18/hlt/project/data/testsets/devset/UNv1.0.devset.ru'
path_dev_zh = '/media/storage/university/sp18/hlt/project/data/testsets/devset/UNv1.0.devset.zh'
path_test_ar = '/media/storage/university/sp18/hlt/project/data/testsets/testset/UNv1.0.testset.ar'
path_test_en = '/media/storage/university/sp18/hlt/project/data/testsets/testset/UNv1.0.testset.en'
path_test_es = '/media/storage/university/sp18/hlt/project/data/testsets/testset/UNv1.0.testset.es'
path_test_fr = '/media/storage/university/sp18/hlt/project/data/testsets/testset/UNv1.0.testset.fr'
path_test_ru = '/media/storage/university/sp18/hlt/project/data/testsets/testset/UNv1.0.testset.ru'
path_test_zh = '/media/storage/university/sp18/hlt/project/data/testsets/testset/UNv1.0.testset.zh'

def preprocess(source_file, target_file):
    
    #Taken from the 10-minute tutorial

    # Vectorize the data.
    source_texts = []
    target_texts = []
    source_characters = set()
    target_characters = set()

    with open(source_file, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
        for line in lines[: len(lines) - 1]:
            source_texts.append(line)
            for char in line:
                if char not in source_characters:
                    source_characters.add(char)


    with open(target_file, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
        for line in lines[: len(lines) - 1]:
            # We use "tab" as the "start sequence" character
            # for the targets, and "\n" as "end sequence" character.
            line_edited = '\t' + line + '\n'
            target_texts.append(line_edited)
            for char in line_edited:
                if char not in target_characters:
                    target_characters.add(char)


    source_characters = sorted(list(source_characters))
    target_characters = sorted(list(target_characters))

    # language sequences constants
    SOURCE_SEQ_LEN = max([len(txt) for txt in source_texts])
    SOURCE_NUM_CHARS = len(source_characters)
    TARGET_SEQ_LEN = max([len(txt) for txt in target_texts])
    TARGET_NUM_CHARS = len(target_characters)

    source_token_index = dict(
        [(char, i) for i, char in enumerate(source_characters)])
    target_token_index = dict(
        [(char, i) for i, char in enumerate(target_characters)])

    encoder_input_data = np.zeros(
        (len(source_texts), SOURCE_SEQ_LEN, SOURCE_NUM_CHARS),
        dtype='float32')
    decoder_input_data = np.zeros(
        (len(source_texts), TARGET_SEQ_LEN, TARGET_NUM_CHARS),
        dtype='float32')
    decoder_target_data = np.zeros(
        (len(source_texts), TARGET_SEQ_LEN, TARGET_NUM_CHARS),
        dtype='float32')

    for i, (source_text, target_text) in enumerate(zip(source_texts, target_texts)):
        for t, char in enumerate(source_text):
            encoder_input_data[i, t, source_token_index[char]] = 1.
        for t, char in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, target_token_index[char]] = 1.
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.

    return encoder_input_data, decoder_input_data, decoder_target_data

def keras_preprocess(source_file, target_file):

    """
    Tried using keras, but I don't think it makes sense here.
    Maybe we could modify this if we really wanted to.
    """

    # Vectorize the data.
    source_texts = []
    target_texts = []
    source_characters = set()
    target_characters = set()

    with open(source_file, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
        for line in lines[: len(lines) - 1]:
            source_texts.append(line)
            for char in line:
                if char not in source_characters:
                    source_characters.add(char)


    with open(target_file, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
        for line in lines[: len(lines) - 1]:
            # We use "tab" as the "start sequence" character
            # for the targets, and "\n" as "end sequence" character.
            line_edited = '\t' + line + '\n'
            target_texts.append(line_edited)
            for char in line_edited:
                if char not in target_characters:
                    target_characters.add(char)

    source_characters = sorted(list(source_characters))
    target_characters = sorted(list(target_characters))

    # language sequences constants
    SOURCE_SEQ_LEN = max([len(txt) for txt in source_texts])
    SOURCE_NUM_CHARS = len(source_characters)
    TARGET_SEQ_LEN = max([len(txt) for txt in target_texts])
    TARGET_NUM_CHARS = len(target_characters)

    source_tokenizer = Tokenizer(char_level=True)
    source_tokenizer.fit_on_texts(source_texts)
    #print(source_tokenizer.word_counts) # number of times each character appears in file
    #print(source_tokenizer.document_count) # number of lines
    #print(source_tokenizer.word_index)
    #print(source_tokenizer.word_docs)
    source_tokens = source_tokenizer.texts_to_matrix(source_texts, mode='count')

    target_tokenizer = Tokenizer(char_level=True)
    target_tokenizer.fit_on_texts(target_texts)
    target_tokens = target_tokenizer.texts_to_matrix(target_texts, mode='count')

    return source_tokens, target_tokens

