import numpy as np
import util

START_SEQUENCE = '\t'
END_SEQUENCE = '\n'

def load_chars(lang):
    with open("./data/chars." + lang, 'r', encoding='utf-8') as f:
        chars = f.read()
    return util.map_characters_to_integers(chars)

def load_source(file_path, num_lines, max_line_length, chars):
    source = np.empty([num_lines, max_line_length+1], dtype=np.int32)
    with open(file_path, 'r', encoding='utf-8') as f:
        line_index = 0
        while True:
            line = f.readline()
            if not line:
                break
            line_len = len(line)
            line = util.encode_for_embedding(line,chars)
            for i in range(0, line_len):
                source[line_index, i] = line[i]
            for i in range(line_len, max_line_length + 1):
                source[line_index, i] = chars[END_SEQUENCE]
            line_index += 1
    return source

def load_target(file_path, num_lines, max_line_length, chars):
    target = np.empty([num_lines, max_line_length+2], dtype=np.int32)
    with open(file_path, 'r', encoding='utf-8') as f:
        line_index = 0
        while True:
            line = f.readline()
            if not line:
                break
            line_len = len(line)
            line = util.encode_for_embedding(line,chars)
            target[line_index, 0] = chars[START_SEQUENCE]
            for i in range(0, line_len):
                target[line_index, i+1] = line[i]
            for i in range(line_len + 1, max_line_length + 2):
                target[line_index, i] = chars[END_SEQUENCE]
            line_index += 1
    return target

def target_to_categorical(target, num_chars):
    cat = np.zeros([target.shape[0], target.shape[1], num_chars])
    for i in range(0, target.shape[0]):
        for j in range(0, target.shape[1]):
            cat[i, j, target[i,j]] = 1
    return cat