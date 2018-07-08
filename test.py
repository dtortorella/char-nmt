import sys
import numpy as np
import math
import keras
import tensorflow as tf

import data
import model
import util

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
weights_file = sys.argv[8]
decoding_is_greedy = len(sys.argv) < 10 or sys.argv[9] == 'greedy'

source_chars = data.load_chars(source_language)
target_chars = data.load_chars(target_language)

source_data = data.load_source(source_file, num_sentences, source_max_len, source_chars)

test_model = model.next_character_model(source_max_len, len(source_chars), len(target_chars))
test_model.load_weights(weights_file, by_name=True)

def greedy_decode(source_sentence):
    decoded_target_sentence = [target_chars['\t']]
    while len(decoded_target_sentence) <= target_max_len+1:
        distribution = test_model.predict([source_sentence, np.array([decoded_target_sentence])])
        next_char = distribution.argmax()
        decoded_target_sentence.append(next_char)
        if next_char == target_chars['\n']:
            break
    return util.decode_from_char_map(decoded_target_sentence, target_chars)

def beam_decode(source_sentence, k, return_k=False):
    sentences = [([target_chars['\t']], 0)]
    sentence_lenght = 0
    while sentence_lenght <= target_max_len:
        candidates = []
        for sentence, score in sentences:
            if sentence[-1] == target_chars['\n']:
                candidates.append((sentence, score))
                continue
            distribution = test_model.predict([source_sentence, np.array([sentence])])
            for i in range(0, len(target_chars)):
                candidates.append((sentence + [i], score + -math.log(distribution[0, i])))
        sorted_candidates = sorted(candidates, key=lambda candidate: candidate[1])
        sentences = sorted_candidates[:k]
        sentence_lenght += 1
    if return_k:
        return [(util.decode_from_char_map(sentence, target_chars), score) for sentence, score in sentences]
    else:
        return (util.decode_from_char_map(sentences[0][0], target_chars), sentences[0][1])

target_sentences = []

for i in range(0, num_sentences):
    if decoding_is_greedy:
        target_sentence = greedy_decode(source_data[i:i+1, :])
    else:
        target_sentence, score = beam_decode(source_data[i:i+1, :], 10)
    target_sentences.append(target_sentence)
    print(target_sentence, flush=True)

with open(target_file, 'r', encoding='utf-8') as f:
    reference_sentences = f.readlines()

bleu_score = util.evaluate_corpus(reference_sentences, target_sentences, target_language)
print(bleu_score, file=sys.stderr)
