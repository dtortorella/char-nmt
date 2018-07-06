import nltk
import jieba

def tokenize(sentence, language):
    """Splits a sentence into tokens.

    It uses the MOSES tokenizer for all languages except Chinese.
    For Chinese tokenization, Jieba is used with the default dictionary.

    Args:
        sentence (str): Sentence to tokenize.
        language (str): Code of the sentence language (e.g., 'en', 'zh', ...).

    Returns:
        List[str]: List of tokens.
    """
    if language == 'zh':
        return jieba.cut(sentence)
    else:
        tokenizer = nltk.tokenize.moses.MosesTokenizer(language)
        return tokenizer.tokenize(sentence)

def evaluate_sentence(reference, candidate, language):
    """Evaluates the BLEU score for a single sentence translation.

    Args:
        reference (str): Reference translation (single sentence).
        candidate (str): Candidate translation.
        language (str): Code of the sentence language.

    Returns:
        float: BLEU score.
    """
    tokenized_reference = tokenize(reference, language)
    tokenized_candidate = tokenize(candidate, language)
    return nltk.translate.bleu_score.sentence_bleu([tokenized_reference], tokenized_candidate)

def evaluate_corpus(references, candidates, language):
    """Evaluates the BLEU score for a corpus of sentences.

    Args:
        references (List[str]): Corpus of reference translations (a single reference for each sentence).
        candidate (List[str]): Corpus of candidate translations.
        language (str): Code of the sentence language.

    Returns:
        float: BLEU score averaged on the corpus.
    """
    tokenized_references = [[tokenize(reference, language)] for reference in references]
    tokenized_candidates = [tokenize(candidate, language) for candidate in candidates]
    return nltk.translate.bleu_score.corpus_bleu(tokenized_references, tokenized_candidates)

def map_characters_to_integers(characters):
    """Builds a character to integer map according to a string of unique characters.

    Args:
        characters (str): String containing unique characters.
    
    Returns:
        Map[char->int]: Map of characters to integer indexes.
    """
    return {characters[i]: i for i in range(len(characters))}

def encode_for_embedding(sentence, characters):
    """Encodes a string into an integer sequece.

    Args:
        sentence (str): Sentence string.
        characters (Map[char->int]): Map of characters to integer indexes.
    
    Returns:
        List[int]: List of integer indexes, according to characters map.
    """
    return [characters[char] for char in sentence]

def decode_from_char_map(sentence, characters):
    """Decodes an integer list into a string.

    Args:
        sentence (List[int]): Sentence made of indexes.
        characters (Map[char->int]): Map of characters to integer indexes.
    
    Returns:
        str: String from integer indexes, according to characters map.
    """
    reverse_map = {index: char for char, index in characters.items()}
    return ''.join([reverse_map[index] for index in sentence])
