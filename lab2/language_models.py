import sys
import os
import regex as re
import math


DIVIDER = '====================================================='
SENTENCE = 'Det var en gång en katt som hette Nils.'


def main():
    text = sys.stdin.read()

    tokens = tokenize(text)

    uni_freq = count_unigrams(tokens)
    bi_freq = count_bigrams(tokens)

    sentence_uni_prob(SENTENCE, uni_freq, len(tokens))
    sentence_bi_prob(SENTENCE, uni_freq, bi_freq, len(tokens))

    indices = [i for i, x in enumerate(tokens) if x == "<s>"]

    text = tokens_to_text(tokens[indices[-5]:])

    return 0


def tokenize(text):
    """uses the punctuation and symbols to break the text into words
    returns a list of words"""
    text = remove_capital_letters(text)

    spaced_tokens = re.sub('([\p{S}\p{P}])', r' \1 ', text)
    one_token_per_line = re.sub('\s+', '\n', spaced_tokens)
    tokens = one_token_per_line.split()

    tokens = remove_unwanted_tokens(tokens)
    tokens = insert_tags(tokens)
    return tokens


def remove_capital_letters(text):
    return text.lower()


def insert_tags(tokens):
    end_of_sentence = {'.', '!', '?'}
    result = ['<s>']

    for t in tokens:
        if t in end_of_sentence:
            result.extend(['</s>', '<s>'])
        else:
            result.append(t)

    return result[:-1]


def remove_unwanted_tokens(tokens):
    not_accepted = ',:;-"123456789/_*()|[]#\xad\{\}»´ `$\''
    return [t for t in tokens if t not in not_accepted]



def tokens_to_text(tokens):
    return ' '.join(tokens).replace('</s> ', '</s>\n')


def count_unigrams(words):
    frequency = {}
    for word in words:
        if word in frequency:
            frequency[word] += 1
        else:
            frequency[word] = 1
    return frequency


def count_bigrams(words):
    bigrams = [tuple(words[inx:inx + 2])
               for inx in range(len(words) - 1)]
    frequencies = {}
    for bigram in bigrams:
        if bigram in frequencies:
            frequencies[bigram] += 1
        else:
            frequencies[bigram] = 1
    return frequencies


def sentence_uni_prob(sentence, frequencies, nbr_of_words):
    print('Unigram model')
    print(DIVIDER)
    print('wi C(wi) #words P(wi)')
    print(DIVIDER)

    words = tokenize(sentence)[1:]

    P = 1

    for wi in words:
        Cwi = frequencies[wi]
        Pwi = Cwi / nbr_of_words

        P *= Pwi

        print('{} {} {} {}'.format(wi, Cwi, nbr_of_words, Pwi))

    prob_unigram = P
    geo_mean_prob = math.pow(P, 1/len(words))
    entropy = math.log2(P) * (-1 / len(words))
    perplexity = math.pow(2, entropy)

    print(DIVIDER)
    print('Prob. unigram: {}'.format(prob_unigram))
    print('Geometric mean prob.: {}'.format(geo_mean_prob))
    print('Entropy rate: {}'.format(entropy))
    print('Perplexity: {}'.format(perplexity))


def sentence_bi_prob(sentence, uni_freq, bi_freq, nbr_of_words):
    print('Bigram model')
    print(DIVIDER)
    print('wi wi+1 Ci,i+1 C(i) P(wi+1|wi)')
    print(DIVIDER)

    words = tokenize(sentence)

    P = 1

    for i in range(len(words) - 1):
        wi = words[i]
        wi1 = words[i+1]
        Ci = uni_freq[wi]
        try:
            Cii1 = bi_freq[(wi, wi1)]
            p = Cii1 / Ci
        except:
            p = Ci / nbr_of_words

        P *= p

        print('{} {} {} {} {}'.format(wi, wi1, Cii1, Ci, p))

    prob_unigram = P
    geo_mean_prob = math.pow(P, 1/len(words))
    entropy = math.log2(P) * (-1 / len(words))
    perplexity = math.pow(2, entropy)
    print(DIVIDER)
    print('Prob. unigram: {}'.format(prob_unigram))
    print('Geometric mean prob.: {}'.format(geo_mean_prob))
    print('Entropy rate: {}'.format(entropy))
    print('Perplexity: {}'.format(perplexity))


if __name__ == '__main__':
    exit(main())
