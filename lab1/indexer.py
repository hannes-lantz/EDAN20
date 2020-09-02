import sys
import os
import re
import pickle
import math
from numpy import dot
from numpy.linalg import norm

def main():
    dir_path = sys.argv[1]

    file_names = get_file_names(dir_path)

    master_index = dict()
    number_of_words = dict()

    for file_name in file_names:
        index = process_file(os.path.join(dir_path, file_name))
        number_of_words[file_name] = 0
        for word, indicies in index.items():
            if word not in master_index:
                master_index[word] = dict()
            master_index[word][file_name] = indicies
            number_of_words[file_name] += len(indicies)

    print_index(master_index)
    best_match = similarity(master_index, number_of_words, file_names)

    print(best_match)

    return 0

def get_file_names(dir_path):
    return [f for f in os.listdir(dir_path) if f.endswith('.txt')]


def process_file(file_path):
    with open(file_path, 'r') as f:
        text = ' '.join(f.readlines()).lower()

    index = dict()

    for w in re.finditer(r'\w+', text):
        i = w.start()
        word = w.group()
        index[word] = [*index[word], i] if word in index else [i]

    return index


def print_index(index):
    pickle.dump(index, open('output.idx', 'wb'))
    print(index['samlar'])


def similarity(master_index, number_of_words, file_names):
    tfidf_collection = dict()
    for file_name in file_names:
        tfidf_collection[file_name] = []

    words = master_index.keys()

    for file_name in file_names:
        tfidf_collection[file_name] = [(word, calc_tfidf(word, file_name, master_index, number_of_words)) for word in words]

    best_match = ('','')
    best_similarity = 0

    for i in range(len(file_names) - 1):
        for j in range(i + 1, len(file_names)):
            a = [value for _, value in tfidf_collection[file_names[i]]]
            b = [value for _, value in tfidf_collection[file_names[j]]]
            cos_sim = dot(a, b)/(norm(a)*norm(b))
            if cos_sim > best_similarity:
                best_similarity = cos_sim
                best_match = (file_names[i], file_names[j])

    return best_match



def calc_tfidf(t, d, master_index, number_of_words):
    tf = (len(master_index[t][d]) if d in master_index[t] else 0)/number_of_words[d]
    idf = math.log10(len(number_of_words.keys())/len(master_index[t].keys()))
    tfidf = tf*idf
    return tfidf


if __name__ == '__main__':
    exit(main())