"""
CoNLL-X and CoNLL-U file readers and writers
"""
__author__ = "Pierre Nugues"

import os


def get_files(dir, suffix):
    """
    Returns all the files in a folder ending with suffix
    Recursive version
    :param dir:
    :param suffix:
    :return: the list of file names
    """
    files = []
    for file in os.listdir(dir):
        path = dir + '/' + file
        if os.path.isdir(path):
            files += get_files(path, suffix)
        elif os.path.isfile(path) and file.endswith(suffix):
            files.append(path)
    return files


def read_sentences(file):
    """
    Creates a list of sentences from the corpus
    Each sentence is a string
    :param file:
    :return:
    """
    f = open(file).read().strip()
    sentences = f.split('\n\n')
    return sentences


def split_rows(sentences, column_names):
    """
    Creates a list of sentence where each sentence is a list of lines
    Each line is a dictionary of columns
    :param sentences:
    :param column_names:
    :return:
    """
    new_sentences = []
    root_values = ['0', 'ROOT', 'ROOT', 'ROOT', 'ROOT', 'ROOT', '0', 'ROOT', '0', 'ROOT']
    start = dict(zip(column_names, root_values))
    for sentence in sentences:
        rows = sentence.split('\n')
        s_dict = dict()
        sentence = [dict(zip(column_names, row.split('\t'))) for row in rows if row[0] != '#']
        for w in sentence:
            s_dict[w['id']] = w

        s_dict[start['id']] = start
        new_sentences.append(s_dict)
    return new_sentences


def save(file, formatted_corpus, column_names):
    f_out = open(file, 'w')
    for sentence in formatted_corpus:
        for row in sentence[1:]:
            # print(row, flush=True)
            for col in column_names[:-1]:
                if col in row:
                    f_out.write(row[col] + '\t')
                else:
                    f_out.write('_\t')
            col = column_names[-1]
            if col in row:
                f_out.write(row[col] + '\n')
            else:
                f_out.write('_\n')
        f_out.write('\n')
    f_out.close()


if __name__ == '__main__':
    column_names_2006 = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']
    conll_u = ['id', 'form', 'lemma', 'upos', 'xpos', 'feats', 'head', 'deprel', 'deps', 'misc']

    train_file = 'swedish_talbanken05_train.conll'
    # train_file = 'test_x'
    test_file = 'swedish_talbanken05_test.conll'
    #English
    #train_file = 'ud-treebanks-v2.4/UD_English-EWT/en_ewt-ud-train.conllu'
    #Spanish
    #train_file = 'ud-treebanks-v2.4/UD_Spanish-AnCora/es_ancora-ud-train.conllu'
    #Norwegian
    #train_file = 'ud-treebanks-v2.4/UD_Norwegian-Bokmaal/no_bokmaal-ud-train.conllu'

    names = ['SS', 'OO']
    #names = ['nsubj','obj']

    sentences = read_sentences(train_file)
    formatted_corpus = split_rows(sentences, column_names_2006)
    #for s in formatted_corpus[0]:
    #    print(s)
    #print(train_file, len(formatted_corpus))
    #print(formatted_corpus[0])

    pairs = dict()
    nbr_of_pairs = 0

    triplets = dict()
    nbr_of_triplets = 0

    for s in formatted_corpus:
        subjects = [x for x in s.values() if x['deprel'] == names[0]]
        s_pairs = list()
        for subject in subjects:
            verb = s[subject['head']]
            pair = (verb['form'].lower(), subject['form'].lower())
            s_pairs.append((verb, subject))

            nbr_of_pairs += 1

            if pair not in pairs:
                pairs[pair] = 1
            else:
                pairs[pair] += 1

        objects = [x for x in s.values() if x['deprel'] == names[1]]

        for obj in objects:
            for p in s_pairs:
                if obj['head'] == p[1]['head']:
                    triplet = (p[0]['form'].lower(), p[1]['form'].lower(), obj['form'].lower())
                    nbr_of_triplets += 1

                    if triplet not in triplets:
                        triplets[triplet] = 1
                    else:
                        triplets[triplet] += 1




    print(nbr_of_pairs)
    most_common = sorted(pairs.items(), key=lambda x: -x[1])[:5]

    for x in most_common:
        print(x)

    print(nbr_of_triplets)
    most_common = sorted(triplets.items(), key=lambda x: -x[1])[:5]

    for x in most_common:
        print(x)

    #column_names_u = ['id', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'head', 'deprel', 'deps', 'misc']

    #files = get_files('../../corpus/ud-treebanks-v2.4/', 'train.conllu')
    #for train_file in files:
    #    sentences = read_sentences(train_file)
    #    formatted_corpus = split_rows(sentences, column_names_u)
    #    print(train_file, len(formatted_corpus))
    #    print(formatted_corpus[0])