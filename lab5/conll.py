"""
CoNLL-X and CoNLL-U file readers and writers
"""
__author__ = "Pierre Nugues"

import os

"""
Transitions for Nivre's parser
The parser state consists of the stack, the queue, and the partial graph
The partial graph is represented as a dictionary
"""

__author__ = "Pierre Nugues"

import conll
import dparser


def shift(stack, queue, graph):
    """
    Shift the first word in the queue onto the stack
    :param stack:
    :param queue:
    :param graph:
    :return:
    """
    stack = [queue[0]] + stack
    queue = queue[1:]
    return stack, queue, graph


def reduce(stack, queue, graph):
    """
    Remove the first item from the stack
    :param stack:
    :param queue:
    :param graph:
    :return:
    """
    return stack[1:], queue, graph


def right_arc(stack, queue, graph, deprel=False):
    """
    Creates an arc from the top of the stack to the first in the queue
    and shifts
    The deprel argument is either read from the manually-annotated corpus
    (deprel=False) or assigned by the parser. In this case, the deprel
    argument has a value
    :param stack:
    :param queue:
    :param graph:
    :param deprel: either read from the manually-annotated corpus (value false)
    or assigned by the parser
    :return:
    """
    graph['heads'][queue[0]['id']] = stack[0]['id']
    if deprel:
        graph['deprels'][queue[0]['id']] = deprel
    else:
        graph['deprels'][queue[0]['id']] = queue[0]['deprel']
    return shift(stack, queue, graph)


def left_arc(stack, queue, graph, deprel=False):
    """
    Creates an arc from the first in the queue to the top of the stack
    and reduces it.
    The deprel argument is either read from the manually-annotated corpus
    (deprel=False) or assigned by the parser. In this case, the deprel
    argument has a value
    :param stack:
    :param queue:
    :param graph:
    :param deprel: either read from the manually-annotated corpus (value false)
    or assigned by the parser
    :return:
    """
    graph['heads'][stack[0]['id']] = queue[0]['id']
    if deprel:
        graph['deprels'][stack[0]['id']] = deprel
    else:
        graph['deprels'][stack[0]['id']] = stack[0]['deprel']
    return reduce(stack, queue, graph)


def can_reduce(stack, graph):
    """
    Checks that the top of the stack has a head
    :param stack:
    :param graph:
    :return:
    """
    if not stack:
        return False
    if stack[0]['id'] in graph['heads']:
        return True
    else:
        return False


def can_leftarc(stack, graph):
    """
    Checks that the top of the has no head
    :param stack:
    :param graph:
    :return:
    """
    if not stack:
        return False
    if stack[0]['id'] in graph['heads']:
        return False
    else:
        return True


def can_rightarc(stack):
    """
    Simply checks there is a stack
    :param stack:
    :return:
    """
    if not stack:
        return False
    else:
        return True


def empty_stack(stack, graph):
    """
    Pops the items in the stack. If they have no head, they are assigned
    a ROOT head
    :param stack:
    :param graph:
    :return:
    """
    for word in stack:
        if word['id'] not in graph['heads']:
            graph['heads'][word['id']] = '0'
            graph['deprels'][word['id']] = 'ROOT'
    stack = []
    return stack, graph


def equal_graphs(sentence, graph):
    """
    Checks that the graph corresponds to the gold standard annotation of a sentence
    :param sentence:
    :param graph:
    :return:
    """
    equal = True
    for word in sentence:
        if word['id'] in graph['heads'] and word['head'] == graph['heads'][word['id']]:
            pass
        else:
            #print(word, flush=True)
            equal = False
    return equal


if __name__ == '__main__':
    pass
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
    start = [dict(zip(column_names, root_values))]
    for sentence in sentences:
        rows = sentence.split('\n')
        sentence = [dict(zip(column_names, row.split())) for row in rows if row[0] != '#']
        sentence = start + sentence
        new_sentences.append(sentence)
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

    train_file = '../../corpus/conllx/sv/swedish_talbanken05_train.conll'
    # train_file = 'test_x'
    test_file = '../../corpus/conllx/sv/swedish_talbanken05_test.conll'

    sentences = read_sentences(train_file)
    formatted_corpus = split_rows(sentences, column_names_2006)
    print(train_file, len(formatted_corpus))
    print(formatted_corpus[0])

    column_names_u = ['id', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'head', 'deprel', 'deps', 'misc']

    files = get_files('../../corpus/ud-treebanks-v1.3/', 'train.conllu')
    for train_file in files:
        sentences = read_sentences(train_file)
        formatted_corpus = split_rows(sentences, column_names_u)
        print(train_file, len(formatted_corpus))
        print(formatted_corpus[0])
