"""
Gold standard parser
"""
__author__ = "Pierre Nugues"

import transition
import conll
import features
from ml import ML


def reference(stack, queue, graph):
    """
    Gold standard parsing
    Produces a sequence of transitions from a manually-annotated corpus:
    sh, re, ra.deprel, la.deprel
    :param stack: The stack
    :param queue: The input list
    :param graph: The set of relations already parsed
    :return: the transition and the grammatical function (deprel) in the
    form of transition.deprel
    """
    # Right arc
    if stack and stack[0]['id'] == queue[0]['head']:
        # print('ra', queue[0]['deprel'], stack[0]['cpostag'], queue[0]['cpostag'])
        deprel = '.' + queue[0]['deprel']
        stack, queue, graph = transition.right_arc(stack, queue, graph)
        return stack, queue, graph, 'ra' + deprel
    # Left arc
    if stack and queue[0]['id'] == stack[0]['head']:
        # print('la', stack[0]['deprel'], stack[0]['cpostag'], queue[0]['cpostag'])
        deprel = '.' + stack[0]['deprel']
        stack, queue, graph = transition.left_arc(stack, queue, graph)
        return stack, queue, graph, 'la' + deprel
    # Reduce
    if stack and transition.can_reduce(stack, graph):
        for word in stack:
            if (word['id'] == queue[0]['head'] or
                        word['head'] == queue[0]['id']):
                # print('re', stack[0]['cpostag'], queue[0]['cpostag'])
                stack, queue, graph = transition.reduce(stack, queue, graph)
                return stack, queue, graph, 're'
    # Shift
    # print('sh', [], queue[0]['cpostag'])
    stack, queue, graph = transition.shift(stack, queue, graph)
    return stack, queue, graph, 'sh'

def process(stack, queue, graph, trans):
    if trans[:2] == 'ra':
        return transition.right_arc(stack, queue, graph)
    elif trans[:2] == 'la':
        return transition.left_arc(stack, queue, graph)
    elif trans[:2] == 're':
        return transition.reduce(stack, queue, graph)
    elif trans[:2] == 'sh':
        return transition.shift(stack, queue, graph)

def extract_features(formatted_corpus, feature_names, training=True, model=None):
    non_proj = []

    X_1 = []
    y_1 = []

    sent_cnt = 0
    for sentence in formatted_corpus:
        sent_cnt += 1
        if sent_cnt % 1000 == 0:
            print(sent_cnt, 'sentences on', len(formatted_corpus), flush=True)
        stack = []
        queue = list(sentence)
        graph = {}
        graph['heads'] = {}
        graph['heads']['0'] = '0'
        graph['deprels'] = {}
        graph['deprels']['0'] = 'ROOT'
        transitions = []
        feats = []
        while queue:
            feats.append(features.extract(stack, queue, graph, feature_names, sentence))
            stack, queue, graph, trans = reference(stack, queue, graph)
            transitions.append(trans)
        stack, graph = transition.empty_stack(stack, graph)
        X_1.extend(feats)
        y_1.extend(transitions)
        #print('Equal graphs:', transition.equal_graphs(sentence, graph))
        if not transition.equal_graphs(sentence, graph):
            non_proj.append(sentence)

        # Poorman's projectivization to have well-formed graphs.
        for word in sentence:
            word['head'] = graph['heads'][word['id']]
        #print(transitions)
        #print(graph)

    #print(len(non_proj))
    #s = sorted(non_proj, key=lambda x: len(x))

    #print([x['form'] for x in s[0]])

    #for x in non_proj:
    #    print(len(x))
    #    print(x)

    return (X_1, y_1)


if __name__ == '__main__':
    train_file = '../lab4/swedish_talbanken05_train.conll'
    test_file = '../lab4/swedish_talbanken05_test_blind.conll'
    column_names_2006 = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']
    column_names_2006_test = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats']
    feature_names = ['stack0_POS', 'stack1_POS', 'stack0_word', 'stack1_word', 'queue0_POS', 'queue1_POS', 'queue0_word', 'queue1_word', 'can-re', 'can-la']

    sentences = conll.read_sentences(train_file)
    formatted_corpus = conll.split_rows(sentences, column_names_2006)

    X_1, y_1 = extract_features(formatted_corpus, feature_names)

    ml = ML()

    print("training...")
    if False:
        ml.load_model()
    else:
        ml.train(X_1, y_1)


    print("testing...")
    ml.test(X_1, y_1)

