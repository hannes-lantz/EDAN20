"""
Gold standard parser
"""
__author__ = "Pierre Nugues"

import transition
import conll
import features

from sklearn import linear_model
from sklearn import tree
from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
import pickle


feature_names_6 = [
    'stack_0_word',
    'stack_0_pos',
    'queue_0_word',
    'queue_0_pos',
    'can-re',
    'can-la'
]

feature_names_10 = [
    'stack_0_word',
    'stack_0_pos',
    'stack_1_word',
    'stack_1_pos',
    'queue_0_word',
    'queue_0_pos',
    'queue_1_word',
    'queue_1_pos',
    'can-re',
    'can-la'
]

feature_names_14 = [
    'stack_0_word',
    'stack_0_pos',
    'stack_1_word',
    'stack_1_pos',
    'queue_0_word',
    'queue_0_pos',
    'queue_1_word',
    'queue_1_pos',
    'after_stack_0_word',
    'after_stack_0_pos',
    'before_stack_0_word',
    'before_stack_0_pos',
    'can-re',
    'can-la'
]

FEATURE_NAMES = feature_names_14

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

def encode_classes(y_symbols):
    classes = sorted(list(set(y_symbols)))

    dict_classes = dict(enumerate(classes))

    inv_dict_classes = {v: k for k, v in dict_classes.items()}

    y = [inv_dict_classes[i] for i in y_symbols]
    return y, dict_classes, inv_dict_classes

def extract_all_features(formatted_corpus):
    sent_cnt = 0

    y_symbols = []
    X_dict = list()

    for sentence in formatted_corpus:
        sent_cnt += 1
        if sent_cnt % 1000 == 0:
            _ = 1
            print(sent_cnt, 'sentences on', len(formatted_corpus), flush=True)
        stack = []
        queue = list(sentence)
        graph = {}
        graph['heads'] = {}
        graph['heads']['0'] = '0'
        graph['deprels'] = {}
        graph['deprels']['0'] = 'ROOT'



#        transitions = []
        while queue:
            x = features.extract(stack, queue, graph, FEATURE_NAMES, sentence)
            X_dict.append(x)

            stack, queue, graph, trans = reference(stack, queue, graph)

            y_symbols.append(trans)
#            transitions.append(trans)
        stack, graph = transition.empty_stack(stack, graph)
 #       print('Equal graphs:', transition.equal_graphs(sentence, graph))

        # Poorman's projectivization to have well-formed graphs.
        for word in sentence:
            word['head'] = graph['heads'][word['id']]
#        print(transitions)
#        print(graph)

    return X_dict, y_symbols

def parse_ml(stack, queue, graph, trans):
    if stack and trans[:2] == 'ra' and transition.can_rightarc(stack):
        stack, queue, graph = transition.right_arc(stack, queue, graph, trans[3:])
        return stack, queue, graph, 'ra'
    if stack and trans[:2] == 'la' and transition.can_leftarc(stack, graph):
        stack, queue, graph = transition.left_arc(stack, queue, graph, trans[3:])
        return stack, queue, graph, 'la'
    if stack and trans[:2] == 're' and transition.can_reduce(stack, graph):
        stack, queue, graph = transition.reduce(stack, queue, graph)
        return stack, queue, graph, 're'
    stack, queue, graph = transition.shift(stack, queue, graph)
    return stack, queue, graph, 'sh'

if __name__ == '__main__':
    train_file = './swedish_talbanken05_train.conll'
    test_file = './swedish_talbanken05_test_blind.conll'
    column_names_2006 = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']
    column_names_2006_test = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats']

    sentences = conll.read_sentences(train_file)
    formatted_corpus = conll.split_rows(sentences, column_names_2006)

    LOGISTIC_MODEL = 'logistic-regression'
    PERCEPTRON_MODEL = 'perceptrion'
    DECISION_MODEL = 'decision-tree-classifier'

    MODEL = LOGISTIC_MODEL

    MODEL_FILE_NAME = "{}.trained_model".format(MODEL)
    DICT_VECTORIZER_FILE_NAME = "dict-vectorizer.trained_model"
    DICT_CLASSES_FILE_NAME = "dict-classes.trained_model"
    FEATURE_NAMES_FILE_NAME = "feature-names.trained_model"
    Y_FILE_NAME = "y.trained_model"
    X_FILE_NAME = "x.trained_model"

    if MODEL == LOGISTIC_MODEL:
        classifier = linear_model.LogisticRegression(penalty='l2', dual=True, solver='liblinear', verbose=1)
#        classifier = linear_model.LogisticRegression(penalty='l2', dual=False, solver='lbfgs', verbose=1, max_iter=5000)
    elif MODEL == PERCEPTRON_MODEL:
        classifier = linear_model.Perceptron(penalty='l2', verbose=True)
    elif MODEL == DECISION_MODEL:
        classifier = tree.DecisionTreeClassifier()

    try:
        model = pickle.load(open(MODEL_FILE_NAME, 'rb'))
        vec = pickle.load(open(DICT_VECTORIZER_FILE_NAME, 'rb'))
        dict_classes = pickle.load(open(DICT_CLASSES_FILE_NAME, 'rb'))
        feature_names = pickle.load(open(FEATURE_NAMES_FILE_NAME, 'rb'))
        y = pickle.load(open(Y_FILE_NAME, 'rb'))
        X = pickle.load(open(X_FILE_NAME, 'rb'))
        if(feature_names != FEATURE_NAMES):
            raise FileNotFoundError

        print("Model was loaded from save")

    except FileNotFoundError:
        print("No saved model was found")

        X_dict, y_train_symbols = extract_all_features(formatted_corpus)
        for i, x in enumerate(X_dict):
            if i > 8:
                break;
            print("x =", list(x.values()), ", y =", y_train_symbols[i])


        vec = DictVectorizer(sparse=True)
        X = vec.fit_transform(X_dict)
        y, dict_classes, inv_dict_classes = encode_classes(y_train_symbols)

        model = classifier.fit(X, y)
        pickle.dump(model, open(MODEL_FILE_NAME, 'wb'))
        pickle.dump(vec, open(DICT_VECTORIZER_FILE_NAME, 'wb'))
        pickle.dump(dict_classes, open(DICT_CLASSES_FILE_NAME, 'wb'))
        pickle.dump(FEATURE_NAMES, open(FEATURE_NAMES_FILE_NAME, 'wb'))
        pickle.dump(y, open(Y_FILE_NAME, 'wb'))
        pickle.dump(X, open(X_FILE_NAME, 'wb'))
    print(model)
    y_pred = model.predict(X)
    aS = accuracy_score(y, y_pred)
    print("Accuracy on train data:", str(aS))


    # Beginning of lab6, reading test data
    sentences = conll.read_sentences(test_file)
    formatted_corpus = conll.split_rows(sentences, column_names_2006_test)

    sentence_counter = 0

    y_predicted_symbols = []

    for sentence in formatted_corpus:
        sentence_counter += 1
        if sentence_counter % 1000 == 0:
            x = True
            print(sentence_counter, 'sentences on', len(formatted_corpus), flush=True)
        stack = []
        queue = list(sentence)
        graph = {}
        graph['heads'] = {}
        graph['heads']['0'] = '0'
        graph['deprels'] = {}
        graph['deprels']['0'] = 'ROOT'

        while queue:
            x = features.extract(stack, queue, graph, FEATURE_NAMES, sentence)


            X_test = vec.transform(x)

            predicted_trans_index = model.predict(X_test)[0]
            predicted_trans = dict_classes[predicted_trans_index]

            # Build graph
            stack, queue, graph, trans = parse_ml(stack, queue, graph, predicted_trans)

            y_predicted_symbols.append(trans)

        stack, graph = transition.empty_stack(stack, graph)

        for word in sentence:
            word_id = word['id']
            try:
                word['head'] = graph['heads'][word_id]
                word['phead'] = graph['heads'][word_id]
            except KeyError:
                word['head'] = '_'
                word['phead'] = '_'

            try:
                word['deprel'] = graph['deprels'][word_id]
                word['pdeprel'] = graph['deprels'][word_id]
            except KeyError:
                word['deprel'] = '_'
                word['pdeprel'] = '_'
            x = True

    conll.save('results.txt', formatted_corpus, column_names_2006)

#    print("Classification report for classifier %s:\n%s\n"
#        % (classifier, metrics.classification_report(y_train_symbols, list(map(lambda y_pred: dict_classes[y_pred], y_predicted_symbols)))))
