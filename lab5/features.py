import transition

def extract(stack, queue, graph, feature_names, sentence):

    features = []

    features.append(stack[0]['postag'] if len(stack) > 0 else 'nil')
    features.append(stack[1]['postag'] if len(stack) > 1 else 'nil')
    features.append(stack[0]['form'] if len(stack) > 0 else 'nil')
    features.append(stack[1]['form'] if len(stack) > 1 else 'nil')
    features.append(queue[0]['postag'] if len(queue) > 0 else 'nil')
    features.append(queue[1]['postag'] if len(queue) > 1 else 'nil')
    features.append(queue[0]['form'] if len(queue) > 0 else 'nil')
    features.append(queue[1]['form'] if len(queue) > 1 else 'nil')
    features.append(transition.can_reduce(stack, graph))
    features.append(transition.can_leftarc(stack, graph))

    return dict(zip(feature_names, features))