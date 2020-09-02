import transition


def extract(stack, queue, graph, feature_names, sentence):
    features = list()

    POS_TAG = "postag"
    WORD_TAG = "form"
    ID_TAG = "id"
    DEPREL_TAG = "deprel"
    NULL_VALUE = "nil"
    HEAD_TAG = "head"
    PHEAD_TAG = "phead"
    PDEPREL_TAG = "pdeprel"

    if stack:
        stack_0_pos = stack[0][POS_TAG]
        stack_0_word = stack[0][WORD_TAG]
    else:
        stack_0_pos = NULL_VALUE
        stack_0_word = NULL_VALUE

    if len(stack) > 1:
        stack_1_pos = stack[1][POS_TAG]
        stack_1_word = stack[1][WORD_TAG]
    else:
        stack_1_pos = NULL_VALUE
        stack_1_word = NULL_VALUE

    if queue:
        queue_0_pos = queue[0][POS_TAG]
        queue_0_word = queue[0][WORD_TAG]
    else:
        queue_0_pos = NULL_VALUE
        queue_0_word = NULL_VALUE

    if len(queue) > 1:
        queue_1_pos = queue[1][POS_TAG]
        queue_1_word = queue[1][WORD_TAG]
    else:
        queue_1_pos = NULL_VALUE
        queue_1_word = NULL_VALUE

    if len(feature_names) == 6:
        features.append(stack_0_word)
        features.append(stack_0_pos)

        features.append(queue_0_word)
        features.append(queue_0_pos)

        features.append(transition.can_reduce(stack, graph))
        features.append(transition.can_leftarc(stack, graph))

    elif len(feature_names) == 10:
        features.append(stack_0_word)
        features.append(stack_0_pos)

        features.append(stack_1_word)
        features.append(stack_1_pos)

        features.append(queue_0_word)
        features.append(queue_0_pos)

        features.append(queue_1_word)
        features.append(queue_1_pos)

        features.append(transition.can_reduce(stack, graph))
        features.append(transition.can_leftarc(stack, graph))

    elif len(feature_names) == 14:
        # No word
        if stack_0_word == NULL_VALUE:
            after_stack_0_word = NULL_VALUE
            after_stack_0_pos = NULL_VALUE

            before_stack_0_word = NULL_VALUE
            before_stack_0_pos = NULL_VALUE

        else:
            id_stack_0 = int(stack[0]['id'])
            # Last word
            if id_stack_0 == len(sentence) - 1:
                after_stack_0_word = NULL_VALUE
                after_stack_0_pos = NULL_VALUE
            else:
                next_word = sentence[id_stack_0 + 1]
                after_stack_0_word = next_word[WORD_TAG]
                after_stack_0_pos = next_word[POS_TAG]
            # First word
            if id_stack_0 == 0:
                before_stack_0_word = NULL_VALUE
                before_stack_0_pos = NULL_VALUE
            else:
                previous_word = sentence[id_stack_0]
                before_stack_0_word = previous_word[WORD_TAG]
                before_stack_0_pos = previous_word[POS_TAG]

        features.append(stack_0_word)
        features.append(stack_0_pos)

        features.append(stack_1_word)
        features.append(stack_1_pos)

        features.append(queue_0_word)
        features.append(queue_0_pos)

        features.append(queue_1_word)
        features.append(queue_1_pos)

        features.append(after_stack_0_word)
        features.append(after_stack_0_pos)

        features.append(before_stack_0_word)
        features.append(before_stack_0_pos)

        features.append(transition.can_reduce(stack, graph))
        features.append(transition.can_leftarc(stack, graph))

    features = dict(zip(feature_names, features))

    return features