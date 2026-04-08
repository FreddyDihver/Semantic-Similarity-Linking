import random


def print_rand_sentences(ori_sentence, tar_sentence, labels, n_sentences):
    """
    Prints n_sentences random sentences from ori_sentence and tar_sentence
    along with their corresponding labels.

    :param ori_sentence: list of original sentences
    :param tar_sentence: list of target sentences
    :param labels: list of labels (0 or 1)
    :param n_sentences: number of sentences to print
    """
    idxs = random.sample(range(len(ori_sentence)), n_sentences)
    print("\n\n".join(
        (str(i) + '\n' + ori_sentence[i] + '\n\n' + tar_sentence[i] + '\nLabel: ' + str(labels[i])) for i in idxs))


def print_sentence(ori_sentence, tar_sentence, ids: tuple):
    """
    Prints the original and target sentences at the given indices.

    :param ori_sentence: list of original sentences
    :param tar_sentence: list of target sentences
    :param ids: tuple of indices
    """
    print(ori_sentence[ids[0]] + '\n\n' + tar_sentence[ids[1]])