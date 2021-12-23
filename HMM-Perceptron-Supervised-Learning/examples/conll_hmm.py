# Copyright 2013 Lars Buitinck / University of Amsterdam.

"""
Generic sequence prediction script using CoNLL format.
"""

from __future__ import print_function
import fileinput
from glob import glob
import sys

from seqlearn.datasets import load_conll
from seqlearn.evaluation import bio_f_score
from seqlearn.perceptron import StructuredPerceptron
from seqlearn.hmm import MultinomialHMM
from sklearn.metrics import accuracy_score
from subprocess import run, PIPE

def valid_conll_eval(fname):

    with open(fname, 'r') as file:
        data = file.read()

    pipe = run(["perl", "eval_conll2000_updated.pl"], stdout=PIPE, input=data, encoding='ascii')
    output = pipe.stdout

    tag_acc = float(output.split()[0])
    phrase_f1 = float(output.split()[1])

    print("tag_acc, phrase_f1", tag_acc, phrase_f1)
    return tag_acc, phrase_f1

def features(sentence, i):
    """Features for i'th token in sentence.

    Currently baseline named-entity recognition features, but these can
    easily be changed to do POS tagging or chunking.
    """

    word = sentence[i]

    yield "word:{}" + word.lower()

    if word[0].isupper():
        yield "CAP"

    if i > 0:
        yield "word-1:{}" + sentence[i - 1].lower()
        if i > 1:
            yield "word-2:{}" + sentence[i - 2].lower()
    if i + 1 < len(sentence):
        yield "word+1:{}" + sentence[i + 1].lower()
        if i + 2 < len(sentence):
            yield "word+2:{}" + sentence[i + 2].lower()


def describe(X, lengths):
    print("{0} sequences, {1} tokens.".format(len(lengths), X.shape[0]))


def load_data():
    # files = glob('nerdata/*.bio')

    # 80% training, 20% test
    print("Loading training data...", end=" ")
    # train_files = [f for i, f in enumerate(files) if i % 5 != 0]
    # train = load_conll(fileinput.input(train_files), features)
    train = load_conll("train.txt", features)
    X_train, _, lengths_train = train
    describe(X_train, lengths_train)

    val = load_conll("validation.txt", features)
    X_val, _, lengths_val = val
    describe(X_val, lengths_val)

    print("Loading test data...", end=" ")
    # test_files = [f for i, f in enumerate(files) if i % 5 == 0]
    # test = load_conll(fileinput.input(test_files), features)
    test = load_conll("test.txt", features)
    X_test, _, lengths_test = test
    describe(X_test, lengths_test)

    print("Loading entire training data...", end=" ")
    trainAndval = load_conll("train+val.txt", features)
    X_trainAndval, _, lengths_trainAndval = trainAndval
    describe(X_trainAndval, lengths_trainAndval)

    return train, val, test, trainAndval


if __name__ == "__main__":
    testfile = "out_hmm.txt"
    valfile = "valFile.txt"

    #print("Loading training data...", end=" ")
    #X_train, y_train, lengths_train = load_conll(sys.argv[1], features)
    #describe(X_train, lengths_train)

    train, val, test, train_val = load_data()
    X_train, y_train, lengths_train = train
    X_val, y_val, lengths_val = val
    X_test, y_test, lengths_test = test
    X_train_val, y_train_val, lengths_train_val = train_val

    lrs = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
    decoders = ["bestfirst", "viterbi"]
    #print("Loading test data...", end=" ")
    #X_test, y_test, lengths_test = load_conll(sys.argv[2], features)
    #describe(X_test, lengths_test)

    best_val_f1 = 0
    best_val_accuracy = 0
    best_lr = 0
    best_decoder = None
    model = None
    best_model = None
    for lr in lrs:
        for decoder in decoders:
            model = MultinomialHMM(alpha=lr, decode=decoder)
            print("Training %s" % model)
            model.fit(X_train, y_train, lengths_train)
            y_pred = model.predict(X_val, lengths_val)
            f = open(valfile, "w")
            for i in range(len(y_pred)):
                f.write("x y " + y_val[i][0] + " " + y_pred[i][0] + "\n")
            val_accuracy, phrase_f1 = valid_conll_eval(valfile)
            print("Accuracy: %.3f" % val_accuracy)
            print("CoNLL F1: %.3f" % phrase_f1)

            if phrase_f1 > best_val_f1:
                best_val_f1 = phrase_f1
                best_val_accuracy = val_accuracy
                best_lr = lr
                best_decoder = decoder
    
    print("The best lr_exponent is ", best_lr)
    print("The best decoder is ", decoder)
    
    model = MultinomialHMM(alpha=best_lr, decode=best_decoder)
    model.fit(X_train_val, y_train_val, lengths_train_val)
    y_pred = model.predict(X_test, lengths_test)
    f = open(testfile, "w")
    for i in range(len(y_pred)):
        f.write("x y " + y_test[i][0] + " " + y_pred[i][0] + "\n")

    f.close()
    test_accuracy, test_f1 = valid_conll_eval(testfile)
    print("Accuracy: %.3f" % test_accuracy)
    print("CoNLL F1: %.3f" % test_f1)
