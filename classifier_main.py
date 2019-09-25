# classifier_main.py

import argparse
import sys
import time
from nerdata import *
from utils import *
from collections import Counter
from optimizers import *
from typing import List
import numpy as np
import string
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--model', type=str, default='BAD', help='model to run (BAD, CLASSIFIER)')
    parser.add_argument('--train_path', type=str, default='data/eng.train', help='path to train set (you should not need to modify)')
    parser.add_argument('--dev_path', type=str, default='data/eng.testa', help='path to dev set (you should not need to modify)')
    parser.add_argument('--blind_test_path', type=str, default='data/eng.testb.blind', help='path to dev set (you should not need to modify)')
    parser.add_argument('--test_output_path', type=str, default='eng.testb.out', help='output path for test predictions')
    parser.add_argument('--no_run_on_test', dest='run_on_test', default=True, action='store_false', help='skip printing output on the test set')
    args = parser.parse_args()
    return args


class PersonExample(object):
    """
    Data wrapper for a single sentence for person classification, which consists of many individual tokens to classify.

    Attributes:
        tokens: the sentence to classify
        labels: 0 if non-person name, 1 if person name for each token in the sentence
    """
    def __init__(self, tokens: List[str], pos: List[str], labels: List[int]):
        self.tokens = tokens
        self.pos = pos
        self.labels = labels

    def __len__(self):
        return len(self.tokens)

def transform_for_classification(ner_exs: List[LabeledSentence]):
    """
    :param ner_exs: List of chunk-style NER examples
    :return: A list of PersonExamples extracted from the NER data
    """
    for labeled_sent in ner_exs:
        tags = bio_tags_from_chunks(labeled_sent.chunks, len(labeled_sent))
        labels = [1 if tag.endswith("PER") else 0 for tag in tags]
        words = [tok.word for tok in labeled_sent.tokens]
        pos = [tok.pos for tok in labeled_sent.tokens]
        yield PersonExample(words, pos, labels)

def checkPunctuation(word):
    if any(x in string.punctuation for x in word):
        return True
    else:
        return False
def checkDigit(word):
    if any(x.isdigit() for x in word):
        return True
    else:
        return False

def word2features(sent, i):
    word = sent.tokens[i].word
    postag = sent.tokens[i].pos
    features = {
        'word' : word,
        'word.lower' : word.lower(),
        'word[-3:]=' : word[-3:],
        'word[-2:]=' : word[-2:],
        'word.isupper' : word.isupper(),
        'word.istitle' : word.istitle(),
        'word.isdigit' : checkDigit(word),
        'postag:' : postag,
        'punctuation' : checkPunctuation(word)
    }
    if i > 0:
        word1 = sent.tokens[i-1].word
        postag1 = sent.tokens[i-1].pos
        features.update({
            '-1:word' : word1,
            '-1:word.lower' : word1.lower(),
            '-1:word.istitle' : word1.istitle(),
            '-1:word.isupper' : word1.isupper(),
            '-1:postag' : postag1
        })
    else:
        features.update({'BOS':1})
        
    if i < len(sent)-1:
        word1 = sent.tokens[i+1].word
        postag1 = sent.tokens[i+1].pos
        features.update({
            '+1:word' : word1,
            '+1:word.lower' : word1.lower(),
            '+1:word.istitle' : word1.istitle(),
            '+1:word.isupper' : word1.isupper(),
            '+1:postag' : postag1
        })
    else:
        features.update({'EOS':len(sent)-1})       
    return features

def get_features(ner_exs: List[LabeledSentence]):

    #get input features matrix
    input_features = []
    pos_list = []
    for sent in ner_exs:
        for i in range(len(sent)):
            if sent.tokens[i].pos not in pos_list:
                pos_list.append(sent.tokens[i].pos)
            input_features.append(word2features(sent, i))
    return input_features

def sent2labels(sent):
    tags = bio_tags_from_chunks(sent.chunks, len(sent))
    labels = np.asarray([1 if tag.endswith("PER") else 0 for tag in tags])
    return labels
    # return [label for token, postag, label in sent]

def get_labels(ner_exs: List[LabeledSentence]):
    labels = []
    for sent in ner_exs:
        tags = bio_tags_from_chunks(sent.chunks, len(sent))
        for tag in tags:
            if tag.endswith("PER"):
                labels.append(1)
            else:
                labels.append(0)
    return labels

def transform_input(ner_exs: List[LabeledSentence], data, vec):
    X = get_features(ner_exs)
    Y = get_labels(ner_exs)
    if data=='test' or data=='dev':
        X = vec.transform(X)
    else:
        X = vec.fit_transform(X)
    print('Feature length ', data,':', X.shape)
    return X, Y

def print_info(ner_exs: List[PersonExample]):
    for ex in ner_exs:
        for idx in range(0, len(ex)):
            if ex.labels[idx] == 1:
                print(ex.tokens[idx],'--',ex.pos[idx])

class CountBasedPersonClassifier(object):
    """
    Person classifier that takes counts of how often a word was observed to be the positive and negative class
    in training, and classifies as positive any tokens which are observed to be positive more than negative.
    Unknown tokens or ties default to negative.
    Attributes:
        pos_counts: how often each token occurred with the label 1 in training
        neg_counts: how often each token occurred with the label 0 in training
    """
    def __init__(self, pos_counts: Counter, neg_counts: Counter):
        self.pos_counts = pos_counts
        self.neg_counts = neg_counts

    def predict(self, tokens: List[str], idx: int):
        if self.pos_counts[tokens[idx]] > self.neg_counts[tokens[idx]]:
            return 1
        else:
            return 0


def train_count_based_binary_classifier(ner_exs: List[PersonExample]):
    """
    :param ner_exs: training examples to build the count-based classifier from
    :return: A CountBasedPersonClassifier using counts collected from the given examples
    """
    pos_counts = Counter()
    neg_counts = Counter()
    for ex in ner_exs:
        for idx in range(0, len(ex)):
            if ex.labels[idx] == 1:
                pos_counts[ex.tokens[idx]] += 1.0
            else:
                neg_counts[ex.tokens[idx]] += 1.0
    print(repr(pos_counts))
    print(repr(pos_counts["Peter"]))
    print(repr(neg_counts["Peter"]))
    print(repr(pos_counts["aslkdjtalk;sdjtakl"]))
    return CountBasedPersonClassifier(pos_counts, neg_counts)


class PersonClassifier(object):
    """
    Classifier to classify a token in a sentence as a PERSON token or not.
    Constructor arguments are merely suggestions; you're free to change these.
    """

    def __init__(self, weights: np.ndarray, indexer: Indexer):
        self.weights = weights
        self.indexer = indexer

    def predict(self, tokens: List[str], idx: int):
        """
        Makes a prediction for token at position idx in the given PersonExample
        :param tokens:
        :param idx:
        :return: 0 if not a person token, 1 if a person token
        """
    def predict(self, tokens, idx):
        raise Exception("Implement me!")


    # def orthographic_structure(indexer:Indexer):



def train_classifier(X, Y_train):
    clf = LogisticRegression(penalty='l2', dual=False, tol=0.001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, max_iter=100)
    # clf = LogisticRegression()
    model = clf.fit(X, Y_train)
    return model

def evaluate_classifier(exs: List[PersonExample], classifier: PersonClassifier):
    """
    Prints evaluation of the classifier on the given examples
    :param exs: PersonExample instances to run on
    :param classifier: classifier to evaluate
    """
    predictions = []
    golds = []
    for ex in exs:
        for idx in range(0, len(ex)):
            golds.append(ex.labels[idx])
            predictions.append(classifier.predict(ex.tokens, idx))
    print_evaluation(golds, predictions)

def evaluate_model(X, Y_train, classifier):
    predictions = []
    golds = []
    golds.append(Y_train)
    predictions.append(classifier.predict(X))
    print_evaluation(golds, predictions)

def print_evaluation(golds: List[int], predictions: List[int]):
    """
    Prints statistics about accuracy, precision, recall, and F1
    :param golds: list of {0, 1}-valued ground-truth labels for each token in the test set
    :param predictions: list of {0, 1}-valued predictions for each token
    :return:
    """
    num_correct = 0
    num_pos_correct = 0
    num_pred = 0
    num_gold = 0
    num_total = 0
    if len(golds) != len(predictions):
        raise Exception("Mismatched gold/pred lengths: %i / %i" % (len(golds), len(predictions)))
    for idx in range(0, len(golds[0])):
        gold = golds[0][idx]
        prediction = predictions[0][idx]
        if prediction == gold:
            num_correct += 1
        if prediction == 1:
            num_pred += 1
        if gold == 1:
            num_gold += 1
        if prediction == 1 and gold == 1:
            num_pos_correct += 1
        num_total += 1
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
    prec = float(num_pos_correct) / num_pred if num_pred > 0 else 0.0
    rec = float(num_pos_correct) / num_gold if num_gold > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec > 0 and rec > 0 else 0.0
    print("Precision: %i / %i = %f" % (num_pos_correct, num_pred, prec))
    print("Recall: %i / %i = %f" % (num_pos_correct, num_gold, rec))
    print("F1: %f" % f1)

def predict_write_output_to_file_bad(exs: List[PersonExample], classifier: PersonClassifier, outfile: str):
    """
    Runs prediction on exs and writes the outputs to outfile, one token per line
    :param exs:
    :param classifier:
    :param outfile:
    :return:
    """
    f = open(outfile, 'w')
    for ex in exs:
        for idx in range(0, len(ex)):
            prediction = classifier.predict(ex.tokens, idx)
            f.write(ex.tokens[idx] + " " + repr(int(prediction)) + "\n")
        f.write("\n")
    f.close()

def predict_write_output_to_file(exs: List[LabeledSentence], X_test, classifier: LogisticRegression, outfile: str):
    """
    Runs prediction on exs and writes the outputs to outfile, one token per line
    :param exs:
    :param classifier:
    :param outfile:
    :return:
    """
    f = open(outfile, 'w')
    predictions = classifier.predict(X_test)
    j = 0
    for ex in exs:
        for idx in range(0, len(ex)):
            f.write(ex.tokens[idx] + " " + repr(int(predictions[j])) + "\n")
            j += 1
        f.write("\n")
    f.close()

if __name__ == '__main__':
    start_time = time.time()
    args = _parse_args()
    print(args)
    # Load the training and test data
    indexer = Indexer()
    vec = DictVectorizer()
    train_class_exs = list(transform_for_classification(read_data(args.train_path)))
    dev_class_exs = list(transform_for_classification(read_data(args.dev_path)))
    X, Y_train = transform_input(read_data(args.train_path),'train',vec)
    X_dev, Y_dev = transform_input(read_data(args.dev_path), 'dev', vec)
    
    print("Data reading and training took %f seconds" % (time.time() - start_time))
    
    # Train the model
    if args.model == "BAD":
        classifier = train_count_based_binary_classifier(train_class_exs)
        print("===Train accuracy===")
        evaluate_classifier(train_class_exs, classifier)
        print("===Dev accuracy===")
        evaluate_classifier(dev_class_exs, classifier)

    else:
        classifier = train_classifier(X, Y_train)
        print("===Train accuracy===")
        evaluate_model(X, Y_train, classifier)
        print("===Dev accuracy===")
        evaluate_model(X_dev, Y_dev, classifier)


    # # Evaluate on training, development, and test data
    # print("===Train accuracy===")
    # # evaluate_model(X, Y_train, classifier)
    # evaluate_classifier(train_class_exs, classifier)
    # print("===Dev accuracy===")
    # # evaluate_model(X_dev, Y_dev, classifier)
    # evaluate_classifier(dev_class_exs, classifier)

    
    if args.run_on_test:
        print("Running on test")
        test_exs = list(transform_for_classification(read_data(args.blind_test_path)))
        X_test, Y_test = transform_input(read_data(args.blind_test_path),'test',vec)
        predict_write_output_to_file(test_exs, X_test, classifier, args.test_output_path)
        # predict_write_output_to_file_bad(test_exs, classifier, args.test_output_path)
        print("Wrote predictions on %i labeled sentences to %s" % (len(test_exs), args.test_output_path))



