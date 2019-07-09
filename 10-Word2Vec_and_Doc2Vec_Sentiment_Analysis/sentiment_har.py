import sys
import collections
import nltk
import time
import random
random.seed(0)

from gensim.models.doc2vec import LabeledSentence, Doc2Vec
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            if (len(words) == 0): continue
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            if (len(words) == 0): continue
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            if (len(words) == 0): continue
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            if (len(words) == 0): continue
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg


def get_word_dict(dataset, stopwords):
    result = {}
    for line in dataset:
        words = set(line)
        for word in words:
            if word not in stopwords:
                if word not in result:
                    result[word] = 1
                else:
                    result[word] += 1
    return result

def get_feature_vector(dataset, features):
    result = []
    for line in dataset:
        vector = dict.fromkeys(features, 0)
        for word in line:
            if word in features:
                vector[word] = 1
        result.append(list(vector.values()))
    return result


def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))

    # Determine a list of words that will be used as features.
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
    
    positive_words = get_word_dict(train_pos, stopwords)
    negative_words = get_word_dict(train_neg, stopwords)

    min_percent = 0.01
    min_positive = int(min_percent * len(train_pos))
    min_negative = int(min_percent * len(train_neg))

    features = set()
    
    for word, count in positive_words.items():
        if count > min_positive and word in negative_words and (count > 2 * negative_words[word] or negative_words[word] > 2 * count):
                features.add(word)

    for word, count in negative_words.items():
        if count > min_negative and word in positive_words and (count > 2 * positive_words[word] or positive_words[word] > 2 * count):
            features.add(word)

    features = list(features)
    print(len(features))    

    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.
    train_pos_vec = get_feature_vector(train_pos, features)
    train_neg_vec = get_feature_vector(train_neg, features)
    test_pos_vec = get_feature_vector(test_pos, features)
    test_neg_vec = get_feature_vector(test_neg, features)

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec


def get_labeled_sentences(dataset, label):
    result = []
    for i, words in enumerate(dataset):
        result.append(LabeledSentence(words = words, tags = [label + '_' + str(i)]))
    return result

def get_feature_vec(model, dataset, label):
    vectors = []
    for i, words in enumerate(dataset):
        vectors.append(model.docvecs[label + '_' + str(i)])
    return vectors


def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires TaggedDocument objects as input.
    # Turn the datasets from lists of words to lists of TaggedDocument objects.

    labeled_train_pos = get_labeled_sentences(train_pos, 'TRAIN_POS')
    labeled_train_neg = get_labeled_sentences(train_neg, 'TRAIN_NEG')
    labeled_test_pos = get_labeled_sentences(test_pos, 'TEST_POS')
    labeled_test_neg = get_labeled_sentences(test_neg, 'TEST_NEG')

    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    print("Doc2Vec")
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)

    # Train the model
    # This may take a bit to run
    for i in range(5):
        print("Training iteration %d" % (i))
        random.shuffle(sentences)
        model.train(sentences,total_examples=model.corpus_count, epochs=model.iter)
    print("end of training")

    # Use the docvecs function to extract the feature vectors for the training and test data
    train_pos_vec = get_feature_vec(model, train_pos, 'TRAIN_POS')
    train_neg_vec = get_feature_vec(model, train_neg, 'TRAIN_NEG')
    test_pos_vec = get_feature_vec(model, test_pos, 'TEST_POS')
    test_neg_vec = get_feature_vec(model, test_neg, 'TEST_NEG')


    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec


def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    X = train_pos_vec + train_neg_vec
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    
    nb_model = BernoulliNB(alpha = 1.0, binarize = None)
    nb_model.fit(X, Y)

    # For LogisticRegression, pass no parameters
    lr_model = LogisticRegression()
    lr_model.fit(X, Y)
    
    return nb_model, lr_model



def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    X = train_pos_vec+train_neg_vec
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    
    nb_model = GaussianNB()
    nb_model.fit(X, Y)

    # For LogisticRegression, pass no parameters
    lr_model = LogisticRegression()
    lr_model.fit(X, Y)

    return nb_model, lr_model


def build_models_DOC_W2V(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    X = train_pos_vec + train_neg_vec
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    
    nb_model = GaussianNB()
    nb_model.fit(X, Y)
    
    lr_model = LogisticRegression()
    lr_model.fit(X, Y)
    
    return nb_model, lr_model


def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    predicted_pos = model.predict(test_pos_vec)
    true_pos = sum(predicted_pos == 'pos')
    false_neg = sum(predicted_pos == 'neg')
    predicted_neg = model.predict(test_neg_vec)
    false_pos = sum(predicted_neg == 'pos')
    true_neg = sum(predicted_neg == 'neg')

    total = float(true_pos + true_neg + false_pos + false_neg)
    accurate = float(true_pos + true_neg)
    accuracy =  accurate / total

    if print_confusion:
        print("predicted:\tpos\tneg")
        print("actual:")
        print("pos\t\t%d\t%d" % (true_pos, false_neg))
        print("neg\t\t%d\t%d" % (false_pos, true_neg))

    print("accuracy: %f" % (accuracy))


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print("python sentiment.py <path_to_data> <nlp|d2v|w2v>")
    exit(1)
path_to_data = sys.argv[1]
method = sys.argv[2]

def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)
    
    # Using saved models and vectors for method == 'nlp'. (Orginal runtime = 5 mins; Current runtime = 10 seconds)
    if method == "nlp" or method == "0":
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)

    # Using saved models and vectors for method == 'd2v'. (Orginal runtime = 10 mins; Current runtime = 10 seconds)
    if method == "d2v" or method == "1":
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)

    print("Naive Bayes")
    print("-----------")
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)

    print("Logistic Regression")
    print("-------------------")
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)

if __name__ == "__main__":
    main()
