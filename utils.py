import csv
import matplotlib.pyplot as plt
import numpy
from collections import defaultdict
from scipy.stats import chisquare, ttest_ind


def n_utterances_counts(f_name, eou='__eou__'):
    n_utterances = []
    with open(f_name, encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for line in reader:
            n_utterances.append(line[0].count(eou))
    return n_utterances


def train_stats(f_name, eou='__eou__', eot='__eot__'):
    pos_utterances = []
    pos_turns = []
    pos_words = []
    neg_utterances = []
    neg_turns = []
    neg_words = []

    with open(f_name, encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for line in reader:
            label = int(float(line[2]))
            if label == 1:
                pos_utterances.append(line[0].count(eou))
                pos_turns.append(line[0].count(eot))
                pos_words.append(len(line[0].split()))
            elif label == 0:
                neg_utterances.append(line[0].count(eou))
                neg_turns.append(line[0].count(eot))
                neg_words.append(len(line[0].split()))
            else:
                print(line[2])

    return pos_utterances, pos_turns, pos_words, neg_utterances, neg_turns, neg_words


def normalize(data):
    total = float(sum(data))
    if total == 0:
        return data
    return data / total


def distribution(data, max_utt):
    counts = defaultdict(int)
    for d in data:
        counts[d] += 1

    total = float(len(data))
    distr = numpy.zeros(max_utt)

    for key, val in counts.items():
        if key < max_utt:
            distr[key] = val

    return distr, normalize(distr)


def plot_histogram(data, title, x_label, y_label, **kwargs):
    n, bins, patches = plt.hist(data, bins=500, facecolor='green', alpha=0.75, **kwargs)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)

    plt.show()


if __name__ == "__main__":

    train_n_utterances = n_utterances_counts("/home/petrbel/ubuntu-ranking-dataset-creator/src/train.csv")
    test_n_utterances = n_utterances_counts("/home/petrbel/ubuntu-ranking-dataset-creator/src/test.csv")
    valid_n_utterances = n_utterances_counts("/home/petrbel/ubuntu-ranking-dataset-creator/src/valid.csv")

    max_utt = max(max(train_n_utterances), max(test_n_utterances), max(valid_n_utterances)) + 1

    train_counts, train_distr = distribution(train_n_utterances, max_utt=max_utt)

    expected_test_counts = train_distr * len(test_n_utterances)
    real_test_counts, test_distr = distribution(test_n_utterances, max_utt=max_utt)
    _, pvalue = chisquare(real_test_counts + 1, expected_test_counts + 1)
    print("TestDataset: ChiSq pvalue={}".format(pvalue))

    expected_valid_counts = train_distr * len(valid_n_utterances)
    real_valid_counts, valid_distr = distribution(valid_n_utterances, max_utt=max_utt)
    _, pvalue = chisquare(real_valid_counts + 1, expected_valid_counts + 1)
    print("ValidDataset: ChiSq pvalue={}".format(pvalue))

    plot_histogram(train_n_utterances, "Train Utterances", "Number of utterances", "Count")
    plot_histogram(test_n_utterances, "Test Utterances", "Number of utterances", "Count")
    plot_histogram(valid_n_utterances, "Valid Utterances", "Number of utterances", "Count")

    print("Train Min: {}".format(min(train_n_utterances)))
    print("Train Max: {}".format(max(train_n_utterances)))
    print("Train Mean: {}".format(numpy.mean(train_n_utterances)))
    print("Train Std: {}".format(numpy.std(train_n_utterances)))

    print("Test Min: {}".format(min(test_n_utterances)))
    print("Test Max: {}".format(max(test_n_utterances)))
    print("Test Mean: {}".format(numpy.mean(test_n_utterances)))
    print("Test Std: {}".format(numpy.std(test_n_utterances)))

    print("Valid Min: {}".format(min(valid_n_utterances)))
    print("Valid Max: {}".format(max(valid_n_utterances)))
    print("Valid Mean: {}".format(numpy.mean(valid_n_utterances)))
    print("Valid Std: {}".format(numpy.std(valid_n_utterances)))

    pvalue = ttest_ind(train_n_utterances, test_n_utterances, equal_var=False).pvalue
    print("ttest: train-test, pvalue={}".format(pvalue))
    pvalue = ttest_ind(train_n_utterances, valid_n_utterances, equal_var=False).pvalue
    print("ttest: train-valid, pvalue={}".format(pvalue))

    pos_utterances, pos_turns, pos_words, neg_utterances, neg_turns, neg_words = train_stats(
        "/home/petrbel/ubuntu-ranking-dataset-creator/src/train.csv"
    )
