import argparse
import os
import csv
import random
import urllib.request
import tarfile

import nltk
from nltk.stem import SnowballStemmer, WordNetLemmatizer

dialog_end_symbol = "__dialog_end__"
end_of_utterance_symbol = "__eou__"
end_of_turn_symbol = "__eot__"


def translate_dialog_to_lists(dialog_filename):
    """
    Translates the dialog file into list of lists of utterances by user turns.
    """
    dialog = []
    with open(dialog_filename, 'r', encoding='utf-8') as dialog_file:
        dialog_reader = csv.reader(dialog_file, delimiter='\t', quoting=csv.QUOTE_NONE)
        first_turn = True
        same_user_utterances = []
        dialog.append(same_user_utterances)
        for dialog_line in dialog_reader:
            if first_turn:
                last_user = dialog_line[1]
                first_turn = False
            if last_user != dialog_line[1]:
                same_user_utterances = []
                dialog.append(same_user_utterances)
            same_user_utterances.append(dialog_line[3])
            last_user = dialog_line[1]
        dialog.append([dialog_end_symbol])
    return dialog


def singe_user_utterances_to_string(utterances_list):
    """
    Join multiple utterances by a single user into a string with __eou__ after each utterance.
    """
    return " ".join([utt + " " + end_of_utterance_symbol for utt in utterances_list])


def dialog_turns_to_string(dialog):
    """
    Join a dialog (list of user utterance lists) into a single string with turns separated by __eot__.
    """
    turns_as_strings = [singe_user_utterances_to_string(turn) for turn in dialog]
    return " ".join([turn + " " + end_of_turn_symbol for turn in turns_as_strings])


def create_random_context(dialog, rng, minimum_context_length=2, max_context_length=20):
    """
    Sample a random dialog context (sub-dialog) and return its string and index of next utterance.
    """
    max_len = min(max_context_length, len(dialog)) - 2
    if max_len <= minimum_context_length:
        context_turns = max_len
    else:
        context_turns = rng.randint(minimum_context_length, max_len)
    return dialog_turns_to_string(dialog[:context_turns]), context_turns


def get_random_utterances_from_corpus(candidate_dialog_paths, rng, utterances_num=9, min_turn=3, max_turn=20):
    """
    Sample random utterances from the corpus dialogs.
    """
    utterances = []
    dialogs_num = len(candidate_dialog_paths)

    for _ in range(utterances_num):
        dialog_path = candidate_dialog_paths[rng.randint(0, dialogs_num - 1)]
        dialog = translate_dialog_to_lists(dialog_path)

        dialog_len = len(dialog) - 1
        if dialog_len < min_turn:
            print(f"Dialog {dialog_path} too short: {dialog_len}")
            exit(1)

        max_ix = min(max_turn, dialog_len) - 1
        if min_turn - 1 == max_ix:
            turn_index = max_ix
        else:
            turn_index = rng.randint(min_turn, max_ix)

        utterance = singe_user_utterances_to_string(dialog[turn_index])
        utterances.append(utterance)
    return utterances


def create_single_dialog_train_example(context_dialog_path, candidate_dialog_paths, rng, positive_probability,
                                       minimum_context_length=2, max_context_length=20):
    """
    Create one training example: (context, response, label).
    """
    dialog = translate_dialog_to_lists(context_dialog_path)

    context_str, next_utterance_ix = create_random_context(dialog, rng,
                                                           minimum_context_length=minimum_context_length,
                                                           max_context_length=max_context_length)
    if positive_probability > rng.random():
        response = singe_user_utterances_to_string(dialog[next_utterance_ix])
        label = 1.0
    else:
        response = get_random_utterances_from_corpus(candidate_dialog_paths, rng, 1,
                                                     min_turn=minimum_context_length + 1,
                                                     max_turn=max_context_length)[0]
        label = 0.0
    return context_str, response, label


def create_single_dialog_test_example(context_dialog_path, candidate_dialog_paths, rng, distractors_num, max_context_length):
    """
    Create one test/valid example: (context, positive_response, [negative_responses]).
    """
    dialog = translate_dialog_to_lists(context_dialog_path)
    context_str, next_utterance_ix = create_random_context(dialog, rng, max_context_length=max_context_length)
    positive_response = singe_user_utterances_to_string(dialog[next_utterance_ix])
    negative_responses = get_random_utterances_from_corpus(candidate_dialog_paths, rng, distractors_num)
    return context_str, positive_response, negative_responses


def create_examples(candidate_dialog_paths, examples_num, creator_function):
    """
    Create a list of examples using a creator function.
    """
    examples = []
    unique_dialogs_num = len(candidate_dialog_paths)
    for i in range(examples_num):
        context_dialog = candidate_dialog_paths[i % unique_dialogs_num]
        if i % 1000 == 0:
            print(i)
        examples.append(creator_function(context_dialog, candidate_dialog_paths))
    return examples


def convert_csv_with_dialog_paths(csv_file, data_root):
    """
    Convert CSV lines to dialog file paths.
    """
    dialog_paths = []
    for line in csv_file:
        file, dir = map(str.strip, line.split(","))
        full_path = os.path.join(data_root, "dialogs", dir, file)
        dialog_paths.append(full_path)
    return dialog_paths


def prepare_data_maybe_download(directory):
    """
    Download and unpack dialogs if needed.
    """
    filename = 'ubuntu_dialogs.tgz'
    url = 'http://cs.mcgill.ca/~jpineau/datasets/ubuntu-corpus-1.0/ubuntu_dialogs.tgz'
    dialogs_path = os.path.join(directory, 'dialogs')

    if not os.path.exists(os.path.join(directory, "10", "1.tst")):
        archive_path = os.path.join(directory, filename)
        if not os.path.exists(archive_path):
            print(f"Downloading {url} to {archive_path}")
            filepath, _ = urllib.request.urlretrieve(url, archive_path)
            print("Successfully downloaded " + filepath)
        if not os.path.exists(dialogs_path):
            print("Unpacking dialogs ...")
            with tarfile.open(archive_path) as tar:
                tar.extractall(path=directory)
            print("Archive unpacked.")


def create_eval_dataset(args, file_list_csv):
    rng = random.Random(args.seed)
    meta_path = os.path.join(os.path.dirname(__file__), "meta", file_list_csv)
    with open(meta_path, 'r', encoding='utf-8') as f:
        dialog_paths = convert_csv_with_dialog_paths(f, args.data_root)

    data_set = create_examples(dialog_paths,
                               len(dialog_paths),
                               lambda context_dialog, candidates: create_single_dialog_test_example(context_dialog, candidates, rng,
                                                                                                 args.n, args.max_context_length))

    with open(args.output, 'w', encoding='utf-8', newline='') as outf:
        w = csv.writer(outf)
        header = ["Context", "Ground Truth Utterance"]
        header.extend([f"Distractor_{i}" for i in range(args.n)])
        w.writerow(header)

        stemmer = SnowballStemmer("english")
        lemmatizer = WordNetLemmatizer()

        for row in data_set:
            translated_row = [row[0], row[1]]
            translated_row.extend(row[2])

            if args.tokenize:
                translated_row = [nltk.word_tokenize(x) for x in translated_row]
                if args.stem:
                    translated_row = [[stemmer.stem(tok) for tok in sub] for sub in translated_row]
                if args.lemmatize:
                    translated_row = [[lemmatizer.lemmatize(tok, pos='v') for tok in sub] for sub in translated_row]

                translated_row = [" ".join(x) for x in translated_row]

            w.writerow(translated_row)
    print(f"Dataset stored in: {args.output}")


def train_cmd(args):
    rng = random.Random(args.seed)
    meta_path = os.path.join(os.path.dirname(__file__), "meta", "trainfiles.csv")
    with open(meta_path, 'r', encoding='utf-8') as f:
        dialog_paths = convert_csv_with_dialog_paths(f, args.data_root)

    train_set = create_examples(dialog_paths,
                                args.examples,
                                lambda context_dialog, candidates:
                                create_single_dialog_train_example(context_dialog, candidates, rng,
                                                                  args.p, max_context_length=args.max_context_length))

    stemmer = SnowballStemmer("english")
    lemmatizer = WordNetLemmatizer()

    with open(args.output, 'w', encoding='utf-8', newline='') as outf:
        w = csv.writer(outf)
        w.writerow(["Context", "Utterance", "Label"])
        for row in train_set:
            translated_row = list(row)
            if args.tokenize:
                translated_row = [nltk.word_tokenize(row[i]) for i in [0, 1]]
                if args.stem:
                    translated_row = [[stemmer.stem(tok) for tok in sub] for sub in translated_row]
                if args.lemmatize:
                    translated_row = [[lemmatizer.lemmatize(tok, pos='v') for tok in sub] for sub in translated_row]

                translated_row = [" ".join(x) for x in translated_row]
                translated_row.append(int(float(row[2])))

            w.writerow(translated_row)
    print(f"Train dataset stored in: {args.output}")


def valid_cmd(args):
    create_eval_dataset(args, "valfiles.csv")


def test_cmd(args):
    create_eval_dataset(args, "testfiles.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Script that creates train, valid and test set from 1 on 1 dialogs in Ubuntu Corpus.")

    parser.add_argument('--data_root', default='..',
                        help='directory where 1on1 dialogs are stored (parent folder with dialogs folder)')

    parser.add_argument('--seed', type=int, default=1234,
                        help='seed for random number generator')

    parser.add_argument('--max_context_length', type=int, default=20,
                        help='maximum number of dialog turns in the context')

    parser.add_argument('-o', '--output', default=None,
                        help='output csv')

    parser.add_argument('-t', '--tokenize', action='store_true',
                        help='tokenize the output')

    parser.add_argument('-l', '--lemmatize', action='store_true',
                        help='lemmatize the output by nltk.stem.WordNetLemmatizer (applied only when -t flag is present)')

    parser.add_argument('-s', '--stem', action='store_true',
                        help='stem the output by nltk.stem.SnowballStemmer (applied only when -t flag is present)')

    subparsers = parser.add_subparsers(help='sub-command help')

    parser_train = subparsers.add_parser('train', help='trainset generator')
    parser_train.add_argument('-p', type=float, default=0.5, help='positive example probability')
    parser_train.add_argument('-e', '--examples', type=int, default=1000000, help='number of examples to generate')
    parser_train.set_defaults(func=train_cmd)

    parser_test = subparsers.add_parser('test', help='testset generator')
    parser_test.add_argument('-n', type=int, default=9, help='number of distractor examples for each context')
    parser_test.set_defaults(func=test_cmd)

    parser_valid = subparsers.add_parser('valid', help='validset generator')
    parser_valid.add_argument('-n', type=int, default=9, help='number of distractor examples for each context')
    parser_valid.set_defaults(func=valid_cmd)

    args = parser.parse_args()

    prepare_data_maybe_download(args.data_root)

    args.func(args)
