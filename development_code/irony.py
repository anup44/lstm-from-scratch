import emoji
from util import *
from bayes import NaiveBayes


#### READ DATASET ####

SEM_EVAL_FOLDER = 'SemEval2018-Task3'
TRAIN_FILE = SEM_EVAL_FOLDER + '/datasets/train/SemEval2018-T3-train-taskA_emoji.txt'
TEST_FILE = SEM_EVAL_FOLDER + '/datasets/goldtest_TaskA/SemEval2018-T3_gold_test_taskA_emoji.txt'


def download_irony():
    if os.path.exists(SEM_EVAL_FOLDER):
        return
    else:
        try:
            git('clone', 'https://github.com/Cyvhee/SemEval2018-Task3.git')
            return
        except OSError:
            # Do nothing. Continue try downloading zip file
            pass

    print('Downloading dataset')
    download_zip('https://github.com/Cyvhee/SemEval2018-Task3/archive/master.zip', SEM_EVAL_FOLDER)


def read_dataset_file(file_path):
    with open(file_path, 'r', encoding='utf8') as ff:
        rows = [line.strip().split('\t') for line in ff.readlines()[1:]]
        _, labels, texts = zip(*rows)
    clean_texts = [emoji.demojize(tex) for tex in texts]
    unique_labels = sorted(set(labels))
    lab2i = {lab: i for i, lab in enumerate(unique_labels)}
    return clean_texts, labels, lab2i


def load_datasets():
    # download dataset from git
    download_irony()

    # read the datasets
    train_texts, train_labels, label2i = read_dataset_file(TRAIN_FILE)
    test_texts, test_labels, _ = read_dataset_file(TEST_FILE)

    return train_texts, train_labels, test_texts, test_labels, label2i


def all_predicts(test_preds, test_labs):
    print('Accuracy:', accuracy(np.array(test_preds), np.array(test_labs)))
    print('Precision:', precision(np.array(test_preds), np.array(test_labs), which_label='1'))
    print('Recall:', recall(np.array(test_preds), np.array(test_labs), which_label='1'))
    print('F1-score:', f1_score(np.array(test_preds), np.array(test_labs), which_label='1'))


def run_nb_baseline():
    train_t, train_labels, test_t, test_labels, label2i = load_datasets()
    train_labels, test_labels = [label2i[l] for l in train_labels], [label2i[l] for l in test_labels]

    train_t_processed = [t.split() for t in train_t]
    test_t_processed = [t.split() for t in test_t]
    #
    # ### Baseline: Naive Bayes ###
    nb = NaiveBayes()
    nb.fit(train_t_processed, train_labels)
    _, train_probs = nb.predict(train_t_processed)

    t_predictions, _ = nb.predict(test_t_processed)

    print('Baseline: Naive Bayes Classifier')
    # '1' represent the ironic class
    print('F1-score Ironic:', f1_score(t_predictions, test_labels, label2i['1']))

    # average f1 score of ironic and non-ironic class
    print('Avg F1-score:', avg_f1_score(t_predictions, test_labels, list(label2i.values())))


if __name__ == '__main__':
    run_nb_baseline()