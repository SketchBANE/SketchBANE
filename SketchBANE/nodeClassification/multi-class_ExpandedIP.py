import numpy as np
from scipy.sparse import random,csc_matrix,hstack
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score,accuracy_score
from scipy.sparse import save_npz, load_npz
from sklearn.model_selection import KFold
import argparse
import random

# multi-class classification
def predict_cv(X, y, C=1., num_workers=1):
    micro, macro, accuracy = [], [], []
    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = OneVsRestClassifier(
                LogisticRegression(
                    C=C,
                    solver="liblinear",
                    multi_class="ovr"),
                n_jobs=num_workers)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        mi = f1_score(y_test, y_pred, average="micro")
        ma = f1_score(y_test, y_pred, average="macro")
        micro.append(mi)
        macro.append(ma)
        accuracy.append(acc)
    return accuracy, micro, macro

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process...')
    parser.add_argument('--data', type=str, help='graph dataset name')
    parser.add_argument('--K', type=int, help='embedding dimensionality')
    parser.add_argument('--T', type=int, help='number of iterations')
    args = parser.parse_args()

    dataset = args.data
    T = args.T
    K = args.K
    labels_path = '../data/' + dataset + '/labels.npy'
    labels = np.load(labels_path)

    accuracies = []
    micro_f1_scores = []
    macro_f1_scores = []

    embedding_path = '../results/' + dataset + '/embedding.T' + str(T) + '.K' + str(K) + '.npz'
    embedding = load_npz(embedding_path)
    embedding = hstack([embedding, csc_matrix(np.logical_not(embedding.toarray()))])
    embedding = csc_matrix(embedding)
    for iteration in range(1,6):
        subset_size = 60000  # Set the subset size as needed
        if dataset == 'ogbn-papers100M':
            non_negative_indices = np.where(labels != -1)[0]
            random_indices = random.sample(list(non_negative_indices), subset_size)
            random_indices = np.array(random_indices)
        else:
            random_indices = np.random.choice(embedding.shape[0], subset_size, replace=False)
        sub_embedding = embedding[random_indices]
        sub_labels = labels[random_indices]
        acc,micro,macro = predict_cv(sub_embedding,sub_labels)

        accuracies.extend(acc)
        micro_f1_scores.extend(micro)
        macro_f1_scores.extend(macro)

    average_accuracy = np.mean(accuracies)
    average_micro_f1 = np.mean(micro_f1_scores)
    average_macro_f1 = np.mean(macro_f1_scores)

    std_accuracy = np.std(accuracies)
    std_micro_f1 = np.std(micro_f1_scores)
    std_macro_f1 = np.std(macro_f1_scores)
    print(
        f'【T={T},K={K}】acc：{average_accuracy * 100:.4f}--std_accuracy：{std_accuracy * 100:.4f}--micro-f1: {average_micro_f1 * 100:.4f}--std_micro：{std_micro_f1 * 100:.4f}--macro-f1: {average_macro_f1 * 100:.4f}--std_macro：{std_macro_f1 * 100:.4f}')

    log_path = '../results/' + dataset + '/multi-class_ExpandedIP.log'
    with open(log_path, "a") as fout:
        fout.write("data: " + args.data + "\n")
        fout.write("dimensionality: " + str(args.K) + "\n")
        fout.write("iteration: " + str(args.T) + "\n")
        fout.write("accuracy: " + str(average_accuracy) + "---std_accuracy: " + str(std_accuracy) + "\n")
        fout.write("micro-f1: " + str(average_micro_f1) + "---std_micro-f1: " + str(std_micro_f1) + "\n")
        fout.write("macro-f1: " + str(average_macro_f1) + "---std_macro-f1: " + str(std_macro_f1) + "\n")
        fout.write("----------------------------------------------------------------------------\n")




