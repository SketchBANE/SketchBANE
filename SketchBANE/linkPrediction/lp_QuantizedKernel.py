import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz, load_npz
from scipy.io import loadmat, savemat
import argparse

def linkPredicton(dataset,T,K):
    ratio = [0.5, 0.6, 0.7, 0.8, 0.9]
    np.random.seed(42)  # Set a seed for reproducibility
    network_dir = '../datas/' + dataset + '/network.npz'
    network = load_npz(network_dir)

    nodeNum = network.shape[0]
    network.setdiag(0)

    count = 1
    times = 10000
    nonexistence = np.zeros((2, times), dtype=int)
    np.random.seed(0)
    while count <= times:
        edgeIds = np.random.randint(0, nodeNum, size=(2, 1))
        if network[edgeIds[0], edgeIds[1]] == 0:
            nonexistence[:, count - 1] = edgeIds.flatten()
            count += 1

    accuracy = []
    emb_path = '../results/' + dataset + '/embedding.T' + str(T) + '.K' + str(K) + '.npz'
    embedding = load_npz(emb_path)
    embedding.setdiag(0)
    embedding = embedding.astype(int)

    for dense in range(1, 6):
        if dataset == 'Amazon' or dataset == 'ogbn-papers100M':
            file_name2 = "../datas/" + dataset + "/lp/" + dataset + "_trainGraph_" + str(
                ratio[dense - 1]) + ".npz"
            file_name3 = "../datas/" + dataset + "/lp/" + dataset + "_testGraph_" + str(
                ratio[dense - 1]) + ".npz"
            trainGraph = load_npz(file_name2)
            testGraph = load_npz(file_name3)
        else:
            file_name2 = "../datas/" + dataset + "/lp/" + dataset + "_" + str(ratio[dense - 1]) + ".mat"
            f = loadmat(file_name2)
            trainGraph = f['trainGraph']
            testGraph = f['testGraph']

        trainGraph.setdiag(0)
        nonexistence_similarity = np.sum(
            embedding[nonexistence[0], :] == embedding[nonexistence[1], :], axis=1
        )

        testGraph.setdiag(0)
        i_test, j_test = testGraph.nonzero()

        testedEdges = np.column_stack((i_test, j_test))
        testedEdges = testedEdges[testedEdges[:, 0] > testedEdges[:, 1]]

        selected_edges = np.random.choice(testedEdges.shape[0], times, replace=True)
        testedEdges = testedEdges[selected_edges, :]

        missing_similarity = np.sum(
            embedding[testedEdges[:, 0], :] == embedding[testedEdges[:, 1], :], axis=1
        )

        greatNum = np.sum(missing_similarity > nonexistence_similarity)
        equalNum = np.sum(missing_similarity == nonexistence_similarity)

        auc = (greatNum + 0.5 * equalNum) / times
        accuracy.append(auc)
        print(f'T={T}--K={K}:auc:{auc}')

    average_accuracy = np.mean(accuracy)
    variance_accuracy = np.var(accuracy)
    print(f'auc:{average_accuracy * 100}---var_auc:{variance_accuracy * 10000}')

    log_path = '../results/' + dataset + '/lp_QuantizedKernel.log'
    with open(log_path, "a") as fout:
        fout.write("data: " + args.data + "\n")
        fout.write("dimensionality: " + str(args.K) + "\n")
        fout.write("iteration: " + str(args.T) + "\n")
        fout.write("auc: " + str(average_accuracy) + "---var_auc: " + str(variance_accuracy) + "\n")
        fout.write("----------------------------------------------------------------------------\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process...')
    parser.add_argument('--data', type=str, help='graph dataset name')
    parser.add_argument('--K', type=int, help='embedding dimensionality')
    parser.add_argument('--T', type=int, help='number of iterations')

    args = parser.parse_args()

    dataset = args.data
    T = args.T
    K = args.K
    linkPredicton(dataset,T,K)










