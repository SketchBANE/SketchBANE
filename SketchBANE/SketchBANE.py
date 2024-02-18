from sklearn import random_projection
import gc
import numpy as np
from scipy.sparse import save_npz, load_npz
import time
import argparse
import os

def Projection(G, A, q, K):
    '''
    Inputs:
        G: sparse adjacency matrix
        A: sparse attributes matrix
        q: order
        K: the dimension of network embedding
    Outputs:
        U: G^q * A * R
    '''

    G.setdiag(1) # generate self-loop matrix
    transformer = random_projection.SparseRandomProjection(n_components=K, density='auto')
    A = transformer.fit_transform(A)   # A * R

    del transformer
    gc.collect()

    for i in range(1, q + 1):  # iterative random projection
        A = G @ A

    del G
    gc.collect()
    A = A > 0  # Quantization of the embedding matrix
    return A

def getEmbedding(T, K=200):
    start_time = time.time()
    embedding = Projection(adjacency_matrix,attribute_matrix,T,K)
    embedding_time = time.time() - start_time
    return embedding,embedding_time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process...')
    parser.add_argument('--data', type=str, help='graph dataset name')
    parser.add_argument('--K', type=int, help='embedding dimensionality')
    parser.add_argument('--T', type=int, help='number of iterations')

    args = parser.parse_args()

    dataset = args.data
    T = args.T
    K = args.K

    print(f'--------------------------{dataset} Datasets-----------------------------------------------------')
    adjacency_path = 'data/' + dataset + '/network.npz'
    attribute_path = 'data/' + dataset + '/attrs.npz'
    attribute_matrix = load_npz(attribute_path)
    adjacency_matrix = load_npz(adjacency_path)

    embedding, embedding_time = getEmbedding(T, K)
    print(f'【T={T}--K={K}】runtime：{embedding_time:.6f}s')

    folder_path = 'results/' + f'{dataset}'
    try:
        os.mkdir(folder_path)
        print("Successfully created a folder！")
    except FileExistsError:
        print("The folder already exists！")
    except Exception as e:
        print("An error occurred：", str(e))

    log_path = folder_path + '/runtime.log'

    with open(log_path, "a") as fout:
        fout.write("data: "+args.data+"\n")
        fout.write("dimensionality: "+str(args.K)+"\n")
        fout.write("iteration: "+str(args.T)+"\n")
        fout.write("embedding time(s): "+str(embedding_time)+"\n")
        fout.write("----------------------------------------------------------------------------\n")

    emb_path = folder_path + '/embedding.T' + str(T) + '.K' + str(K) + '.npz'

    print(f'save embedding......')
    save_npz(emb_path, embedding)
