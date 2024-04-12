import numpy as np
from scipy.sparse import csr_matrix,csc_matrix
from scipy.sparse import save_npz, load_npz
import argparse
import os

def read_data_from_file(filename):
    with open(filename, 'r') as file:
        data = file.read()
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process...')
    parser.add_argument('--dataset', type=str, help='graph dataset name')
    parser.add_argument('--K', type=int, help='embedding dimensionality')
    parser.add_argument('--T', type=int, help='number of iterations')
    args = parser.parse_args()

    dataset = args.dataset
    T = args.T
    K = args.K

    data_path = 'emb/' + dataset + '/values.txt'
    indices_path = 'emb/' + dataset + '/indices.txt'
    indptr_path = 'emb/' + dataset + '/indptr.txt'
    info_path = 'emb/' + dataset + '/info.txt'

    data = read_data_from_file(data_path)
    data = [int(i)>0 for i in data.split()]

    indices = read_data_from_file(indices_path)
    indices = [int(i) for i in indices.split()]

    indptr = read_data_from_file(indptr_path)
    indptr = [int(i) for i in indptr.split()]

    info = read_data_from_file(info_path)
    info = [int(i) for i in info.split()]
    n, m, nnz = info

    print("n:", n)
    print("m:", m)
    print("nnz:", nnz)

    sparse_matrix = csr_matrix((data, indices, indptr), shape=(n, m))

    print(sparse_matrix.todense())
    print(sparse_matrix.dtype)
    sparse_matrix.eliminate_zeros()
    print(sparse_matrix.todense())

    folder_path = 'results/' + f'{dataset}'
    try:
        os.mkdir(folder_path)
        print("Successfully created a folder！")
    except FileExistsError:
        print("The folder already exists！")
    except Exception as e:
        print("An error occurred：", str(e))

    print('正在保存.....')
    f_path = f'results/'+ dataset +'/embedding.T' + str(T) +'.K' + str(K) + '.npz'
    save_npz(f_path,sparse_matrix)