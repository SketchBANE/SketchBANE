import numpy as np
from scipy.sparse import csr_matrix, hstack, random
import argparse

def generate_sparse_matrix(m, K):
    density = 1 / np.sqrt(m)
    sparse_matrix = random(m, K, density=density, format='csr')
    sparse_matrix.data = sparse_matrix.data * 2 - 1
    return sparse_matrix

def write_array_to_file(array, filename):
    with open(filename, 'w') as f:
        for element in array:
            f.write(str(element) + ' ')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process...')
    parser.add_argument('--dataset', type=str, help='graph dataset name')
    parser.add_argument('--K', type=int, help='embedding dimensionality')
    args = parser.parse_args()
    dataset = args.dataset
    K = args.K

    print(f'--------------------------{dataset} Datasets-----------------------------------------------------')
    attrs_path = 'data/' + dataset + '/attrs/info.txt'
    with open(attrs_path, 'r') as file:
        data = file.read()
    data = [int(i) for i in data.split()]
    n, m, nnz = data
    print("n:", n)
    print("m:", m)
    print("nnz:", nnz)

    R = generate_sparse_matrix(m,K)
    print(R.todense())

    # Save data, indices, and indptr to the appropriate file
    write_array_to_file(R.data, f'data/{dataset}/SRMatrix/values.txt')
    write_array_to_file(R.indices, f'data/{dataset}/SRMatrix/indices.txt')
    write_array_to_file(R.indptr, f'data/{dataset}/SRMatrix/indptr.txt')

    # Save rows, cols, and nnz to info.txt
    with open(f'data/{dataset}/SRMatrix/info.txt', 'w') as f:
        f.write(f"{R.shape[0]} {R.shape[1]} {R.nnz}")

    print(f'R.data:{R.data}')
    print(f'R.indices:{R.indices}')
    print(f'R.indptr:{R.indptr}')