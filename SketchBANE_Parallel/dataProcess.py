from scipy.sparse import csr_matrix
import numpy as np
import argparse
from scipy.sparse import save_npz, load_npz
import argparse

def write_array_to_file(array, filename):
    with open(filename, 'w') as f:
        for element in array:
            f.write(str(element) + ' ')

def save(csr_matrix, dataset, data_type):
    write_array_to_file(csr_matrix.data, f'data/{dataset}/{data_type}/values.txt')
    write_array_to_file(csr_matrix.indices, f'data/{dataset}/{data_type}/indices.txt')
    write_array_to_file(csr_matrix.indptr, f'data/{dataset}/{data_type}/indptr.txt')

    # Save rows, cols, and nnz to info.txt
    with open(f'data/{dataset}/{data_type}/info.txt', 'w') as f:
        f.write(f"{csr_matrix.shape[0]} {csr_matrix.shape[1]} {csr_matrix.nnz}")

    print(f'matrix.data:{csr_matrix.data}')
    print(f'matrix.indices:{csr_matrix.indices}')
    print(f'matrix.indptr:{csr_matrix.indptr}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process...')
    parser.add_argument('--dataset', type=str, help='graph dataset name')
    args = parser.parse_args()
    dataset = args.dataset

    print(f'--------------------------{dataset} Datasets-----------------------------------------------------')
    adjacency_path = 'data/' + dataset + '/network.npz'
    attribute_path = 'data/' + dataset + '/attrs.npz'
    attribute_matrix = load_npz(attribute_path)
    adjacency_matrix = load_npz(adjacency_path)
    attribute_matrix = csr_matrix(attribute_matrix)
    adjacency_matrix = csr_matrix(adjacency_matrix)

    n, m = adjacency_matrix.shape
    # Create the identity matrix corresponding to the adjacency matrix
    identity_matrix = csr_matrix((n, m), dtype=int)
    # Set diagonal elements to 1
    identity_matrix.setdiag(1)
    print(f'adjacency_matrix：{type(adjacency_matrix)}---attribute_matrix：{type(attribute_matrix)}---identity_matrix：{type(identity_matrix)}')
    save(adjacency_matrix, dataset, 'network')
    save(attribute_matrix, dataset, 'attrs')
    save(identity_matrix, dataset, 'IMatrix')

