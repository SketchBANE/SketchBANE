#pragma once
#include <cstdint>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <vector>
#include <chrono>
#include <memory>
#include <stdio.h>
#include <iomanip>
#include "mkl_spblas.h"
#include "mkl.h"
#include "mkl_cblas.h"

double saveEmb(const std::string& filename, sparse_matrix_t csrA, double runtime) {
    sparse_index_base_t indexing;
    MKL_INT nrows;
    MKL_INT ncols;
    MKL_INT* col_indptr_start;
    MKL_INT* col_indptr_end;
    MKL_INT* indices;
    float* csr_values;
    mkl_sparse_s_export_csr(csrA, &indexing, &nrows, &ncols, &col_indptr_start, &col_indptr_end, &indices, &csr_values);

    std::string indices_path = filename + "/indices.txt";
    std::string indptr_path = filename + "/indptr.txt";
    std::string values_path = filename + "/values.txt";
    std::string info_path = filename + "/info.txt";
    std::string runtime_path = filename + "/runtime.txt";

    // Create a file flow object
    std::ofstream indicesFile(indices_path);
    std::ofstream indptrFile(indptr_path);
    std::ofstream valuesFile(values_path);
    std::ofstream infoFile(info_path);
    std::ofstream runtimeFile(runtime_path);

    for (int i = 0; i <= nrows; i++) {
        indptrFile << col_indptr_start[i] << (i < nrows ? " " : "");
    }
    indptrFile << std::endl;

    int actual_nnz = col_indptr_start[nrows]; // The value of the nrows position represents the actual number of non-zero elements
    // Record the time when quantization began
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < actual_nnz; i++) {
        indicesFile << indices[i] << (i < actual_nnz - 1 ? " " : "");
        bool value = csr_values[i] > 0; // Change to "true" if it is greater than 0, otherwise change to "false"
        valuesFile << value << (i < actual_nnz - 1 ? " " : "");
    }
     // Record the end time point of quantization
    auto end = std::chrono::high_resolution_clock::now();
    // Calculate the time difference and convert it to seconds
    std::chrono::duration<double> duration = end - start;
    double seconds = duration.count();
    seconds = seconds + runtime;
    std::cout << "final runtime of SketchBANE_CPP2 " << seconds << " seconds." << std::endl;

    indicesFile << std::endl;
    valuesFile << std::endl;

    // Write the final time to the runtime.txt file
    runtimeFile << seconds << " (s)" << std::endl;
    // Save the number of rows, columns, and non-zero elements to the info.txt file
    infoFile << nrows << " " << ncols << " " << actual_nnz << std::endl;

    // Close the files
    indicesFile.close();
    indptrFile.close();
    valuesFile.close();
    infoFile.close();
    return seconds;
}

void printCsrMatrixValues(sparse_matrix_t csrA) {
    // Get matrix information and data from csrA
    sparse_index_base_t indexing;
    MKL_INT rows, cols;
    MKL_INT *rows_start, *rows_end, *columns;
    float *values;
    sparse_status_t status = mkl_sparse_s_export_csr(csrA, &indexing, &rows, &cols, &rows_start, &rows_end, &columns, &values);

    if (status != SPARSE_STATUS_SUCCESS) {
        std::cerr << "Failed to export CSR matrix." << std::endl;
        return;
    }
    // Prints each non-zero value and its column and column subscript
    std::cout << "Row, Column, Value:" << std::endl;
    for (MKL_INT row = 0; row < rows; ++row) {
        for (MKL_INT idx = rows_start[row]; idx < rows_start[row + 1]; ++idx) {
            MKL_INT col = columns[idx]; // Get column index
            float value = values[idx];  // Gets the corresponding value
            std::cout << row << ", " << col << ", " << value << std::endl;
        }
    }
}

int readValue(const std::string& filename, std::vector<float>& values) {
    std::ifstream infile(filename);
    float val;
    if (!infile.is_open()) {
        std::cerr << "Unable to open file for reading." << std::endl;
        return 1;
    }
    values.clear();
    while (infile >> val) {
        values.push_back(val);
    }
    infile.close();
    return 0;
}
int readIndicesOrIndptr(const std::string& filename, std::vector<MKL_INT>& col_indx) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Unable to open file for reading." << std::endl;
        return 1;
    }
    col_indx.clear(); // Make sure the vector is empty
    MKL_INT val;
    while (infile >> val) {
        col_indx.push_back(val);
    }
    infile.close();
    return 0;
}
void readMatrixInfo(const std::string& filename, MKL_INT& rows, MKL_INT& cols, MKL_INT& nnz) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    // Read the number of rows, columns, and non-zero elements directly
    if (!(file >> rows >> cols >> nnz)) {
        std::cerr << "Failed to read matrix info from file: " << filename << std::endl;
        return;
    }
    file.close();
}

int main(int argc, char* argv[]) {
    std::string dataset;
    int T = 0; // The default value is 0

    for (int i = 1; i < argc; ++i) { // Start with 1, because argv[0] is the program name
        std::string arg = argv[i];
        if (arg == "--dataset") {
            if (i + 1 < argc) { // Make sure you have the next parameter as the value
                dataset = argv[++i]; // Gets the value and adds the index
            } else {
                std::cerr << "--dataset option requires one argument." << std::endl;
                return 1;
            }
        } else if (arg == "--T") {
            if (i + 1 < argc) {
                T = std::atoi(argv[++i]);
            } else {
                std::cerr << "--T option requires one argument." << std::endl;
                return 1;
            }
        }
    }
    // Output the obtained parameters and verify that they are correct
    std::cout << "Dataset: " << dataset << std::endl;
    std::cout << "T: " << T << std::endl;

    MKL_INT rowsG, colsG, nnzG, rowsA, colsA, nnzA, rowsI, colsI, nnzI, rowsR, colsR, nnzR;
    readMatrixInfo("data/"+ dataset +"/network/info.txt", rowsG, colsG, nnzG);
    std::cout << "nG:"<<rowsG<<" mG:"<<colsG<<" nnzG:"<<nnzG<< std::endl;

    sparse_matrix_t G, A, I, R;//G represents an adjacency matrix, and G1 represents an adjacency matrix with a closed neighborhood
    sparse_status_t status;

    std::vector<float> valuesG, valuesA, valuesI, valuesR;
    std::vector<MKL_INT> indicesG, indicesA, indicesI, indicesR;
    std::vector<MKL_INT> indptrG, indptrA, indptrI, indptrR;

    // Read the data and create the adjacency matrix
    readValue("data/" + dataset + "/network/values.txt", valuesG);
    readIndicesOrIndptr("data/"+ dataset +"/network/indices.txt", indicesG);
    readIndicesOrIndptr("data/"+ dataset +"/network/indptr.txt", indptrG);
    status = mkl_sparse_s_create_csr(&G, SPARSE_INDEX_BASE_ZERO, rowsG, colsG, indptrG.data(), indptrG.data() + 1, indicesG.data(), valuesG.data());
    if (status != SPARSE_STATUS_SUCCESS) {
        printf("Error creating CSC matrix.\n");
        return 1;
    }
    printf("Succeeded in creating G:\n");

    // Read the data and create the attributes matrix
    readMatrixInfo("data/"+ dataset +"/attrs/info.txt", rowsA, colsA, nnzA);
    std::cout << "nA:"<<rowsA<<" mA:"<<colsA<<" nnzA:"<<nnzA<< std::endl;
    readValue("data/"+ dataset +"/attrs/values.txt",valuesA);
    readIndicesOrIndptr("data/"+ dataset +"/attrs/indices.txt",indicesA);
    readIndicesOrIndptr("data/"+ dataset +"/attrs/indptr.txt",indptrA);
    status = mkl_sparse_s_create_csr(&A, SPARSE_INDEX_BASE_ZERO, rowsA, colsA, indptrA.data(), indptrA.data() + 1, indicesA.data(), valuesA.data());
    if (status != SPARSE_STATUS_SUCCESS) {
        printf("Error creating CSC matrix.\n");
        return 1;
    }
    printf("Succeeded in creating A:\n");

    // Read the data and create the Identity matrix
    readMatrixInfo("data/"+ dataset +"/IMatrix/info.txt", rowsI, colsI, nnzI);
    readValue("data/"+ dataset +"/IMatrix/values.txt",valuesI);
    readIndicesOrIndptr("data/"+ dataset +"/IMatrix/indices.txt",indicesI);
    readIndicesOrIndptr("data/"+ dataset +"/IMatrix/indptr.txt",indptrI);
    status = mkl_sparse_s_create_csr(&I, SPARSE_INDEX_BASE_ZERO, rowsI, colsI, indptrI.data(), indptrI.data() + 1, indicesI.data(), valuesI.data());
    if (status != SPARSE_STATUS_SUCCESS) {
        printf("Error creating CSC matrix.\n");
        return 1;
    }
    printf("Succeeded in creating I:\n");

    // Read the data and create the Sparse Random matrix
    readMatrixInfo("data/"+ dataset +"/SRMatrix/info.txt", rowsR, colsR, nnzR);
    readValue("data/"+ dataset +"/SRMatrix/values.txt",valuesR);
    readIndicesOrIndptr("data/"+ dataset +"/SRMatrix/indices.txt",indicesR);
    readIndicesOrIndptr("data/"+ dataset +"/SRMatrix/indptr.txt",indptrR);
    status = mkl_sparse_s_create_csr(&R, SPARSE_INDEX_BASE_ZERO, rowsR, colsR, indptrR.data(), indptrR.data() + 1, indicesR.data(), valuesR.data());
    if (status != SPARSE_STATUS_SUCCESS) {
        printf("Error creating CSC matrix.\n");
        return 1;
    }
    printf("Succeeded in creating R:\n");

    float alpha = 1.0; // Scalar multiple factor of addition

    // Record the start time
    auto start = std::chrono::high_resolution_clock::now();
    // Construct the adjacency matrix of the closed neighborhood
    mkl_sparse_s_add(SPARSE_OPERATION_NON_TRANSPOSE, G, alpha, I, &G);
    sparse_matrix_t H;
    mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, A, R, &H);
    for(int i=0;i<T;i++){
        mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, G, H, &H);
        printf("Values of the %d res:\n",i+1);
    }
    // Record end time point
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double seconds = duration.count();
    std::cout << "embedding time of SketchBANE_CPP " << seconds << " seconds." << std::endl;

    std::string emb_path = "emb/" + dataset;
    std::cout << "Save embedding...... " << std::endl;
    double finalRuntime = saveEmb(emb_path, H, seconds);
    std::cout << "final runtime of SketchBANE_CPP " << finalRuntime << " seconds." << std::endl;

    mkl_sparse_destroy(G);
    mkl_sparse_destroy(R);
    mkl_sparse_destroy(A);
    mkl_sparse_destroy(I);
    mkl_sparse_destroy(H);
    return 0;
}

