## Description
The datasets and source code of SketchBANE_CPP are for Time- and Space-Efficiently Sketching Billion-Scale Attributed Networks.

## Install Intel MKL
Intel MKL is used for basic linear algebra operations.
You can download the mkl library online according to the official website
https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html?operatingsystem=linux&distributions=offline

Remember the intel installation path {intel_PATH}

# Compile

To compile SketchBANE_CPP, you may need to edit Makefile when you install MKL. You need to set something like:
```
INCLUDE_DIRS = -I./ligra -I./pbbslib -I./mklfreigs -I"{ANACONDA_PATH}/envs/SketchBANE_CPP/include" -I"{intel_PATH}/oneapi/mkl/2023.2.0/include"
LINK_DIRS = -L"{ANACONDA_PATH}/envs/SketchBANE_CPP/lib" -L"{intel_PATH}/oneapi/mkl/2023.2.0/lib/intel64"
```
To clean the compiled file, run `make clean`.
Then run `make` to compile.

## Data download
ogbn-product can be downloaded from [here](https://v50tome-my.sharepoint.com/:f:/g/personal/sketchbane_v50tome_onmicrosoft_com/EkMKM8-9Mz5FpA_weM7hp_EBvzu7qRdtvzQnpPHOarnYDA).  
Amazon can be downloaded from [here](https://v50tome-my.sharepoint.com/:f:/g/personal/sketchbane_v50tome_onmicrosoft_com/EmHKeYP9bNBNk0hVYc83OO4BUglv5uaOT__bYCtfHpbA9g).  
MAG-Scholar-C can be downloaded from [here](https://v50tome-my.sharepoint.com/:f:/g/personal/sketchbane_v50tome_onmicrosoft_com/EgV5_AHINy9HiEysga07hwEB1HS0nr5UKUiUY14UmvWEgA).  
ogbn-papers100M can be downloaded from [here](https://v50tome-my.sharepoint.com/:f:/g/personal/sketchbane_v50tome_onmicrosoft_com/EtrAbfCZCStEtNZ-1wiXhRsB_omuhfc2VXfp4YCC4P6UQA).
```
data
├── Amazon       
├── MAG-Scholar-C  
├── ogbn-papers100M  
├── ogbn-product 
```
## Data Prcocess
Create four folders for each data set under data, such as ogbn-product. With the data download from the previous step, the ogbn-product directory should have three files and four folders plus an lp folder, as well as the other data sets.
```
data  
├── ogbn-product ├── attrs      # Store attribute matrix information  
                 ├── IMatrix    # Store identity matrix information
                 ├── network    # Store adjancy matrix information
                 ├── SRMatix    # Store sparse random matrix information
                 ├── lp         # Data used to link predictions
                 ├── attrs.np    # attributes matrix
                 ├── labels.npy    # label matrix
                 ├── network.npz    # adjancy matrix
```
```
cd SketchBANE_CPP
python dataProcess.py --K 200 --T 4 --dataset ogbn-product   
```

## Generate embeddings using SketchBANE_CPP
```
cd SketchBANE_CPP
makedir emb  #Create a folder to store embeddings information. 
cd emb
makedir ogbn-product   #Create a folder to store embedding information for the dataset.
python SparseRandomMatrix.py --dataset ogbn-product --K 200   # Generate sparse random matrix   
./main --dataset ogbn-product --T 4
```

## Get embeddings
```
mkdir results  #Create a folder to store embeddings in Python format within the directory "SketchBANE_CPP".
python getEmb.py --dataset ogbn-product --K 200 --T 4  # Generate sparse embedding matrices in python format   
```

## Node Classification
```
cd nodeClassification
python multi-class.py --dataset ogbn-product --K 200 --T 4     # multi-class classification using inner products
python multi-class_ExpandedIP.py --dataset ogbn-product --K 200 --T 4     # multi-class classification using expanded inner products
python multi-label.py --dataset Amazon --K 200 --T 1     # multi-label classification using inner products
python multi-label_ExpandedIP.py --dataset Amazon --K 200 --T 1      # multi-label classification using expanded inner products
```

## Link Prediction
```
cd linkPrediction
python lp_InnerProduct.py --dataset ogbn-product --K 200 --T 4     # link prediction using inner products
python lp_ExpandedIP.py --dataset ogbn-product --K 200 --T 4     # link prediction using expanded inner products
python lp_QuantizedKernel.py --dataset ogbn-product --K 200 --T 4    # link prediction using  quantized kernel
```
