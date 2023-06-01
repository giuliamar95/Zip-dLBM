# Zip-dLBM
This GitHub repository contains the code implementation for the dynamic co-clustering approach proposed in the article "A Deep Dynamic Latent Block Model for the Co-clustering of Zero-Inflated Data Matrices”.

## Authors: Giulia Marchello, Benjamin Navet, Marco Corneli, Charles Bouveyron.

## Start project
- conda create -n Zip-dLBM
- conda activate Zip-dLBM
- conda remove mkl    (Only for MacOs users)  
- conda install nomkl (Only for MacOs users)  
- conda install numpy 
- conda install  scipy scikit-learn matplotlib pandas
- conda install pytorch torchvision torchaudio -c pytorch (alternatively: pip3 install torch)
- conda install -c conda-forge r-base r-slam r-rlab r-reticulate
- pip install line-profiler 
- conda install pathos 
- Rscript_Exp1.R (or Rscript_Real.R for London's Bike data)
