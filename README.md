# Charge_VAE_Example
Author: Alex Golinski
golin010@umn.edu

Toy model to see if a VAE can generate sequences with a given charge.

Currently an unsupervised seq-to-seq VAE is implemented. 

datasets.py creates a dataset of sequences from 2 classes:
Positive- enriched in H, K, R
Negative- enriched in D, E

m1_vae.py describes the functions/object of the unsupervised VAE
plot_functions.py is a helper file for plotting

charge_driver.py is the main file to run to generate results:
Example: > python3 charge_driver.py 
This will train and sample the VAE automatically. 

Conda environment is saved as package-list.txt (open to see install instructions). 
