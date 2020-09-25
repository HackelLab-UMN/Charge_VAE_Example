'''
Driver script that trains the model.
'''

import m1_vae
import pandas as pd 

#load data
df_train=pd.read_pickle('./charge_training_dataset.pkl')
df_test=pd.read_pickle('./charge_testing_dataset.pkl')
seq_len=10 #assumes sequences alligned&padded

#initialize model
unsupervised_vae=m1_vae.initalize_model(seq_len=seq_len)

#train model
print('Training')
unsupervised_vae=m1_vae.train_VAE(unsupervised_vae,df_train,df_test)
print('Done Training')
#print samples from latent space:
z0_low=-1
z0_high=1
z1_low=-0.05
z1_high=0.15
m1_vae.sample_latent_space(unsupervised_vae,z0_low,z0_high,z1_low,z1_high)