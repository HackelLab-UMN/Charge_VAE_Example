'''
This set of functions creates, loads, encodes, and saves DataFrames
of each sequence.
Pos: H(6), K(8), R(14)
Neg: D(2), E(3)
'''

import numpy as np 
import pandas as pd 
from sklearn import preprocessing

import plot_functions

def ordinal_decode(seq):
	'ordinal to amino acid sequence'
	AAlist=np.array(list("ACDEFGHIKLMNPQRSTVWY")) 
	enc_OH=preprocessing.OrdinalEncoder().fit(AAlist.reshape(-1,1))
	AAlist=enc_OH.inverse_transform(seq.reshape(-1, 1)).flatten()
	AA_sequence_list=''.join(AAlist)
	return AA_sequence_list

def get_charge(seq):
	'return # pos and # neg AA in seq'
	seq=np.array(seq)
	n_pos=sum(np.where((seq==6)|(seq==8)|(seq==14),1,0))
	n_neg=sum(np.where((seq==2)|(seq==3),1,0))
	return n_pos, n_neg

def make_datasets(n_samples=1000,seq_len=10):
	'Pos: 0-19 stay, 20-29: H, 30-39:K, 40-49:R'
	pos_data=np.random.randint(low=0,high=49,size=[n_samples,seq_len])
	pos_list=[[]]*len(pos_data)

	for i, seq in enumerate(pos_data):
		seq_adj=np.where(((seq>=20)&(seq<30)),6,seq)
		seq_adj=np.where(((seq_adj>=30)&(seq_adj<40)),8,seq_adj)
		seq_adj=np.where(((seq_adj>=40)&(seq_adj<50)),14,seq_adj)
		AA_seq=ordinal_decode(seq_adj)
		n_pos,n_neg=get_charge(seq_adj)
		pos_list[i]=[list(seq_adj),AA_seq,n_pos,n_neg]


	pos_df=pd.DataFrame(pos_list)
	pos_df.columns=['Ordinal','AA','N_Pos','N_Neg']
	pos_df['Class']='Positive'

	'Neg: 0-19 stay, 20-29:D, 30-39:E'
	neg_data=np.random.randint(low=0,high=39,size=[n_samples,seq_len])
	neg_list=[[]]*len(neg_data)
	for i, seq in enumerate(neg_data):
		seq_adj=np.where(((seq>=20)&(seq<30)),2,seq)
		seq_adj=np.where(((seq_adj>=30)&(seq_adj<40)),3,seq_adj)
		AA_seq=ordinal_decode(seq_adj)
		n_pos,n_neg=get_charge(seq_adj)
		neg_list[i]=[list(seq_adj),AA_seq,n_pos,n_neg]

	neg_df=pd.DataFrame(neg_list)
	neg_df.columns=['Ordinal','AA','N_Pos','N_Neg']
	neg_df['Class']='Negative'

	df=pd.concat([pos_df,neg_df],ignore_index=True)
	return df

def plot_charge_distributions():
	df=pd.read_pickle('./charge_testing_dataset.pkl')
	plot_functions.plot(savename='charge_test_dist',data=df,x='N_Neg',x_label='# of Neg AA in Sequence',y='N_Pos',y_label='# of Pos AA in Sequence',hue='Class')


def main():
	'Make a dataset for VAE training'
	df=make_datasets(n_samples=10000)
	df.to_pickle('charge_training_dataset.pkl')

	'Make a dataset for VAE testing'
	df=make_datasets(n_samples=2000)
	df.to_pickle('charge_testing_dataset.pkl')

	'Plot the charge distribution for the test sequences to compare to latent space'
	plot_charge_distributions()

if __name__ == '__main__':
	main()


