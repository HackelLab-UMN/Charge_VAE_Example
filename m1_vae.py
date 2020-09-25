'''
Contains the tensorflow model and associated code for the 
m1 (unsupervised) VAE model. 
Class object defines model 
Functions after define training
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.random.set_seed(42)
import numpy as np 
import pandas as pd 
import time

import plot_functions
from datasets import ordinal_decode

class m1_VAE(tf.keras.Model):
	'underpervised VAE for charge model'

	def __init__(self, latent_dim, seq_len, beta):
		super(m1_VAE,self).__init__()
		self.latent_dim = latent_dim
		self.seq_len = seq_len
		self.beta= beta

		self.encoder = tf.keras.Sequential(
			[
				tf.keras.layers.Embedding(input_dim=20,output_dim=2),
				tf.keras.layers.Bidirectional(tf.keras.layers.GRU(10)),
				tf.keras.layers.Dense(latent_dim*2,activation=None) 
			]
		)

		self.decoder = tf.keras.Sequential(
			[
				tf.keras.layers.InputLayer(input_shape=(latent_dim)), #start with sample from latent space
				tf.keras.layers.RepeatVector(seq_len),
				tf.keras.layers.GRU(10, return_sequences=True),
				tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(20,activation=None))
			]
		)

	@tf.function
	def sample(self, eps=None,no_to_generate=10):
		if eps is None:
			eps = tf.random.normal(shape=(no_to_generate, self.latent_dim)) #default sample from normal distribution of latent space
		return self.decode(eps, apply_maxfilter=True)


	def encode(self, x):
		'convert input sequence to latent space, defined by a mean and variance'
		mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
		return mean, logvar

	def reparameterize(self, mean, logvar):
		'add noise to sample from normal distribution'
		eps = tf.random.normal(shape=mean.shape)
		return eps * tf.exp(logvar * .5) + mean

	def decode(self, z, apply_maxfilter=False):
		'convert sample of latent space to a new generated sequence'
		logits = self.decoder(z)
		if apply_maxfilter:
			ord_seq = tf.argmax(logits,axis=-1)
			return ord_seq
		return logits


def log_normal_pdf(sample, mean, logvar, raxis=1):
	'estimates sample probability from distribution'
	log2pi = tf.math.log(2. * np.pi)
	return tf.reduce_sum(
	  -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
	  axis=raxis)

def compute_loss(model, x, print_ind_loss=False):
	''' ELBO Loss (estimated by MC sampling). 
	P(sample given latent space)+p(from mu=0,var=1 latent space)-prop(latent space given sapmles)
	Frist term represents accuracy of recreating image from the cross-entropy.
	Second and 3rd term are 'regularization' terms, aiming to force all to normal distirbution.
	'''
	beta=model.beta
	mean, logvar = model.encode(x) #sequence to latent space
	z = model.reparameterize(mean, logvar) #sample from latent space
	x_logits=model.decode(z) #recreate sequence from sample of latent space

	#calculate cross_entropy of prediction (probably of sample given laten space)
	cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(x, x_logits)
	logpx_z = -tf.reduce_sum(cross_ent, axis=[1])

	#calculate difference in probabilty of sample from latent space  
	logpz = log_normal_pdf(z, 0., 0.) #prob of latent space observation in guassian 
	logqz_x = log_normal_pdf(z, mean, logvar) #prob of latent space observation given sample
	recon_loss=tf.reduce_mean(logpx_z)
	KL_loss=tf.reduce_mean(logpz - logqz_x)
	if print_ind_loss:
		print('Recon:')
		print(recon_loss)
		print('KL:')
		print(KL_loss)
	return -(recon_loss+beta*KL_loss) #insert beta here

@tf.function
def train_step(model,x,optimizer):
	"""Executes one training step and returns the loss.
	This function computes the loss and gradients, and uses the latter to
	update the model's parameters.	"""
	with tf.GradientTape() as tape:
		loss = compute_loss(model, x)
		gradients = tape.gradient(loss, model.trainable_variables)
		optimizer.apply_gradients(zip(gradients, model.trainable_variables))
	return loss


def find_mean_latent_space(model,df):
	'Finds the mean location in latent space for pos/neg, and generates sequence'
	
	df_local_pos=df[df['Class']=='Positive']
	pos_z0_mean=np.mean(df_local_pos['Z0'].values)
	pos_z1_mean=np.mean(df_local_pos['Z1'].values)
	pos_cord=np.array([pos_z0_mean,pos_z1_mean])

	df_local_neg=df[df['Class']=='Negative']
	neg_z0_mean=np.mean(df_local_neg['Z0'].values)
	neg_z1_mean=np.mean(df_local_neg['Z1'].values)
	neg_cord=np.array([neg_z0_mean,neg_z1_mean])

	in_array=np.array([pos_cord,neg_cord])
	samples=model.sample(in_array)

	print('avg positive',round(pos_z0_mean,3),round(pos_z1_mean,3),ordinal_decode(np.array(samples[0])))
	print('avg negative',round(neg_z0_mean,3),round(neg_z1_mean,3),ordinal_decode(np.array(samples[1])))



def plot_latent_distribution(model,epoch,df_test):
	'Plots the latent space of the test sequences at any epoch of training'
	encoder_input=df_test['Ordinal'].values
	encoder_input=np.stack(encoder_input).astype('int32')

	latent_mean , _ = model.encode(encoder_input) #plot just the mean latent value

	df_local=df_test.copy()
	df_local['Z0']=latent_mean[:,0]
	df_local['Z1']=latent_mean[:,1]

	plot_functions.plot(savename='./temp_images/M1VAE_E'+str(epoch),data=df_local,x='Z0',x_label='Z0',y='Z1',y_label='Z1',hue='Class')

	#find the averate of each class and generate sample sequence from that class
	find_mean_latent_space(model,df_local)

def sample_latent_space(model,z0_low,z0_high,z1_low,z1_high):
	'prints sequences generated from 3x3 grid in latent space'
	x=np.linspace(z0_low,z0_high,3)
	y=np.linspace(z1_low,z1_high,3)

	ls_cord_list=[]
	for i in x:
		for j in y:
			ls_cord_list.append([i,j])

	ord_list = model.sample(np.array(ls_cord_list))

	seq_list = [ordinal_decode(np.array(seq)) for seq in ord_list]

	print('Z0, Z1, Generated Sequence:')
	for sample in zip(ls_cord_list,seq_list):
		print(sample)

def initalize_model(latent_dim=2,seq_len=None,beta=1):
	model=m1_VAE(latent_dim,seq_len,beta) #initialize model
	return model

def train_VAE(model,df_train,df_test,min_epochs=10,max_epochs=1000,save_rate=10,print_ind_loss=False):
	'Trains m1 VAE until test loss stops improving. Returns the trained model.'
	x_train=df_train['Ordinal'].values
	x_train=np.stack(x_train).astype('int32')

	x_test=df_test['Ordinal'].values
	x_test=np.stack(x_test).astype('int32')
	
	optimizer = tf.keras.optimizers.Adam() #could adjust learning rate here

	epoch=0
	start_time = time.time()
	plot_latent_distribution(model,epoch,df_test) #untrained model should be random 
	train_loss,test_loss=[],[]
	while True:
		#train model
		train_loss.append(float(train_step(model, x_train, optimizer)))
		
		#test model
		test_loss_per_epoch = tf.keras.metrics.Mean()
		test_loss.append(float(test_loss_per_epoch(compute_loss(model, x_test, print_ind_loss))))
		
		end_time = time.time()
		epoch=epoch+1
		
		#plot and sample every 'save_rate' epochs of training 
		if epoch%save_rate==0: 
			plot_latent_distribution(model,epoch,df_test)
			print('Epoch: {0:.0f}, Train ELBO: {1:.2f}, Test ELBO: {2:.2f}, time: {3:.1f}'
			    .format(epoch, train_loss[-1], test_loss[-1], end_time - start_time))

		#end training if loss stops improving or max is reached
		if epoch>min_epochs:
			cur_loss=np.array(test_loss[-1])
			history=np.array(test_loss[-11:-1]) #histroy is last 10 epochs

			if not (cur_loss<history).any(): #stops if not better than any in history
				break
			if epoch==max_epochs:
				print('max reached')
				break

	#save final information
	print('Epoch: {0:.0f}, Train ELBO: {1:.2f}, Test ELBO: {2:.2f}, time: {3:.1f}'
	    .format(epoch, train_loss[-1], test_loss[-1], end_time - start_time))
	plot_functions.plot_loss_per_epoch(train_loss,test_loss)
	plot_functions.make_gif('M1VAE_E',save_rate)
	plot_latent_distribution(model,epoch,df_test)
	
	return model

