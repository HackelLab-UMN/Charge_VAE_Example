'''
Contains plotting functions for the VAE project
'''

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd
from PIL import Image
import glob

def plot(savename,data,x,x_label,y,y_label,hue=None):
	'Makes a 2d jointplot from a dataframe, used for charge and latent space visualizations'
	fig,ax=plt.subplots(1,1)
	s=sns.jointplot(data=data,x=x,y=y,kind='kde',height=4,ratio=5,joint_kws={'fill':False},hue=hue)
	s.ax_joint.set_xlabel(x_label)
	s.ax_joint.set_ylabel(y_label)
	plt.tight_layout()
	plt.savefig('./'+savename+'.png',dpi=300)
	plt.close()
	plt.clf()
	plt.close('all')


def plot_loss_per_epoch(train_loss,test_loss):
	'Creates a plot showing the loss per epoch during training'
	fig,ax=plt.subplots(1,1,figsize=[5,5])
	train_df=pd.DataFrame([list(range(len(train_loss))),train_loss])
	train_df=train_df.T
	train_df.columns=['Epoch','Loss']
	train_df['Dataset']='Training'

	test_df=pd.DataFrame([list(range(len(test_loss))),test_loss])
	test_df=test_df.T
	test_df.columns=['Epoch','Loss']
	test_df['Dataset']='Testing'

	df=pd.concat([train_df,test_df],ignore_index=True)

	df.loc[:,'Epoch']=df['Epoch']+1

	sns.lineplot(data=df,x='Epoch',y='Loss',hue='Dataset',ax=ax)
	plt.tight_layout()
	plt.savefig('./loss_per_epoch.png',dpi=300)
	plt.close()
	plt.clf()

def make_gif(commonname,save_rate):
	'combines images to make a gif. Used for watching latent space evolve.'
	fp_in = "./temp_images/"+commonname+"*.png"
	n_images=len(glob.glob(fp_in))
	sort_name=[]
	for i in range(0,n_images*save_rate,save_rate):
		sort_name.append("./temp_images/"+commonname+str(i)+'.png')
	img, *imgs = [Image.open(f) for f in sort_name]

	fp_out = "./VAE_latent.gif"

	img.save(fp=fp_out, format='GIF', append_images=imgs,
	         save_all=True, duration=200, loop=0)