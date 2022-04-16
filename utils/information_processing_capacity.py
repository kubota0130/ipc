#!/usr/bin/env python
# coding: utf-8
import numpy as np
import cupy as cp
import pandas as pd
import os,json
from utils.degdelaysets import single_input_degdelaysets
from utils.polynomials import ipc_univariate_polynomials

class ipc:
	def __init__(self):
		pass

	def singular_value_decomposition(self,x_numpy,finfo=None,thresh='N',rank=None,**kwargs):
		# Resize x
		x = cp.c_[x_numpy] if x_numpy.ndim>1 else cp.c_[x_numpy].T
		N = x.shape[0]   		# Node
		T = x.shape[1]-self.Two	# Total time steps
		# Debiasing x
		xmean = cp.c_[cp.mean(x[:,self.Two:],1)]
		x = x - xmean.dot( cp.ones((1,self.Two+T)) )
		x = x[:,self.Two:]
		# SVD
		u,sigma,v = cp.linalg.svd(x,full_matrices=False)
		sigma,v = cp.asnumpy(sigma),cp.asnumpy(v)

		# Machine limit for floating point types
		finfo = np.finfo(x.dtype).eps if finfo is None else finfo
		# Determine threshold for singular values
		if thresh=='N':
			eps =  (sigma**2).max()*N*finfo
		else: # self.thresh=='sqrt'
			eps =  (sigma**2).max()*0.5*np.sqrt(2*N+1)*finfo
		if rank==None:
			index = np.where((sigma**2)>eps)[0]
		else:
			index = np.arange(rank)
		P = v[index]

		return P,thresh,finfo

	def get_coef(self,P,z):
		return cp.dot(P,z)/cp.sqrt(cp.dot(z,z))

	def get_ipc(self,P,z):
		return cp.sum( (cp.dot(P,z)/cp.sqrt(cp.dot(z,z)))**2 )



class single_input_ipc(ipc):
	def __init__(self,zeta,Two,degdelays,
				 poly='gramschmidt',distr=None,Nseed=200,zerobased=True,poly_params={}):
		self.zeta = zeta
		self.Two = Two # Washout time
		self.degdelays = degdelays
		self.degs = np.sort(np.unique([deg for deg,delay in self.degdelays]))
		self.degmax = np.max(self.degs)
		self.delaymax = np.max([delay for deg,delay in self.degdelays])
		self.Nseed = Nseed	# Number of shuffle surrogates
		self.poly = poly	# Polynomial type
		self.poly_params = poly_params
		self.distr = 'bernoulli' if len(np.unique(zeta))==2 else distr	# Input distribution
		self.zerobased = zerobased	# Polynomial delay is 0-based (True) or 1-based (False)
		self.configlist = ['N','T','Two','Nseed','degdelays','zerobased','poly','distr','thresh','finfo','rank']
		self.indexlist = ['ipcs_degree','ipcs_delay','Ctots','ranks']
		# Prepare univariate polynomials
		u = cp.array(self.zeta)
		univariate_polys = ipc_univariate_polynomials(u,self.degmax,self.poly,**self.poly_params)
		self.bases = univariate_polys.bases
		# univariate_polys.check_internalproduct()
		# Prepare families of sets of degrees and delays
		self.degdelaysets = single_input_degdelaysets(distr=self.distr,zerobased=self.zerobased)

	def svd(self,x_numpy,**kwargs):
		x = np.c_[x_numpy] if x_numpy.ndim>1 else np.c_[x_numpy].T
		P,thresh,finfo = self.singular_value_decomposition(x,**kwargs)
		self.N = x.shape[0]		# Number of nodes
		self.rank = P.shape[0]	# Rank of state matrix (correlation matrix)
		self.T = P.shape[1]		# Time length (after washout)
		self.thresh = thresh	# Threshold type for SVD
		self.finfo = finfo		# 
		self.P = P				# linearly-independent state time series (i.e., singular vectors) R^{r x T}


	def save_config(self,path):
		self.path = path
		savedir = path[:path.rindex('/')]
		os.makedirs(savedir,exist_ok=True)
		d = dict()
		for c in self.configlist:
			exec('d[\'%s\'] = self.%s'%(c,c))
		print(d)
		with open(path+'.json', 'w') as f:
			json.dump(d,f,indent=2,ensure_ascii=False)

	def load_config(self,path):
		with open(path+'.json') as f:
			d = json.load(f)
		for c in self.configlist:
			exec('self.%s = d[\'%s\']'%(c,c))

	def compute(self,degree,delay,**kwargs):
		Two = self.Two
		T = self.T
		# Convert numpy.array to cupy.array
		P = cp.array(self.P)
		# Load families of sets of degrees and delays
		ddsets = self.degdelaysets.load(degree,delay)

		if self.rank>0:
			##### Compute IPCs
			ipcs = cp.array([ self.get_ipc(P,cp.prod(cp.stack([self.bases[deg,Two-delay:Two+T-delay] for deg,delay in ddset]),0)) for ddset in ddsets ])
			str_ddsets = [ str(ddset) for ddset in ddsets]
			ipcdf = pd.DataFrame({'degdelaysets':str_ddsets,'ipcs':cp.asnumpy(ipcs)})

			##### Compute IPCs with shuffled inputs
			# Load ddsets for shuffled inputs
			ddsets = self.degdelaysets.load(degree,delay,shuffle=True)
			# Prepare bases shuffled in the time direction
			bases = []
			for seed in range(self.Nseed):
				# Prepare shuffled input
				np.random.seed(seed)
				u = cp.array(np.random.permutation(self.zeta))
				# Prepare univariate polynomials
				univariate_polys = ipc_univariate_polynomials(u,degree,self.poly,**self.poly_params)
				bases.append( univariate_polys.bases )
			# 
			ipcs = [ cp.array([ self.get_ipc(P,cp.prod(cp.stack([bases[seed][deg,Two-delay:Two+T-delay] for deg,delay in ddset]),0)) for ddset in ddsets ]) for seed in range(self.Nseed) ]
			surrogate = cp.asnumpy( cp.vstack([c for c in ipcs]).T )
			# 
			str_ddsets = [ str(ddset) for ddset in ddsets ]
			surdf = pd.DataFrame(surrogate,index=str_ddsets)
		else:
			ipcs = np.zeros(len(ddsets))
			str_ddsets = [ str(ddset) for ddset in ddsets]
			ipcdf = pd.DataFrame({'degdelaysets':str_ddsets,'ipcs':ipcs})

			surrogate = np.zeros(len(ddsets))
			surdf = pd.DataFrame(surrogate,index=str_ddsets)

		ipcdf.to_pickle('%s_ipc_%d_%d.pkl'%(self.path,degree,delay))
		surdf.to_pickle('%s_sur_%d_%d.pkl'%(self.path,degree,delay))
		return ipcdf,surdf

	def threshold(self,ipcdf,surdf,deg,delay,th_scale=None,display=True):
		if th_scale!=None:
			self.th_scale = th_scale

		th_idx = np.load("%s/%d_%d.npz"%(self.degdelaysets.ddsetdir,deg,delay),allow_pickle=True)['threshold_index']

		if len(th_idx)>0:
			surmax = np.max(surdf.values,1)
			thr = self.th_scale*np.c_[surmax[th_idx]]
			truncated = ipcdf.iloc[np.where(ipcdf['ipcs'].values>thr[th_idx][:,0])[0]]
		else:
			truncated = pd.DataFrame({'degdelaysets':{},'ipcs':{}})

		if display==True:
			print(truncated)

		return truncated

	def max_delay(self,degdelayset):
		return np.max([int(delay) for delay in degdelayset.replace('[','').replace(']','').split(', ')[1::2]])

	def get_indicators(self,npzname,paths,th_scale=None,th_isolate=0.99):
		if th_scale!=None:
			self.th_scale = th_scale

		N = len(paths)
		self.ipcs_degree = np.nan*np.ones((N,len(self.degdelays)))
		self.ipcs_delay = np.zeros((N,self.delaymax+1))
		self.Ctots = np.zeros(N)
		self.ranks = np.nan*np.ones(N)

		for i,path in enumerate(paths):
			self.load_config(path)
			self.ranks[i] = self.rank
			# Added on version 1.1
			self.Neffs[i] = self.Neff

			for j,[deg,delay] in enumerate(self.degdelays):
				fn = '%s_ipc_%d_%d.pkl'%(path,deg,delay)
				sfn = '%s_sur_%d_%d.pkl'%(path,deg,delay)
				if os.path.exists(fn) & os.path.exists(sfn):
					ipcs = pd.read_pickle(fn)
					surs = pd.read_pickle(sfn)
					truncated = self.threshold(ipcs,surs,deg,delay,th_scale=self.th_scale,display=False)
					#
					Ctot_d = np.sum(truncated['ipcs'].values)
					self.Ctots[i] += Ctot_d
					self.ipcs_degree[i,j] = Ctot_d
					for degdelayset,cap in zip(truncated['degdelaysets'].values,truncated['ipcs'].values):
						k = self.max_delay(degdelayset)
						self.ipcs_delay[i,k] += cap
		d = dict()
		for i in self.indexlist:
			exec('d[\'%s\'] = self.%s'%(i,i))
		print(d)
		os.makedirs(npzname[:npzname.rindex('/')],exist_ok=True)
		np.savez(npzname,**d)

	def load_indicators(self,npzname):
		npz = np.load(npzname)
		for i in self.indexlist:
			exec('self.%s = npz[\'%s\']'%(i,i))

	def get_individuals(self,degs,degdelaysets,paths,th_scale=None):
		if th_scale is not None:
			self.th_scale = th_scale

		#Extract individual IPCs
		N = len(paths)
		indiv_ipcs = np.zeros((N,len(degdelaysets)))
		rest_ipcs,Ctots = np.zeros((N,len(self.degs))),np.zeros((N,len(self.degs)))
		for i,path in enumerate(paths):
			for j,[deg,delay] in enumerate(self.degdelays):
				#Load IPCs and surrogate data
				ipcs = pd.read_pickle('%s_ipc_%d_%d.pkl'%(path,deg,delay))
				surs = pd.read_pickle('%s_sur_%d_%d.pkl'%(path,deg,delay))
				#Threshold IPCs with surrogate data
				truncated = self.threshold(ipcs,surs,deg,delay,display=False)
				ipcs = truncated['ipcs'].values
				dds = truncated['degdelaysets'].values
				#Total IPC for each degree
				Ctots[i,j] += np.sum(ipcs)
				#Individual IPC
				total_indiv_ipc = 0
				for k,degdelayset in enumerate(degdelaysets):
					indiv_ipc = np.sum(ipcs[dds==str(degdelayset)])
					indiv_ipcs[i,k] += indiv_ipc
					total_indiv_ipc += indiv_ipc
				#Summarize the rest of dth-order IPCs (i.e., IPCs other than individual IPCs)
				rest_ipcs[i,j] = Ctots[i,j] - total_indiv_ipc

		str_ddsets = [ str(ddset) for ddset in degdelaysets ]
		self.indiv = pd.DataFrame(indiv_ipcs,columns=str_ddsets)
		self.rest = pd.DataFrame(rest_ipcs,columns=self.degs)




