#!/usr/bin/env python
# coding: utf-8
import numpy as np
import sys,os

class single_input_degdelaysets:
	def __init__(self,distr=None,zerobased=True,basedir='../utils'):
		self.zerobased = zerobased
		self.str0based =  '_%dbased'%(0 if zerobased==True else 1)
		self.distr = distr
		self.str_binary = 'binary' if self.distr=='bernoulli' else ''
		self.ddsetdir = '%s/%sdegdelaysets%s'%(basedir,self.str_binary,self.str0based)
		os.makedirs(self.ddsetdir,exist_ok=True)

	def load(self,degree,delay,shuffle=False):
		# Loading degdelaysets
		fn = '%s/%d_%d.npz'%(self.ddsetdir,degree,delay)
		if not os.path.exists(fn):
			if self.distr=='bernoulli':
				self.make_binary_degdelaysets(degree,delay)
			else:
				self.make_degdelaysets(degree,delay)
		str_shuffle = '_shuffle' if shuffle==True else ''
		npz = np.load(fn,allow_pickle=True)
		ddsets = npz['degdelaysets%s'%str_shuffle].tolist()
		return ddsets

	def degset(self,n):
		if n==0:
			yield []
			return
		for degset in self.degset(n-1):
			degset.append(1)
			yield degset
			degset.pop()
			if degset:
				degset[-1] += 1
				yield degset

	def degdelaysets(self,degset,delayset,degdelaysets=[]):
		if len(degset)==1:
			for delay in delayset:
				yield [[degset[0],delay]]
			return
		for degdelayset in self.degdelaysets(degset[:-1],delayset,degdelaysets):
			for delay in delayset:
				if np.sum([degdelay[1]==delay for degdelay in degdelayset])==0:
					yield degdelayset + [[degset[-1],delay]]

	def make_shuffle_degdelaysets(self,degsets):
		degdelaysets_shuffle = []
		for degs in degsets:
			ddset = [[d,repd+1] for repd,d in enumerate(degs)]
			degdelaysets_shuffle.append(ddset)
		# print('degdelaysets_shuffle',degdelaysets_shuffle)
		return degdelaysets_shuffle

	def sort_degdelaysets(self,degdelaysets):
		base = 10**(int(np.log10(len(degdelaysets)))+1)
		maxdelays = [ np.sum([delay*(base**(repd-len(degdelays))) for repd,[deg,delay] in enumerate(degdelays)]) for degdelays in degdelaysets ]
		degdelaysets_sorted = [ degdelaysets[i] for i in np.argsort(maxdelays) ]
		return degdelaysets_sorted

	def make_threshold_index(self,degsets,degdelaysets):
		threshold_index = []
		for degdelayset in degdelaysets:
			degs = sorted([d for d,_ in degdelayset])[::-1]
			tmp = [_degs==degs for _degs in list(degsets)]
			idx = np.where(tmp)[0][0]
			threshold_index.append(idx)
		return threshold_index

	def make_degdelaysets(self,deg,delay):
		print('Making degdelaysets of %d-order %d-delay'%(deg,delay))

		##### Make families of sets of degrees
		degsets = []
		for degs in self.degset(deg):
			if len(degs)<=delay:
				degsets.append(sorted(degs,reverse=True))
		if len(degsets)>1:
			degsets = np.unique(degsets)
		# print('degsets',degsets)

		##### Make families of sets of degrees and delays
		degdelaysets = []
		delayset = np.arange(1 if self.zerobased==False else 0,delay+1)
		for degset in degsets:
			for degdelayset in self.degdelaysets(degset,delayset.tolist()):
				ddset = np.array(degdelayset)
				ddset = ddset[np.argsort(ddset[:,1])]
				ddset = ddset[np.argsort(ddset[:,0])]
				degdelaysets.append(ddset.tolist())
		if len(degsets)>1:
			degdelaysets = np.unique(degdelaysets).tolist()
		# print('degdelaysets',degdelaysets,len(degdelaysets))
		if len(degdelaysets)>0:
			degdelaysets = self.sort_degdelaysets(degdelaysets)

		##### Make families of sets of degrees and delays for shuffle surrogate
		degdelaysets_shuffle = self.make_shuffle_degdelaysets(degsets)
		if len(degdelaysets_shuffle)>0:
			degdelaysets_shuffle = self.sort_degdelaysets(degdelaysets_shuffle)

		##### 
		threshold_index = self.make_threshold_index(degsets,degdelaysets)

		np.savez("%s/%d_%d.npz"%(self.ddsetdir,deg,delay),degdelaysets=degdelaysets,
														  degdelaysets_shuffle=degdelaysets_shuffle,
														  threshold_index=threshold_index)

	def make_binary_degdelaysets(self,degree,delay):
		print('Usage: binary input')
		sys.exit()
