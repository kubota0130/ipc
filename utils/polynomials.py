#!/usr/bin/env python
# coding: utf-8

import cupy as cp
import sys

# Univariate polynomials for IPC
# Prepare univariate bases whose order is 0 to 'degree.'
class ipc_univariate_polynomials:
	def __init__(self,u,degree,name='gramschmidt',**kwargs):
		self.names = ['legendre','jacobi','laguerre','hermite',
					  'charlier','krawtchouk','meixner','hahn',
					  'gramschmidt']
		if name in self.names:
			self.name = name
			self.u = cp.array(u)
			self.T = len(u)
			self.degree = degree
			self.N = self.degree + 1
			self.kwargs = kwargs
			exec('self.%s()'%name)
		else:
			print('Usage: polynomial name %s'%name)
			sys.exit()

	# Input must be a gaussian random number [i.e., numpy.random.randn(T)]
	def hermite(self):
		P = cp.ones((self.N,self.T))
		P[1] = self.u
		for n in range(1,self.N-1):
			P[n+1] = self.u*P[n] - n*P[n-1]
		self.bases = P
		self.normalize()

	# Input must be a gamma random number [numpy.random.gamma(a+1,1,T)]
	def laguerre(self):
		a = self.kwargs['a']
		L = cp.ones((self.N,self.T))
		L[1] = 1+a-self.u
		for n in range(1,self.N-1):
			L[n+1] = ((2*n+1+a-self.u)*L[n]-(n+a)*L[n-1])/(n+1)
		self.bases = L
		self.normalize()

	# Input must be a beta random number [2*numpy.random.beta(alpha+1,beta+1,T)-1]
	def jacobi(self):
		a,b = self.kwargs['alpha'],self.kwargs['beta']
		P = cp.ones((self.N,self.T))
		P[1] = ((a+b+2)*self.u+a-b)/2
		for n in range(1,self.N-1):
			A = (2*n+a+b+1)*((2*n+a+b)*(2*n+a+b+2)*self.u+(a**2)-(b**2))/(2*(n+1)*(n+a+b+1)*(2*n+a+b))
			B = -(n+a)*(n+b)*(2*n+a+b+2)/((n+1)*(n+a+b+1)*(2*n+a+b))
			P[n+1] = A*P[n]+B*P[n-1]
		self.bases = P
		self.normalize()

	# Input must be a uniform random number in the range of [-1,1] [i.e., 2*numpy.random.rand(T)-1]
	def legendre(self):
		self.kwargs['alpha'],self.kwargs['beta'] = 0,0
		self.jacobi()

	# Input must be a Poisson random number [i.e., numpy.random.poisson(lam=a,T)]
	def charlier(self):
		a = self.kwargs['a']
		C = cp.ones((self.N,self.T))
		C[1] = 1-self.u/a
		for n in range(1,self.N-1):
			C[n+1] = ((n+a-self.u)*C[n] - n*C[n-1])/a
		self.bases = C
		self.normalize()

	# Input must be a Krawtchouk random number [i.e., numpy.random.binomial(N,p,T)]
	def krawtchouk(self):
		N = self.kwargs['N']
		p = self.kwargs['p']
		K = cp.ones((self.N,self.T))
		K[1] = (1-self.u/(p*N))
		for n in range(1,self.N-1):
			K[n+1] = ( (p*(N-n)+n*(1-p)-self.u)*K[n] - n*(1-p)*K[n-1] )/(p*(N-n))
		self.bases = K
		self.normalize()

	# Input must be a Meixner random number [i.e., numpy.random.negative_binomial(beta,1-c,T)]
	def meixner(self):
		c,b = self.kwargs['c'],self.kwargs['beta']
		M = cp.ones((self.N,self.T))
		M[1] = ((c-1)/(b*c)*self.u+1)*M[0]
		for n in range(1,self.N-1):
			M[n+1] = ((c-1)*self.u+n+(n+b)*c)/(c*(n+b))*M[n] - n/(c*(n+b))*M[n-1]
		self.bases = M
		self.normalize()

	# Input must be a hypergeometric random number [i.e., numpy.random.hypergeometric(-1-alpha,-1-beta,N,T)]
	def hahn(self):
		a,b,N = self.kwargs['alpha'],self.kwargs['beta'],self.kwargs['N']
		Q = cp.ones((self.N,self.T))
		A = N*(a+1)/(a+b+2)
		Q[1] = 1-self.u/A
		for n in range(1,self.N-1):
			A = (n+a+b+1)*(n+a+1)*(N-n)/(2*n+a+b+1)/(2*n+a+b+2)
			C = n*(n+a+b+N+1)*(n+b)/(2*n+a+b)/(2*n+a+b+1)
			Q[n+1] = (A+C-self.u)/A*Q[n] - C/A*Q[n-1]
		self.bases = Q
		self.normalize()

	# Gram-schmidt polynonmial covers arbitrary types of input distribution
	def gramschmidt(self):
		# Orthogonalization
		P = cp.ones((self.N,self.T))
		coef = cp.eye(self.N)
		for n in range(1,self.N):
			for i in range(n):
				coef[n,i] = self.gs_coef(self.u**n,P[i])
			P[n] = self.u**n + cp.sum(cp.vstack([coef[n,j]*P[j] for j in range(n)]),axis=0)
		self.bases = P
		self.normalize()
	def gs_coef(self,new,old):
		return -new.dot(old)/old.dot(old)
	
	def normalize(self):
		for n in range(self.N):
			self.bases[n] = self.bases[n]/cp.sqrt(self.bases[n].dot(self.bases[n]))

	def check_internalproduct(self):
		for i in range(self.N):
			for j in range(i+1,self.N):
				print(i,j,self.bases[i].dot(self.bases[j])/cp.linalg.norm(self.bases[i],2)/cp.linalg.norm(self.bases[j],2))





