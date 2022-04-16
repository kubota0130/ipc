# Information processing capacity
The information processing capacity (IPC) [1,2] is a measure to comprehensively examine computational capabilities of dynamial systems that receives random input 
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle u_t }#gh-light-mode-only">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle\color{white} u_t }#gh-dark-mode-only">. 
The system state 
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle x_t }#gh-light-mode-only">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle\color{white} x_t }#gh-dark-mode-only">
holds the past processed inputs 
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle z_t }#gh-light-mode-only">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle\color{white} z_t }#gh-dark-mode-only"> (e.g.,
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle z_t=u_{t-1},u_{t-2},u_{t-1}u_{t-2} }#gh-light-mode-only">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle\color{white} z_t=u_{t-1},u_{t-2},u_{t-1}u_{t-2} }#gh-dark-mode-only">). 
We emulate the processed inputs 
<img src="https://render.githubusercontent.com/render/math?math={ z_t }#gh-light-mode-only">
<img src="https://render.githubusercontent.com/render/math?math={\color{white} z_t }#gh-dark-mode-only"> with linear regression of the state as follows: 

<img src="https://render.githubusercontent.com/render/math?math={\displaystyle \hat{w} = \arg\min_w \sum_t (z_t - w^\top\cdot x_t)^2, }#gh-light-mode-only">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle\color{white} \hat{w} = \arg\min_w \sum_t (z_t - w^\top\cdot x_t)^2, }#gh-dark-mode-only">

<img src="https://render.githubusercontent.com/render/math?math={\displaystyle \hat{z}_t = \hat{w}^\top\cdot x_t. }#gh-light-mode-only">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle\color{white} \hat{z}_t = \hat{w}^\top\cdot x_t. }#gh-dark-mode-only">

Subsequently, we quantify the IPC as the amount of the held input using the following accuracy: 

<img src="https://render.githubusercontent.com/render/math?math={\displaystyle C = 1 - \frac{\sum_t (\hat{z}_t-z_t)^2}{\sum_t z_t^2}. }#gh-light-mode-only">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle\color{white} C = 1 - \frac{\sum_t (\hat{z}_t-z_t)^2}{\sum_t z_t^2}. }#gh-dark-mode-only"> 

Under the assumption that the state is a function of only input history 
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle x_t = f(u_{t-1},u_{t-2},\ldots) }#gh-light-mode-only">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle\color{white} x_t = f(u_{t-1},u_{t-2},\ldots) }#gh-dark-mode-only">, the total capacity 
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle C_{\rm tot} }#gh-light-mode-only">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle\color{white} C_{\rm tot} }#gh-dark-mode-only">, which is the summation of IPCs over all types of 
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle z_t }#gh-light-mode-only">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle\color{white} z_t }#gh-dark-mode-only">, is equivalent to the rank of correlation matrix 
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle r }#gh-light-mode-only">
<img src="https://render.githubusercontent.com/render/math?math={\displaystyle\color{white} r }#gh-dark-mode-only"> (see [1,2] for further details). 


# Environment
The scripts use a GPU to compute IPCs fast and adopt CuPy library for multiprocessing. 
Please install required libraries through the following procedure. 
1. Check your CUDA version by 
    ```
    /usr/local/cuda/bin/nvcc --version
    ```
2. Check your CuPy version [here](https://docs.cupy.dev/en/stable/install.html#installing-cupy)

3. Install the libraries by 
    ```
    pip install jupyter numpy matplotlib pandas cupy-cudaXXX
    ```

# Example codes
1. An echo state network (ESN)

    First, we demonstrate IPCs of an ESN to explain basic usage of the library. 
    Please read `sample1_esn.ipynb` for details. 
    After running it, we get the following IPC decomposion, which summarizes capacities for each order of input. 
    The total capacity 
    <img src="https://render.githubusercontent.com/render/math?math={\displaystyle C_{\rm tot} }#gh-light-mode-only">
    <img src="https://render.githubusercontent.com/render/math?math={\displaystyle\color{white} C_{\rm tot} }#gh-dark-mode-only">
    is equaivalent to the rank 
    <img src="https://render.githubusercontent.com/render/math?math={\displaystyle r }#gh-light-mode-only">
    <img src="https://render.githubusercontent.com/render/math?math={\displaystyle\color{white} r }#gh-dark-mode-only"> in the ordered region 
    (the spectral radius <img src="https://render.githubusercontent.com/render/math?math={\displaystyle \rho<1.0 }#gh-light-mode-only">
    <img src="https://render.githubusercontent.com/render/math?math={\displaystyle\color{white} \rho<1.0 }#gh-dark-mode-only">). 

    <img src="sample1.png" width=350>
    
2. Input distribution

    The input for IPC must be random but you can use an arbitrary type of input distribution other than uniform random input [2]. 
    `sample2_dist.ipynb` explains how to use seven other basic distributions. 
    Even if your input distribution is not included in the eight ones, you can compute IPCs using arbitrary polynomial chaos (aPC) [2]. 
    `sample2_dist.ipynb` also provides how to use aPC using a complex distribution such as a mixed Gaussian one. 



    

If you compute IPCs of your reservoir, please replace input, state, and a set of degrees and delays with yours. 

# Release notes
- version 0.10: 
A version for single input. 
You can compute IPCs using arbitrary input distribution except for bernoulli one. 

# References 
[1] Joni Dambre, David Verstraeten, Benjamin Schrauwen, and Serge Massar. ''Information processing capacity of dynamical systems.'' Scientific reports 2.1 (2012): 1-7.

[2] Tomoyuki Kubota, Hirokazu Takahashi, and Kohei Nakajima. ''Unifying framework for information processing in stochastically driven dynamical systems.'' Physical Review Research 3.4 (2021): 043135.
