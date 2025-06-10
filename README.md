# Generative Models for MNIST discrimination and generation
## Author : Jimmy Zhang

## Description
This project explores both **discriminative** and **generative** modeling approaches for classification and unsupervised learning on the MNIST dataset, implemented entirely from scratch in `NumPy` from first principles — no high-level ML libraries. Techniques span supervised naive Bayes and discriminative analysis to unsupervised EM and MCMC for latent mixture modeling.

## Notebooks
### 1 `NaiveBayesClassifier.ipynb`
Implements supverised generative classifier training algorithm for **binarized** version of MNIST dataset.  The model estimates class priors and feature-wise Bernoulli parameters for each class using maximum likelihood estimation, under the naive Bayes assumption of conditional independence between pixels given class.


### 2 `DiscriminativeAnalysis.ipynb`
Implements two supervised discriminative analysis approaches on the standard MNIST dataset using maximum likelihood estimation (MLE). That is, given a labeled dataset, each class-conditional distribution is modeled independently, and its parameters are estimated using MLE based on the data from that class.
#### 2.1 Linear Discriminative Analysis (LDA)  
LDA assumes that all class-conditional Gaussian distributions share a common covariance matrix. This constraint leads to linear decision boundaries between classes.
   
#### 2.2 Quadratic Discriminative Analysis (QDA)
QDA relaxes the shared covariance assumption by allowing each class to have its own covariance matrix, resulting in quadratic decision boundaries.

### 3 `MixedBernEM.ipynb`
Implements an unsupervised Bernoulli Mixture Model approach for learning latent clusters in a *binarized* MNIST dataset using **Expectation maximization**.  

### 3.1 Expectation Maximization
A frequentist implementation of the **Expectation-Maximization (EM)** algorithm for unsupervised clustering of binarized MNIST digits using a Bernoulli mixture model. The algorithm iteratively estimates cluster assignments as soft responsibilities and updates the model parameters via maximum likelihood estimation to improve the data likelihood at each step.

### 4 `MixedBernMCMC.ipynb`

Implements an unsupervised Bernoulli Mixture Model approach for learning latent clusters in a *binarized* MNIST dataset using **Markov Chain Monte Carlo (MCMC)**.  

### 3.2 Markov Chain Monte Carlo
A fully Bayesian approach that treats latent cluster assignments and model parameters as random variables modelled by a joint posterior distribution. Unlike point-estimate methods, this approach captures uncertainty in all variables. The posterior is approximated using **Markov Chain Monte Carlo (MCMC)** sampling — specifically, the **Gibbs Sampling** algorithm, which iteratively samples from the conditional distributions to infer both the cluster assignments and model parameters.
