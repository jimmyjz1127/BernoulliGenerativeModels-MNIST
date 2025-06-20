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

### 5 `MixedBernMCMCMissing.ipynb`
This notebook builds upon `MixedBernMCMC.ipynb` by extending the Bernoulli mixture model to handle incomplete binary data, specifically applied to binarized MNIST images with missing pixels. It investigates and compares both **Uncollapsed** and **Collapsed** Gibbs sampling approaches for inferring latent cluster assignments in the presence of missing data. Three variants are explored:

#### 5.1 Uncollapsed Gibbs with Sampling of Missing Pixels
In this approach, missing pixels are explicitly imputed at each iteration of the Gibbs sampler. The imputed values are then used alongside the observed data to sample both the Bernoulli parameters (feature-wise success probabilities) and latent cluster assignments. This introduces additional uncertainty into the model but can improve mixing when the missing rate is moderate.

#### 5.3 Uncollapsed Gibbs with Marginalization over Missing Pixels
Instead of imputing missing values, this variant treats them as latent variables and integrates them out analytically. Only the observed pixels are used when sampling cluster assignments and Bernoulli parameters. This avoids making hard guesses about missing values but can underutilize potentially informative structure in the missingness pattern.

#### 5.4 Collapsed Gibbs Sampling
This variant performs fully collapsed Gibbs sampling, where both the Bernoulli parameters and the component mixing proportions are marginalized out analytically. At each iteration, the sampler draws only the latent cluster assignments and imputes missing pixels using the posterior predictive distribution. This approach leverages conjugacy for computational efficiency and typically results in faster mixing and better uncertainty quantification under high missingness regimes.