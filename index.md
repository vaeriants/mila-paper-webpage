---
layout: default
title: {{ site.name }}
---


---

Introduction
============
Deep generative models are popular nowadays when used for unsupervised
learning and for modeling structured outputs. In both cases, the
variable that one wants to model contains some structure that cannot be
modeled simply via describing its lower order moment or linear
correlation. Examples include natural images, natural languages, audio
data, etc.

There exists many ways to model the structure of a multivariate
distribution, such as 
   1. **autoregressive models** which assume the joint
probability of the random variables can be factorized as a product of
distributions, 
   2. **energy-based models** that define the interaction, or
compatibility between the values of the random variables, and 
   3. **latent-variable models** that assume there exists an unobserved latent
variate which can often be thought of as a lower dimensional
representation of the data. 

We focus on the last category in this
monograph, as for the problems that we are going to visit it is a
natural assumption to make.

Specifically, the Variational Autoencoders introduced by \[26\] assumes
a continuous latent variable model, with a Gaussian prior in the latent
space. It is a reasonable assumption to make in many cases, such as for
visual imagery: when we move the latent representation of an image in
the vicinity of the latent space, it corresponds to changes in shape,
color, intensity of light, location of an object, or different kinds of
high level semantics of the image.

![Frey Face Interpolation](https://github.com/vaeriants/vaeriants.github.io/blob/master/assets/kingma14_freyface_10x10.pdf Frey Face Interpolation)
 
However, training deep continuous latent variable models with nonlinear
mapping between the latent code and the observed variable was not so
successful only until recently. This is because evaluating the
likelihood of a data instance relies on inferring the latent
representation, and for non-linear, non-conjugate prior-likelihood pair
inference cannot be conducted exactly, or in a computationally tractable
way. So we need to *approximately* infer the latent representation. We
will talk more about this in \[\].

We rely heavily on a family of approximate inference techniques known as
the **Variational Inference** (VI), which is to cast the inference process
as an optimization problem by dealing with a lower or upper bound on the
original objective. This is different from the sampling-based *Markov
Chain Monte Carlo* (MCMC) methods in many ways. One important aspect is
the **bias-and-variance** trade-off underlying these two classes of
algorithms. The MCMC methods often result in high variance estimates but
are asymptotically unbiased, while the VI methods have lower variance
but due to its choice of (often well-understood and tractable)
variational family of distributions are often biased. Even though the
variational methods are known to have lower variance, the gradient
estimate needed for updating the parameters during training can be very
noisy, due to the intractable VI objective that can only be optimized
via stochastic gradient methods. We will extend the discussion of this
perspective more in \[\].

The goal of this monograph is to cover the (1) necessary background to
understand latent variable models and its training, different views on
the well-known variational autoencoders: both (2) from the generative
model point of view and (3) from the information theoretic point of
view. We will also cover the (4) the intuition of optimizing the
objective, (5) training details, and (6) different ways to improve the
model, with a hope to help deep learning researchers to better
understand the probabilistic aspect of deep generative models and how to
use them as a tool.

---

# Table of contents
* Background
    * Latent Variable Models: What is it good for?
    * Training Latent Variable Models
        * Maximum likelihood principle recap
        * Make your best guess. The EM algorithm
* Variational inference
    * The evidence lower bound
    * Stochastic variational inference
    * Amortized variational inference
* Information theoretic intuition of the ELBO
* Autoencoders as generative models
* The double-edged sword: 
$$D_{KL}(q||p)$$
* Improving the models: the Holy Trinity
* Evaluating the model
* Variance reduction

---


