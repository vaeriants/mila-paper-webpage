---
layout: post
title: Improving the Lower Bound (The Holy Trinity)
nextpages:
 - improving_pos
 - improving_likelihood
 - improving_prior
 - variance_reduction
---
The Holy Trinity
================

If we consider the standard, single-latent-variable model, there are three main pieces we define, and then optimise over:

- **The approximate posterior**

  $$\condprob{q}{}{\z}{\x} $$

  _Otherwise called the _encoder_, or the _inference network in the context of VAEs, where they're parameterised by a deep network (written here as $f$).
  The conditional distribution over $\z$ given the data $\x$ is usually defined as Gaussian, with parameters that are a function of $\x$,  $$\condprob{q}{}{\z}{\x} = \N\left(f_\mu(\x), f_\sigma(\x)\right)$$.
- **The likelihood**

  $$\condprob{p}{}{\x}{\z}$$

  Otherwise called the _decoder_, or the _generative_ network in the context of VAEs, where they're also parameterised by a deep network.
- **The prior**

  $$\prob{p}{}{\z}$$

  In {% cite kingma2013auto %}, this is defined $\N(\mathbf{0}, \mathbf{1})$  and not parameterised.


All three aspects of the model can be improved, not only at the level of architecture, but in the types of distributions that are assumed for each of them in the model.

