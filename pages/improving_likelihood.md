---
layout: post
title: Improving the Likelihood
nextpages:
 - improving_pos
 - improving_prior
---
Improving the Likelihood
========================

Consider modelling an image using a single latent variable model.
In order to produce a reasonable, realistic looking image, we need to decide the objects in the image, their location, abstract stuff.
But beyond these abstract ideas, there are also the details, like lighting and texture that we need to model. 

When the probability of the given datapoint is defined as being factorised over every dimension, the implication is that the distribution over the observation is assumed to be completely dependent upon the latent variable.
However, the latent variable has limited capacity, and may model only the aspects that contribute to most of the reconstruction loss (abstract concepts) and there may be variations at a lower level of abstraction that may depend on other parts of the input (details).

Modelling the data this way are what we'll call autoregressive generative models, and some examples of such models are {% cite gulrajani2016pixelvae %} for images, with a slightly more general discussion found in {% cite chen2016variational %}, and {% cite bowman2015generating %} for text.

