---
layout: post
title: More familiar with Deep Learning?
nextpages:
 - prob_intro
 - background
---

Introduction
============

In the long history of research into artificial neural networks, autoencoders are an interesting topic.
The autoencoder is trained to be an identity 


They are commonly used to learn an encoding of the input. Most use a dimension bottleneck in the hidden layers to achieve a non-linear dimension reduction, but it is also sometimes useful to use a higher dimension, like in sparse autoencoders.
During the resurgence of neural networks as deep learning, autoencoders were largely used to do pretraining.
{% cite bengio2013generalized %} first interprets a special kind of autoencoder, called the denoising autoencoders, as a generative model.
One connection to make is that in their work, an unstructured noise is injected to the data, 

The introduction of VAEs {% cite kingma2013auto %} interprets the autoencoder as a graphical model with the encoding being a latent variable to be inferred.
While autoencoders on their own weren’t a probabilistic model before, the VAE framework offers an alternative look at them as such.

Because they were introduced as a generative model, one of the more common applications of VAEs is to generate images, i.e: sampling from a Gaussian distribution and pushing that through the decoder.
Another application of VAEs are what autoencoders were used for before: learning non-linear representations of the input.

However, the framework VAEs introduce is capable of much more. 
In DeepMind's paper on the topic, they refer to these models as Deep Latent Gaussian Models (DLGMs) {% cite rezende2014stochastic %}.
We think this may be more descriptive, although we need not limit ourselves to Gaussian distributions.
Sure, you _can_ use VAEs to autoencode, but using them for just that is missing the point {% cite chen2016variational %}.

