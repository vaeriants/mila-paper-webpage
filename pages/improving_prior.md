---
layout: post
title: Improving the Prior
nextpages:
 - improving_likelihood
 - improving_pos
---
Improving the Prior
========================

Just as we can modify the approximate posterior to fit the prior better, we can also modify the prior such that it is learned from the data.

The expressivity of neural networks and developments in improving the posteriors allow the data to be mapped really well to simple priors like a Gaussian.
So if we can already modify the posterior, why bother changing the prior?
One reason may be due to discontinuity when interpolating in the latent space.
If we imagine each colour code to be a different modality in the data, moving from one colour group to another will represent a drastic change in the type of data decoded <!---CW: talk about inductive bias-->. 
This means that sampling from close regions in the latent space may not represent close relationships in the data space.
Allowing the prior to model a lower density between these different modalities may be more ideal when the type of representation learned is more important than simply sampling data points from the model.

{% cite tomczak2017vae %} proposes parameterising the prior as a mixture over a sample of the dataset, or using learned _pseudo-inputs_. {% cite serban2016multi %} introduces a prior parameterised by a piece-wise linear function. In {% cite huang2017led %}, the authors parameterise the prior with Real NVP transformations {% cite dinh2016density %}.
