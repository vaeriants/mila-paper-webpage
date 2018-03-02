---
layout: post
title: Information Theoretic Intuition
---

An Information Theoretic Intuition
==================================

For this view, we are trying to minimise the following objective:

$$\underbrace{-\expected{\condprob{q}{}{\z}{\x}}{\log \condprob{p}{}{\x}{\z}}}_\text{
 Reconstruction loss}
+ \underbrace{\KL{\condprob{q}{}{\z}{\x}}{\log \prob{p}{}{\z}}}_\text{
 Information gain
}$$

which is the standard VAE loss to _minimise_.

Karol Gregor explains [this lecture](https://youtu.be/P78QYjWh5sM?t=207). At the timestamp, he talks about an information-theoretic interpretation of the VAE loss. From this perspective, the KL-divergence term is viewed as an information bottleneck: The objective is to reconstruct the input while transmitting as little information as possible from the encoder to the decoder.


Thinking of the KL-divergence term as a regulariser allows us to relate it back to things like sparse autoencoders or contractive autoencoders. The reconstruction term is balanced with a regularisation term on the hidden representation. In a standard autoencoder, the encoder maps $\x$ to the hidden representation deterministically, with no constraints. The decoder then maps this representation back to the original input. In VAEs, the KL-divergence term regularises this representation $\z$ so that in aggregate, they form a Gaussian distribution. See Figure \ref{fig:ae_vs_vae}. This then allows you to sample from the prior $\prob{p}{}{\z}$ which is a Gaussian to generate images similar to your data.


However, one misunderstanding about the KL-divergence term is that it should be close to 0 for a good model. What this interpretation tells us that this shouldn’t be the case: a KL-divergence term close to 0 means no information is being transmitted by the encoder to the decoder. In our running example about salaries, knowing someone's salary should narrow down which group he belongs to, giving us some **information** about $\z$.

The great thing about looking at it this way is that you can think  of the term as a measurement of information flow that you can monitor during training.  As we will see later, in some circumstances where we have a powerful decoder, the KL-divergence does go to 0. We’ll talk about several ways we can overcome such issues.

This view has been covered in {% cite higgins2016beta %} and {% cite alemi2017information %} in much more detail.

