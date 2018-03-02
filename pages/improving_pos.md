---
layout: post
title: Improving the Posterior
nextpages:
 - improving_likelihood
 - improving_prior
---
Improving the Posterior
=======================

{% include image.html url="/figures/2bits_gaussian.png" caption="Gaussian approximate posterior"%}
{% include image.html url="/figures/2bits_dsf.png" caption="Non-gaussian aproximate posterior."%}

Recall again that one of the goals of the generative model is to allow us to sample from the distribution over $\z$, and generate reasonable examples of our data. Ideally, this also means that there has to be a mapping from the data $\x$ to the distribution over $\z$ such that marginalising out the data, we recover the defined prior over $\z$.

Unfortunately, defining the approximate posterior as a Gaussian may not satisfy this requirement well enough. 
Consider a two-bit example: we sample $(x_a,x_b)$ from two independent Bernoulli distributions both with probability being $0.5$.
There are four possibilities. 
And we want to learn a VAE on this dataset.
If we assume the approximate posteriors to be Gaussians (see left of Figure \ref{ref:gauss_vs_flow}), then we would end up with an aggregate posterior $q(z) = \sum_{i=1}^4 q(z_i|x_i)\tilde{p}(x_i)$ being a mixture of Gaussians that fails to be prior-like ($p(z)=\N(z;0,I)$ in this case).
This is because the posterior distributions fail to capture the true ones, and if our inference network $q(z|x)$ is good enough to perfectly model the true posteriors, the aggregate posteriors should be the shape of the prior (see right of Figure \ref{ref:gauss_vs_flow}). See {% cite hoffman2016elbo %} or {% cite huang2017led %} for the derivation.

Turns out, the easiest remedy to this is to have a better posterior. 
With a bit of rearrangement and the introduction of a few new notations, the variational gap can be divided into the following three parts:


$$\begin{split}
\log p_{\theta}(x) - &\ELBO(\theta, \phi; x) = \\
&\underset{\text{Gap 1}}{\underbrace{\KL{q^*}{p}}} + \underset{\text{Gap 2}}{\underbrace{[\KL{q^*_\theta}{p}-\KL{q^*}{p}]}} + \underset{\text{Gap 3}}{\underbrace{[\KL{q_\theta}{p}-\KL{q^*_\theta}{p}]}}
\end{split}
$$

where
- $p$ is the true posterior
- $q^*$ is the optimal approximate posterior within the family $$\mathcal{Q}=\{q\in \mathcal{Q}\}$$
- $$q^*_{\phi}$$ is the optimal approximate posterior within the family $$\mathcal{Q}_\phi\subseteq\mathcal{Q}$$ due to the amortized inference via the conditional mapping $\phi\in\Phi:x\rightarrow \pi_q(x;\phi)$ where $\pi_q$ denotes the parameters of distribution $q$, and finally
- $q_\phi$ is the approximate posterior that we are analyzing. 


The first gap and the combination of Gap 2 and Gap 3 are what {% cite cremer2017inference %} called _approximation gap_ and _amortization gap_.
With these, we know that there are three pieces to improve in order to make the variational gap smaller:

1. better $\mathcal{Q}$, i.e. the choice of family of distributions. This can be achieved by choosing a larger family of distributions (such as normalizing flows).
2. better $\Phi$, i.e. better amortized inference. This can be achieved by having a higher capacity encoder.
3. better $\phi$, i.e. better encoder within the family $\Phi$. This can be achieved by improving the optimization scheme. 


<!---
In Table \ref{tb:betterpos} we list a few approaches designed to improve variational inference by making the variational gap smaller. 

\begin{table}
\caption{Methods that aim at reducing the variational gap.}
\label{tb:betterpos}
\centering
\begin{tabular}{lccc}
\toprule
& Gap 1 & Gap 2 & Gap 3 \\
\midrule
Better optimization &&&\xmark\\
Better encoder \cite{cremer2017inference}&&\xmark&\\
Normalizing flows \cite{rezende2015variational,kingma2016improving,huang2017facilitating} &\xmark&&\\
Hierarchical VI \cite{ranganath2016hierarchical,maaloe2016auxiliary}&\xmark&&\\
Implicit VI \cite{mescheder2017adversarial,huszar2017variational}&\xmark&&\\
Variational boosting \cite{miller2016variational}&\xmark&&\\
MCMC as part of $q_\phi$ \cite{salimans2015markov}&\xmark&&\\
MCMC on top of $q_\phi$ \cite{de2001variational,hoffman2017learning} &\xmark&\xmark&\xmark\\
Gradient flow \cite{duvenaud2016early} &\xmark&\xmark&\xmark\\
Importance sampling \cite{burda2015importance}&\xmark&\xmark&\xmark\\
\bottomrule
\end{tabular}
\end{table} //-->
