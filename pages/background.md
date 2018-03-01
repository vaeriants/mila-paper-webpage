---
layout: post
title: Background on Variational Inference
nextpages:
 - vae
---

Variational Inference
=====================

## Evidence Lower Bound (ELBO)
Consider the following setting: We have an $\x$ we observe, and the generative process for this $\x$ is to sample $\z$ from a simple distribution that we know ($\prob{p}{}{\z}$) and some mysterious stochastic process turns $\z$ into $\x$  that we also know ($\condprob{p}{}{\x}{\z}$). So since we know these two pieces, figuring out how likely any given $\x$ is requires us to do this:

$$\prob{p}{}{\x} = \int \condprob{p}{}{\x}{\z} \prob{p}{}{\z} \mathrm{d}\z$$

Ooh. Big scary integral. And we have to maximise over it. What do?

Unless our $\condprob{p}{}{\x}{\z}$ is simple (a linear mapping, for example), this is intractable to compute. One way around this, is to think of it as an expectation over $\z$:

$$\prob{p}{}{\x} = \expected{\prob{p}{}{\z}}{\condprob{p}{}{\x}{\z}}$$

then we can estimate this quantity by sampling from $\prob{p}{}{\z}$ multiple times, and then averaging over the output,

$$ \approx \frac{1}{N} \sum^N_{i=1} \condprob{p}{}{\x}{\z_i} $$

<!--{% include image.html url="/figures/mixture_bad.png" description="When sampling from the prior (say, it's a uniform binary code), we end up updating the same class-conditional ${\condprob{p}{}{\x=x}{\z=j}}$ for some $j\in\{1,2\}$ and for all $x$ in the data set. 
Thus, there's no difference between the learned latent code $j$ (top).
If we sample more cleverly by grouping the data points, and assigning larger probability to $p_1(z=j)$ for group 1 and to $p_2$ for group 2, different latent code $j$ will be updated to correspond to a different and sharper class-conditional likelihood ${\condprob{p}{}{\x}{\z=j}}$ (bottom)." %} -->

{% include image.html url="/figures/q_vs_noq.png" description="Left: Sampling from $\prob{p}{}{\x}$, Right: Sampling from $\condprob{q}{}{\x}{\z}$. Using an approximate posterior $q$ narrows down the region in the latent space." %}

The problem with this approach is that it has _high variance_ (see Figure \ref{fig:fromprior1} and \ref{fig:fromprior2}). Think about what we are trying to do here: We have $\x$, and we're trying to maximise $\prob{p}{}{\z}$. However, what the latent variable model is saying is that there is \emph{some} or \emph{several} $\z$s that will generate the observed $\x$. But because we don't know what those $\z$s are, we're just randomly picking out a few from the distribution, pushing them through the generative process, and hoping they hit the mark.

Here's where $q$ comes in to save the day (Yes, just like Star Trek).

We come up with an _approximate posterior_, or in deep learning parlance, an encoder, $\condprob{q}{}{\z}{\x}$. Its mission? To -boldly go where- narrow down the area in the prior distribution $\prob{p}{}{\z}$ so that we have an easier time generating something close to the true $\z$ that corresponds to the observed $\x$. 

Now, to relate it back to the information-theoretic point-of-view, narrowing down the search space is akin to giving the model more ``information'' about where in the latent space the corresponding latent state is.

Of course, none of this comes for free. Let's take a look at what introducing $q$ means:

$$
\begin{align*}
\prob{p}{}{\x} 
&= \int \condprob{p}{}{\x}{\z} \prob{p}{}{\z} \mathrm{d}\z \\
&= \int \underbrace{\condprob{q}{}{\z}{\x} \frac{\condprob{p}{}{\x}{\z} \prob{p}{}{\z}}{\condprob{q}{}{\z}{\x}}}_{\text{multiplication by 1!}} \mathrm{d}\z = \expected{\condprob{q}{}{\z}{\x}}{\condprob{p}{}{\x}{\z} 
\frac{\prob{p}{}{\z}}{\condprob{q}{}{\z}{\x}}} 
\end{align*}
$$

Again, we can sample, but this time from $\condprob{q}{}{\z}{\x}$. 
However, notice that the quantity is now weighted by the ratio $\frac{\prob{p}{}{\z}}{\condprob{q}{}{\z}{\x}}$. 

So, if we're trying to maximise $\log \prob{p}{}{\x}$,

$$
\begin{align*}
\log \prob{p}{}{\x} &= 
\log \expected{\condprob{q}{}{\z}{\x}}{
\condprob{p}{}{\x}{\z} \frac{\prob{p}{}{\z}}{\condprob{q}{}{\z}{\x}}} \\
&\geq \expected{\condprob{q}{}{\z}{\x}}{\log
\condprob{p}{\theta}{\x}{\z} \frac{\prob{p}{\theta}{\z}}{\condprob{q}{\phi}{\z}{\x}}} = \ELBO(\theta, \phi;x) 
\end{align*}
$$

and we get the lower-bound. We use subscripts to denote parameters of the distributions.

The gap between $\log \prob{p}{}{\x}$ and this lower-bound is known as the \emph{variational gap}, and has its own interpretation (Covered in some appendix).
