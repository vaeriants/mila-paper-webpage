---
bibliography: 'bib.bib'
csl: 'acm-sig-proceedings.csl'
title: |
    VAE Cookbook:\
    its why and how, and the recent advances
---

\maketitle
\tableofcontents
\newpage
Overview
========

Deep generative models are popular nowadays when used for unsupervised
learning and for modeling structured outputs. In both cases, the
variable that one wants to model contains some structure that cannot be
modeled simply via describing its lower order moment or linear
correlation. Examples include natural images, natural languages, audio
data, etc.

There exists many ways to model the structure of a multivariate
distribution, such as (1) autoregressive models which assume the joint
probability of the random variables can be factorized as a product of
distributions, (2) energy-based model that define the interaction, or
compatibility between the values of the random variables, and (3)
latent-variable models that assume there exists an unobserved latent
variate which can often be thought of as a lower dimensional
representation of the data. We focus on the last category in this
monograph, as for the problems that we are going to visit it is a
natural assumption to make.

Specifically, the Variational Autoencoders introduced by \[26\] assumes
a continuous latent variable model, with a Gaussian prior in the latent
space. It is a reasonable assumption to make in many cases, such as for
visual imagery: when we move the latent representation of an image in
the vicinity of the latent space, it corresponds to changes in shape,
color, intensity of light, location of an object, or different kinds of
high level semantics of the image.

However, training deep continuous latent variable models with nonlinear
mapping between the latent code and the observed variable was not so
successful only until recently. This is because evaluating the
likelihood of a data instance relies on inferring the latent
representation, and for non-linear, non-conjugate prior-likelihood pair
inference cannot be conducted exactly, or in a computationally tractable
way. So we need to *approximately* infer the latent representation. We
will talk more about this in \[\].

We rely heavily on a family of approximate inference techniques known as
the *Variational Inference* (VI), which is to cast the inference process
as an optimization problem by dealing with a lower or upper bound on the
original objective. This is different from the sampling-based *Markov
Chain Monte Carlo* (MCMC) methods in many ways. One important aspect is
the *bias-and-variance* trade-off underlying these two classes of
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

LINKS (navigating page)

1.  Latent Variable Models

2.  Training: the Evidence Lower Bound

3.  Information theoretic intuition of the ELBO

4.  Autoencoders as generative models

5.  The double-edged sword: $D_{\mathrm{KL}}\left( q \| p \right)$

6.  Improving the models: the Holy Trinity

7.  Evaluation

8.  Variance reduction

\todo{[CW] overall missing: DRAW, variational memory models (few shot generation), variational addressing, beta-VAE, over-pruning problem, discrete LVM}
Variational Autoencoders: Not just an autoencoder.
==================================================

\todo{rewrite "Autoencoders as generative models"}
\todo{talk about amortized inference -> introduce the concept of encoder}
\todo{intuition of how an autoencoder can be thought of as a generative model}
\todo{"Not just an autoencoder." can be another sub-section -> lossy encoding? hierarchy of representations}
In the long history of research into artificial neural networks,
autoencoders are an interesting topic. The autoencoder is trained to be
an identity

They are commonly used to learn an encoding of the input. Most use a
dimension bottleneck in the hidden layers to achieve a non-linear
dimension reduction, but it is also sometimes useful to use a higher
dimension, like in sparse autoencoders. During the resurgence of neural
networks as deep learning, autoencoders were largely used to do
pretraining.

The introduction of VAEs \[26\] interprets the autoencoder as a
graphical model with the encoding being a latent variable to be
inferred. While autoencoders on their own weren't a probabilistic model
before, the VAE framework offers an alternative look at them as such.

Because they were introduced as a generative model, one of the more
common applications of VAEs is to generate images, i.e: sampling from a
Gaussian distribution and pushing that through the decoder. Another
application of VAEs are what autoencoders were used for before: learning
non-linear representations of the input.

However, the framework VAEs introduce is capable of much more. In
DeepMind's paper on the topic, they refer to these models as Deep Latent
Gaussian Models (DLGMs) \[36\]. We think this may be more descriptive,
although we need not limit ourselves to Gaussian distributions. Sure,
you *can* use VAEs to autoencode, but using them for just that is
missing the point.

\todo[inline, color=green!40]{Explain how to improve}
Thus far, VAEs have been used to:

-   Isolate known factors from unknown factors causing the observed
    data: e.g. \[24\].

-   Disentangling (to some extent) unknown factors causing the observed
    data: e.g. \[16\].

-   Leveraging bigger datasets to achieve meta-learning on a smaller set
    of data: e.g. \[14\]

Latent Variable Models: What is it good for? 
=============================================

Deep neural networks (DNN) have been shown to be a powerful inference
tool in multiple domains in recent years. Its expressiveness allows one
to take the available information to answer a question of interest. For
example, given someone's occupation and age, we can predict his salary.
Given a patient's X-ray image, we want to determine whether he has
cancer. Or by giving the computer a set of images, we can ask it which
of them are dogs and which of them are cats, which is one of the popular
tasks in machine learning.

##### Representation and Modeling.

The problems above can all be formulated as a conditional probability of
a pair of variables $(X,Y)$, e.g. $X$ being X-ray images and $Y$ being a
Bernoulli variable: $1$ if someone has cancer and $0$ if not. Here we
are simply interested in $p(Y|X)$. As stated, DNNs are powerful tools to
model this kind of conditional probabilities because of its
expressiveness and the simplicity in thinking about the conditional
relationship. But can we do more?

Lets think about the case where we only observe variable $X$, which
denotes one's salary. In our belief, there is a latent state $z$
associated with each instantiation, $x$, of the observed variable $X$.
Modeling this latent variable along with the observed variable allows us
to better capture the characteristics of the unknown distribution of
interest. For instance, the true income distribution might have multiple
modes. We thus assume the prior distribution $p(z)$ to be a multinoulli
distribution; each $z$ can take on one of the three indices $\{0,1,2\}$
denoting low, average and high income groups. In some simple cases, one
can infer the posterior probability $p(z|x)$ (in exact form), which in
our example means the probability of someone falling in one of these
groups given his salary information.

What if we want to model more complicated relationships between latent
states and observable attributes? Take natural images for example. One
might want the latent state to encapsulate some "semantic" information
of an image, such as the identity of a person in the image, whether
someone is smiling and wearing glasses, if there's a cat or dog in the
image, etc. On the other hand, there might be different levels of
factors that result in the image we now see, including lighting, angle,
whether the lens of the camera is clean or not, and a lot of other
variations. With slight abuse of language, we refer to all of these
unobserved factors that are correlated with the observed variable as
latent state. We are now interested in modeling the complicated mapping
from the latent state to the observed variable, and naturally choose to
use DNNs to define such a mapping. This is also known as deep latent
variable models, or
[*deepification*](http://videolectures.net/deeplearning2017_welling_inference/)
of latent variable models (LVM).

##### Limited and Incomplete Data.

Another usage of LVM is to deal with the scarcity of the data.

When data instances are lacking, it will be better to leverage prior
knowledge on functions into modeling, to prevent overfitting or help the
model generalize better. Such approaches stem from the Bayesian
community and have gained tremendous popularity when combined with Deep
Learning techniques. Treating weights of a DNN as a random variable, for
instance, allows us to calibrate predictive uncertainty, by sampling
i.i.d. samples of weights from the so-called posterior distribution. The
posterior predictive is better calibrated since making a prediction
requires averaging over multiple samples of weights, which gives it an
*ensemble* interpretation. Think of each sample as an expert, who has
its own knowledge about how to capture the correlation among the
variables (e.g. $Y|X$). All the experts need to reach a consensus for
the posterior predictive to express high confidence in one prediction;
otherwise the great dispersion of the posterior predictive can serve as
a warning for one not to trust the model!

Another kind of scarcity is the incompleteness of data features. For
example, the latent state examples above all fall into this category, as
we can think of the unobserved variables (semantics, lighting
variations, modal characteristics of the data distribution) as features
that are never provided, thus always incomplete.

In machine learning problems, we often represent our data in tensor
form. In 2D cases, we consider a dataset matrix; the number of rows
($N$) corresponds to the number of data samples and the number of
columns ($M$) corresponds to the number of features. When $N$ is small,
then we don't have enough data instances and when $M$ is smaller than we
believe, there are some features that are unobserved. These are the two
general cases that are just discussed above. We can also represent the
dataset in higher dimensional tensor or a graph, and there are other
kinds of incompleteness, such as views, labels, etc. Below, we discuss
some more applications of deep LVM and try to connect to the perspective
of scarcity.

Applications of Deep Latent Variable Models
-------------------------------------------

##### Semisupervised Classification.

Lets turn to our all time favorite dataset, the MNIST hand-written
digit. The dataset comes in image-label ($x$-$y$) pairs. We'd like to
create a semi-supervised scenario where the labels are only partially
observed: sometimes you have them, sometimes you don't.

\[24, 28\] introduce a way to model partially labeled data with
variational autoencoders. The idea is to build a LVM on the images
($\mathbf{x}$). Remember that the latent state of the observed images
can be used to model high level semantics, low level variations and the
modal characteristics of the data distribution. Here we want to separate
the digit's class ($y$) from other variation ($\mathbf{z}$), such as
style, orientation, and the thickness of the brush. In the generative
process, given the class of the digit $y=j$, the digit is generated from
the latent state that is sampled independently from a prior of all the
other variation $x\sim p(x|z,y)$ and $z\sim p(z)$. For data points that
are labeled, we infer the latent state $z$ as usual for training (which
we describe below), and for data points that are not labeled with
classes, naturally, the thing to do is to marginalize over all possible
labels. In plain words, lets assume we are given a classifier of digit
classes $q(y|x)$, which can be trained jointly on labeled data. We need
to consider all possible classes of a certain image ($i$), which has
probability $p(y_i=j|x_i)$, and maximize the likelihood of the data
point being in that class.

This approach allows us to treat the regressand as a latent variable and
train the joint probability generatively when labels are not provided
(incomplete data), and train the inference model/classifier
discriminatively when labels are given.

##### Unobserved Confounders.

To analyze the effect of an independent variable $x$ on the dependent
variable $y$, sometimes we need to suppress the "confounding\" effect of
an extraneous variable $z$ that influences both $x$ and $y$. The
variable $z$ is known as the confounder of the causal relationship
between $x$ and $y$. Consider $x$ to be whether a Facebook post is
associated with an image, a selfie for example, or not, and $y$ to be
the number of likes it has. We now want to know whether taking a selfie
while posting something has a positive effect on the number of likes the
post can get. If we simply collect data from the general public, we will
also need to take how popular and how outgoing someone is into account,
which has an influence on whether his or her posts usually come with a
selfie and on the number of likes he or she can get.

Another interesting example is given by  \[27\]. In this paper, they
treat the confounder $z$ as a latent variable, as a confounder is
usually not observed (again, incomplete data). The confounder (such as
one's socio-econo status) has a direct effect on the intervention $x$,
such as the medication a patient has access to, and the outcome $y$,
such as the patient's health condition. Without measuring the
confounder, there's no general way to isolate the causal effect of
medication on the patient's health. They propose to incorporate a noisy
view $v$ on the latent confounder, for example the patient's income,
that is highly correlated with the latent confounder. The observed
confounder thus can be used to infer the latent confounder and allow us
to inspect individual causal effects.

##### High-Level Semantics.

![Variation in different levels of a kind of hierarchical VAE called
PixelVAE \[15\] (with autoregressive likelihood instead of conditionally
independent distribution such as fully factorized Gaussian). The samples
are generated by sampling from certain layer while holding the others
fixed as deterministic hidden units (taking only the mean of a
Gaussian). Left: Sampling from the top level layer ($z_2$), Middle:
Sampling from the middle layer ($z_1$), Right: Sampling only from the
pixel level ($x$).](figures/pixelvae_z2.png "fig:"){width="32%"}
![Variation in different levels of a kind of hierarchical VAE called
PixelVAE \[15\] (with autoregressive likelihood instead of conditionally
independent distribution such as fully factorized Gaussian). The samples
are generated by sampling from certain layer while holding the others
fixed as deterministic hidden units (taking only the mean of a
Gaussian). Left: Sampling from the top level layer ($z_2$), Middle:
Sampling from the middle layer ($z_1$), Right: Sampling only from the
pixel level ($x$).](figures/pixelvae_z1.jpg "fig:"){width="32%"}
![Variation in different levels of a kind of hierarchical VAE called
PixelVAE \[15\] (with autoregressive likelihood instead of conditionally
independent distribution such as fully factorized Gaussian). The samples
are generated by sampling from certain layer while holding the others
fixed as deterministic hidden units (taking only the mean of a
Gaussian). Left: Sampling from the top level layer ($z_2$), Middle:
Sampling from the middle layer ($z_1$), Right: Sampling only from the
pixel level ($x$).](figures/pixelvae_x.jpg "fig:"){width="32%"}
[\[ref:pixelvae\_difflayers\]]{#ref:pixelvae_difflayers
label="ref:pixelvae_difflayers"}

Extracting different levels of representation of the data is a
fundamental problem of deep learning. The aim of an autoencoder is to
learn a meaningful encoding of the data that can be used to reconstruct
the original data points. Through reconstruction, the structure in the
original data distribution is recovered. This is done by mapping from
the representation, or the code, through the decoder which defines the
structure, i.e. the information that is lost during encoding. By
imposing a "simple" prior distribution on the latent representation, we
hope the encoder will destroy such structure in the data distribution
and rely on the decoder to "semanticate" the unstructured codes. It is
also possible to gradually remove the structure to have a hierarchy of
representations.

Assuming a hierarchy of latent states allows us to do this. Consider the
following generative model, in which the latent variables $z$ are split
into $L$ layers:
$$p(x,z) = p(x|z_1)p(z_L)\prod_{l=1}^{L-1}p(z_l|z_{>l})$$

This allows us to model different layers of latent representations. At
the top layer, high level information that everything else depends on is
represented. For instance, if the model knows the image has a frog in it
(conditioned on $z_L$ being equal to or close to some vector
representation of a frog), the lower level representation could be
natural scenes such as a pond, magnolia leaves or insects that are
highly correlated with a frog in an image.

\[40\] is one of the first papers that build on hierarchical VAEs,
followed by \[15\] and \[7\]. In the latter two, the authors proposed to
use an autoregressive decoder called PixelCNN \[32\] to model the low
level correlation between the pixels. This helps to separate high level
semantics from the textural details of the image. See Figure
[\[ref:pixelvae\_difflayers\]](#ref:pixelvae_difflayers){reference-type="ref"
reference="ref:pixelvae_difflayers"} for an illustration. The same idea
was explored independently by \[1\] using RealNVP \[11\] to make the
adjacent pixels dependent on each other given the latent state. Another
line of research focusing on adversariallly trained inference \[12\]
mechanism has also been exploring the hierarchical representations of
data \[22\]. Finally, \[5\] tried to explore the same idea with
language, hoping to separate semantics of a sentence from its syntax
which can be modeled by a conditional language model (such as an RNN).

##### Neural Statistician.

Earlier we talked about treating parameters of a model as random
variables, and since it is not observed, we need to infer it based on
the observations, in this case the whole training data. Lets denote
these stochastic parameters as $c$, and we are interested in inferring
the most likely $c$ given $\mathcal{D}$, i.e. $p(c|\mathcal{D})$.
Carrying on the terminology of deep LVM, $c$ is the latent state of
$\mathcal{D}$, which is now not an observation of a single data point
but the whole set of training data.

\[cont'd\]

Two Views of the Variational Lower Bound
========================================

Information-theoretic (intuitive)
---------------------------------

![Above: A standard autoencoder, which maps from the $x$ space to the
$z$ space, and then maps that back with a reconstruction term depicted
in red. Below: The variational autoencoder maps from $x$ to a
*distribution* over $z$, with a KL-divergence term over the approximate
posterior (green) and the prior distribution (blue), which can be seen a
regulariser.](figures/autoencoder_vs_vae.png){width="75%"}

[\[fig:ae\_vs\_vae\]]{#fig:ae_vs_vae label="fig:ae_vs_vae"}

For this view, we are trying to minimise the following objective:
$$\underbrace{-\mathbb{E}_{{q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right)}\left[ \log {p}_{} \left( \mathbf{x} \middle| \mathbf{z} \right) \right]}_\text{
 Reconstruction loss}
+ \underbrace{D_{\mathrm{KL}}\left( {q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right) \| \log {p}_{} \left( \mathbf{z} \right) \right)}_\text{
 Information gain
}$$ which is the standard VAE loss to *minimise*.

Karol Gregor explains VAEs in [this
lecture](https://youtu.be/P78QYjWh5sM?t=207). At the timestamp, he talks
about an information-theoretic interpretation of the VAE loss. From this
perspective, the KL-divergence term is viewed as an information
bottleneck: The objective is to reconstruct the input while transmitting
as little information as possible from the encoder to the decoder.

Thinking of the KL-divergence term as a regulariser allows us to relate
it back to things like sparse autoencoders or contractive autoencoders.
The reconstruction term is balanced with a regularisation term on the
hidden representation. In a standard autoencoder, the encoder maps
$\mathbf{x}$ to the hidden representation deterministically, with no
constraints. The decoder then maps this representation back to the
original input. In VAEs, the KL-divergence term regularises this
representation $\mathbf{z}$ so that in aggregate, they form a Gaussian
distribution. See Figure
[\[fig:ae\_vs\_vae\]](#fig:ae_vs_vae){reference-type="ref"
reference="fig:ae_vs_vae"}. This then allows you to sample from the
prior ${p}_{} \left( \mathbf{z} \right)$ which is a Gaussian to generate
images similar to your data.

However, one misunderstanding about the KL-divergence term is that it
should be close to 0 for a good model. What this interpretation tells us
that this shouldn't be the case: a KL-divergence term close to 0 means
no information is being transmitted by the encoder to the decoder. In
our running example about salaries, knowing someone's salary should
narrow down which group he belongs to, giving us some *information*
about $\mathbf{z}$.

The great thing about looking at it this way is that you can think of
the term as a measurement of information flow that you can monitor
during training. As we will see later, in some circumstances where we
have a powerful decoder, the KL-divergence does go to 0. We'll talk
about several ways we can overcome such issues.

This view has been covered in \[16\] and \[2\] in much more detail.

Evidence Lower BOund (ELBO)
---------------------------

Consider the following setting: We have an $\mathbf{x}$ we observe, and
the generative process for this $\mathbf{x}$ is to sample $\mathbf{z}$
from a simple distribution that we know
(${p}_{} \left( \mathbf{z} \right)$) and some mysterious stochastic
process turns $\mathbf{z}$ into $\mathbf{x}$ that we also know
(${p}_{} \left( \mathbf{x} \middle| \mathbf{z} \right)$). So since we
know these two pieces, figuring out how likely any given $\mathbf{x}$ is
requires us to do this: $${p}_{} \left( \mathbf{x} \right) 
=   \int {p}_{} \left( \mathbf{x} \middle| \mathbf{z} \right) {p}_{} \left( \mathbf{z} \right) \mathrm{d}\mathbf{z}$$
Ooh. Big scary integral. And we have to maximise over it. What do?

Unless our ${p}_{} \left( \mathbf{x} \middle| \mathbf{z} \right)$ is
simple (a linear mapping, for example), this is intractable to compute.
One way around this, is to think of it as an expectation over
$\mathbf{z}$: $$\begin{aligned}
{p}_{} \left( \mathbf{x} \right) &= \mathbb{E}_{{p}_{} \left( \mathbf{z} \right)}\left[ {p}_{} \left( \mathbf{x} \middle| \mathbf{z} \right) \right]
\intertext{then we can estimate this quantity by sampling from ${p}_{} \left( \mathbf{z} \right)$ multiple times, and then averaging over the output,}
&\approx \frac{1}{N} \sum^N_{i=1} {p}_{} \left( \mathbf{x} \middle| \mathbf{z}_i \right)\end{aligned}$$

![Left: Sampling from ${p}_{} \left( \mathbf{x} \right)$, Right:
Sampling from ${q}_{} \left( \mathbf{x} \middle| \mathbf{z} \right)$.
Using an approximate posterior $q$ narrows down the region in the latent
space.](figures/q_vs_noq.png){width="85%"}

The problem with this approach is that it has *high variance*. Think
about what we are trying to do here: We have $\mathbf{x}$, and we're
trying to maximise ${p}_{} \left( \mathbf{z} \right)$. However, what the
latent variable model is saying is that there is *some* or *several*
$\mathbf{z}$s that will generate the observed $\mathbf{x}$. But because
we don't know what those $\mathbf{z}$s are, we're just randomly picking
out a few from the distribution, pushing them through the generative
process, and hoping they hit the mark.

Here's where $q$ comes in to save the day (Yes, just like Star Trek).

We come up with an *approximate posterior*, or in deep learning
parlance, an encoder,
${q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right)$. Its job? To
~~boldly go where~~ narrow down the area in the prior distribution
${p}_{} \left( \mathbf{z} \right)$ so that we have an easier time
generating something close to the true $\mathbf{z}$ that corresponds to
the observed $\mathbf{x}$.

Now, to relate it back to the information-theoretic point-of-view,
narrowing down the search space is akin to giving the model more
"information" about where in the latent space the corresponding latent
state is.

Of course, none of this comes for free. Let's take a look at what
introducing $q$ means: $$\begin{aligned}
{p}_{} \left( \mathbf{x} \right) 
&= \int {p}_{} \left( \mathbf{x} \middle| \mathbf{z} \right) {p}_{} \left( \mathbf{z} \right) \mathrm{d}\mathbf{z}\\
&= \int \underbrace{{q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right) \frac{{p}_{} \left( \mathbf{x} \middle| \mathbf{z} \right) {p}_{} \left( \mathbf{z} \right)}{{q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right)}}_{\text{multiplication by 1!}} \mathrm{d}\mathbf{z}= \mathbb{E}_{{q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right)}\left[ {p}_{} \left( \mathbf{x} \middle| \mathbf{z} \right) 
\frac{{p}_{} \left( \mathbf{z} \right)}{{q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right)} \right] \end{aligned}$$
Again, we can sample, but this time from
${q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right)$. However, notice
that the quantity is now weighted by the ratio
$\frac{{p}_{} \left( \mathbf{z} \right)}{{q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right)}$.

So, if we're trying to maximise $\log {p}_{} \left( \mathbf{x} \right)$,
$$\begin{aligned}
\log {p}_{} \left( \mathbf{x} \right) &= 
\log \mathbb{E}_{{q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right)}\left[ {p}_{} \left( \mathbf{x} \middle| \mathbf{z} \right) \frac{{p}_{} \left( \mathbf{z} \right)}{{q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right)} \right] 
\intertext{then, due to Jensen's inequality,}
&\geq \mathbb{E}_{{q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right)}\left[ \log{p}_{\theta} \left( \mathbf{x} \middle| \mathbf{z} \right) \frac{{p}_{\theta} \left( \mathbf{z} \right)}{{q}_{\phi} \left( \mathbf{z} \middle| \mathbf{x} \right)} \right] = \mathcal{L}(\theta, \phi;x) \end{aligned}$$
and we get the lower-bound. We use subscripts to denote parameters of
the distributions.

The gap between $\log {p}_{} \left( \mathbf{x} \right)$ and this
lower-bound is known as the *variational gap*, and has its own
interpretation (Covered in some appendix).

The Holy Trinity
================

\todo[inline, color=green!40]{Briefly explain what improving each of the following aspects entails.}
If we consider the standard, single-latent-variable model, there are
three main pieces we define, and then optimise over:

-   **The approximate posterior**
    $${q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right)$$ Otherwise
    called the *encoder*, or the *inference* network in the context of
    VAEs, where they're parameterised by a deep network (written here as
    $f$). The conditional distribution over $\mathbf{z}$ given the data
    $\mathbf{x}$ is usually defined as Gaussian, with parameters that
    are a function of $\mathbf{x}$,
    ${q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right) = \mathcal{N}\left(f_\mu(\mathbf{x}), f_\sigma(\mathbf{x})\right)$.

-   **The likelihood**
    $${p}_{} \left( \mathbf{x} \middle| \mathbf{z} \right)$$ Otherwise
    called the *decoder*, or the *generative* network in the context of
    VAEs, where they're also parameterised by a deep network.

-   **The prior** $${p}_{} \left( \mathbf{z} \right)$$ In \[26\], this
    is defined $\mathcal{N}(\mathbf{0}, \mathbf{1})$ and not
    parameterised.

All three aspects of the model can be improved, not only at the level of
architecture, but in the types of distributions that are assumed for
each of them in the model.

Better posteriors
-----------------

![Left: Gaussian approximate posterior, Right: Non-gaussian aproximate
posterior.](figures/2bits_gaussian.png "fig:"){width="50%"} ![Left:
Gaussian approximate posterior, Right: Non-gaussian aproximate
posterior.](figures/2bits_dsf.png "fig:"){width="50%"}
[\[ref:gauss\_vs\_flow\]]{#ref:gauss_vs_flow label="ref:gauss_vs_flow"}

Recall again that one of the goals of the generative model is to allow
us to sample from the distribution over $\mathbf{z}$, and generate
reasonable examples of our data. Ideally, this also means that there has
to be a mapping from the data $\mathbf{x}$ to the distribution over
$\mathbf{z}$ such that marginalising out the data, we recover the
defined prior over $\mathbf{z}$.

Unfortunately, defining the approximate posterior as a Gaussian may not
satisfy this requirement well enough. Consider a two-bit example: we
sample $(x_a,x_b)$ from two independent Bernoulli distributions both
with probability being $0.5$. There are four possibilities. And we want
to learn a VAE on this dataset. If we assume the approximate posteriors
to be Gaussians (see left of Figure
[\[ref:gauss\_vs\_flow\]](#ref:gauss_vs_flow){reference-type="ref"
reference="ref:gauss_vs_flow"}), then we would end up with an aggregate
posterior $q(z) = \sum_{i=1}^4 q(z_i|x_i)\tilde{p}(x_i)$ being a mixture
of Gaussians that fails to be prior-like ($p(z)=\mathcal{N}(z;0,I)$ in
this case). This is because the posterior distributions fail to capture
the true ones, and if our inference network $q(z|x)$ is good enough to
perfectly model the true posteriors, the aggregate posteriors should be
the shape of the prior (see right of Figure
[\[ref:gauss\_vs\_flow\]](#ref:gauss_vs_flow){reference-type="ref"
reference="ref:gauss_vs_flow"}).[^1]

Turns out, the easiest remedy to this is to have a better posterior.
With a bit of rearrangement and the introduction of a few new notations,
the variational gap can be divided into the following three parts:
$$\begin{split}
\log p_{\theta}(x) - &\mathcal{L}(\theta, \phi; x) = \\
&\underset{\text{Gap 1}}{\underbrace{D_{\mathrm{KL}}\left( q^* \| p \right)}} + \underset{\text{Gap 2}}{\underbrace{[D_{\mathrm{KL}}\left( q^*_\theta \| p \right)-D_{\mathrm{KL}}\left( q^* \| p \right)]}} + \underset{\text{Gap 3}}{\underbrace{[D_{\mathrm{KL}}\left( q_\theta \| p \right)-D_{\mathrm{KL}}\left( q^*_\theta \| p \right)]}}
\end{split}$$ where

-   $p$ is the true posterior

-   $q^*$ is the optimal approximate posterior within the family
    $\mathcal{Q}=\{q\in \mathcal{Q}\}$

-   $q^*_\phi$ is the optimal approximate posterior within the family
    $\mathcal{Q}_\phi\subseteq\mathcal{Q}$ due to the amortized
    inference via the conditional mapping
    $\phi\in\Phi:x\rightarrow \pi_q(x;\phi)$ where $\pi_q$ denotes the
    parameters of distribution $q$, and finally

-   $q_\phi$ is the approximate posterior that we are analyzing.

The first gap and the combination of Gap 2 and Gap 3 are what \[9\]
called *approximation gap* and *amortization gap*. With these, we know
that there are three pieces to improve in order to make the variational
gap smaller:

1.  better $\mathcal{Q}$, i.e. the choice of family of distributions.
    This can be achieved by choosing a larger family of distributions
    (such as normalizing flows).

2.  better $\Phi$, i.e. better amortized inference. This can be achieved
    by having a higher capacity encoder.

3.  better $\phi$, i.e. better encoder within the family $\Phi$. This
    can be achieved by improving the optimization scheme.

In Table [\[tb:betterpos\]](#tb:betterpos){reference-type="ref"
reference="tb:betterpos"} we list a few approaches designed to improve
variational inference by making the variational gap smaller.

\centering
                                        Gap 1   Gap 2   Gap 3
  ------------------------------------ ------- ------- -------
  Better optimization                                  
  Better encoder \[9\]                                 
  Normalizing flows \[19, 25, 35\]                     
  Hierarchical VI \[28, 34\]                           
  Implicit VI \[21, 30\]                               
  Variational boosting \[31\]                          
  MCMC as part of $q_\phi$ \[37\]                      
  MCMC on top of $q_\phi$ \[10, 17\]                   
  Gradient flow \[13\]                                 
  Importance sampling \[6\]                            

  : Methods that aim at reducing the variational
  gap.[]{label="tb:betterpos"}

Better likelihood
-----------------

Consider modelling an image using a single latent variable model. In
order to produce a reasonable, realistic looking image, we need to
decide the objects in the image, their location, abstract stuff. But
beyond these abstract ideas, there are also the details, like lighting
and texture that we need to model.

When the probability of the given datapoint is defined as being
factorised over every dimension, the implication is that the
distribution over the observation is completely dependent upon the
latent variable. However, the latent variable has limited capacity, and
may model only the aspects that contribute to most of the reconstruction
loss (abstract concepts) and there may be variations at a lower level of
abstraction that may depend on other parts of the input (details).

Modelling the data this way are what we'll call autoregressive
generative models, and some examples of such models are \[15\] for
images, with a slightly more general discussion found in \[7\], and
\[5\] for text.

\todo{CW: missing VAE w/o pixelwise reconstruction}
Better priors
-------------

\centering
![](figures/mmp.pdf "fig:"){width="50%"} [\[ref:mmp\]]{#ref:mmp
label="ref:mmp"}

Just as we can modify the approximate posterior to fit the prior better,
we can also modify the prior such that it is learned from the data.

The expressivity of neural networks and developments in improving the
posteriors allow the data to be mapped really well to simple priors like
a Gaussian. So if we can already modify the posterior, why bother
changing the prior? One reason may be due to discontinuity when
interpolating in the latent space. Refer again to Figure
[\[ref:gauss\_vs\_flow\]](#ref:gauss_vs_flow){reference-type="ref"
reference="ref:gauss_vs_flow"}. If we imagine each colour code to be a
different modality in the data, moving from one colour group to another
will represent a drastic change in the type of data decoded. This means
that sampling from close regions in the latent space may not represent
close relationships in the data space. Allowing the prior to model a
lower density between these different modalities may be more ideal when
the type of representation learned is more important than simply
sampling data points from the model.

\[41\] proposes parameterising the prior as a mixture over a sample of
the dataset, or using learned *pseudo-inputs*. \[38\] introduces a prior
parameterised by a piece-wise linear function. In \[20\], the authors
parameterise the prior with Real NVP transformations \[11\].

Evaluating the model
====================

Depending on what your metric of model "goodness" is, there may be
different ways to evaluate a given model. Some may care about
disentanglement of factors in the latent variable, while others may care
more about generating realistic looking images[^2]. In this article, we
discuss evaluation in terms of estimating the probability density
defined by the model for a given datapoint.

While we may train the model using the VLB, we need to evaluate
$\log {p}_{} \left( \mathbf{x} \right)$. The procedure for doing this is
simply importance sampling, as described in the previous section,
$$\begin{aligned}
\log {p}_{} \left( \mathbf{x} \right) &= 
\log \mathbb{E}_{{q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right)}\left[ {p}_{} \left( \mathbf{x} \middle| \mathbf{z} \right) \frac{{p}_{} \left( \mathbf{z} \right)}{{q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right)} \right] \\
&\approx \log \sum_{k=1}^K 
{p}_{} \left( \mathbf{x} \middle| \mathbf{z} \right) \frac{{p}_{} \left( \mathbf{z}_k \right)}{{q}_{} \left( \mathbf{z}_k \middle| \mathbf{x} \right)}, &\mathbf{z}_k \sim {q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right)\end{aligned}$$
However, as with most instances in machine learning, the probabilities
are computed in log space, $$\begin{aligned}
&\log \sum_{k=1}^K 
{p}_{} \left( \mathbf{x} \middle| \mathbf{z} \right) \frac{{p}_{} \left( \mathbf{z}_k \right)}{{q}_{} \left( \mathbf{z}_k \middle| \mathbf{x} \right)} =\\
&\qquad \underbrace{\log \sum_{k=1}^K \exp}_\text{There's a trick for that}\left(
\log {p}_{} \left( \mathbf{x} \middle| \mathbf{z} \right) + \log {p}_{} \left( \mathbf{z}_k \right) - \log {q}_{} \left( \mathbf{z}_k \middle| \mathbf{x} \right)
\right)\end{aligned}$$ And we can use the $\mathrm{logsumexp}$ trick to
deal with that sum over the probabilities. This has been mentioned in
and \[36\] Appendix E.

\[26\] Appendix D,

How to Train your VAE
=====================

The purpose of this section is to explicitly introduce what various key
papers on VAEs have demonstrated about modelling data or training using
VAEs.

1.  Design the graphical model,

2.  Decide the types of distributions over the different variables,

3.  Make some simplifying assumptions if it's too complicated,

Semi-supervised settings
------------------------

\todo[inline, color=green!40]{``how'' more in details}
The partially available data need not be a discrete variable, like a
label. It could be parts of the data which are missing in some parts of
the dataset, and making a prediction requires doing some form of
*imputation* over the missing random variables. For conditional
distributions of such missing data that have distributions you can
sample from during training, the same reparameterisation trick can be
applied.

Hierarchy of Latent Variables
-----------------------------

For modeling images, it might be useful to model noise at different
abstractions of the generation process. One way to view this generation
process may be a hierarchy of latent variables, starting at the highest
level of abstraction $z_L$, down to the lowest $z_1$. Consider a VAE
with an inference model $q$,
$$q(z_1, z_2,\dots,z_L | x) = q(z_1|x)q(z_2|z_1) \dots q(z_L|z_{L-1})$$
and generative model,
$$p(x|z_1, z_2,\dots,z_L) = p(z_L) p(z_{L-1}|z_L)\dots p(z_1|z_2)p(x|z_1)$$

One interesting thing to note is that the resulting loss for such a
model (where the generative model is not autoregressive), is that the
reconstruction loss (at the bottom) never directly backpropagates
gradients to the higher layers.

$$\begin{aligned}
&\mathbb{E}_{{q}_{\phi} \left( z_1,\hdots,z_L \middle| x \right)}\left[ \log p(x|z_1) \right] \\
&\qquad  - \mathbb{E}_{{q}_{\phi} \left( z_1,\hdots,z_L \middle| x \right)}\left[ \log {q}_{\phi} \left( z_1,\hdots,z_L \middle| x \right) - 
 \log {p}_{\theta} \left( z_1,\hdots,z_L \right)
 \right] \\
 &= \mathbb{E}_{{q}_{\phi} \left( z_1 \middle| x \right)}\left[{\log p(x|z_1)}\right. \\
&\qquad
    - \mathbb{E}_{{q}_{\phi} \left( z_2 \middle| z_1 \right)}\left[
    \log {q}_{\phi} \left( z_1 \middle| x \right) - 
    \log {p}_{\theta} \left( z_1 \middle| z_2 \right)\right.
\\
&\qquad\qquad\left.\left.
    - \mathbb{E}_{{q}_{\phi} \left( z_3 \middle| z_2 \right)}\left[
    \log {q}_{\phi} \left( z_2 \middle| z_1 \right) - 
    \log {p}_{\theta} \left( z_2 \middle| z_3 \right) \hdots \right]\right]\right]
\\
 &\approx \mathbb{E}_{{q}_{\phi} \left( z_1 \middle| x \right)}\left[ \log p(x|z_1) \right] \\
&\qquad - D_{\mathrm{KL}}\left( {q}_{\phi} \left( z_1 \middle| x \right) \| {p}_{\theta} \left( z_1 \middle| z_2 \right) \right) \\
&\qquad - D_{\mathrm{KL}}\left( {q}_{\phi} \left( z_L \middle| z_{L-1} \right) \| {p}_{} \left( z_L \right) \right) \\
&\qquad - \sum^{L-1}_{l=2} D_{\mathrm{KL}}\left( {q}_{\phi} \left( z_l \middle| z_{l-1} \right) \| {p}_{\theta} \left( z_l \middle| z_{l+1} \right) \right)\end{aligned}$$

As seen in the derivation, the reconstruction term uses only $z_1$, the
first latent variable in the hierarchy, the subsequent layers are
optimised by the KL-divergence terms.

Discrete variables
------------------

One option to deal with discrete latent variables is by replacing it
with a continuous distribution that's "close enough" to the discrete
distribution. Both \[29\] and \[23\] introduce a relaxation of the
discrete distributions using the Gumbel distribution, which a similar
reparameterisation trick can be used in order to sample from it.

Another option is to sample from the distribution, and estimate the
gradient during backpropagation. Some options are the straight-through
estimator \[3\], and REINFORCE \[42\].

Sequential data
---------------

There has been some work that deals with sequential data using VAEs.
\[8\] incorporates latent variables into an RNN. One thing to highlight
here is the possibility of conditioning the prior of the latent variable
on past, seen, data.

In \[39\], the approximate posterior is conditioned on the entire
sequence. This makes sense since if $\mathbf{z}_t$ is responsible for
the observation $\mathbf{x}_t$, and $\mathbf{z}_t$ is dependent on the
*previous* time step's latent variable $\mathbf{z}_{t-1}$, then
$\mathbf{z}_t$ is conditionally dependent on the entire sequence. The
resulting model can still be used as a generative model to generate
sequence, despite the approximate posterior being conditioned on the
full $\mathbf{x}$.

Strong Decoders
---------------

In some autoregressive models, it may be possible for the generative
model to ignore the latent variable. Examples of such models are
language models conditioned on a latent variable \[5\], and PixelVAEs
\[15\]. Because the models already work relatively well without the
global information provided by the latent variable, the KL-divergence
term goes to 0.

One quick remedy for this is to set an annealing term $\beta$ for the
KL-divergence term,
$$\mathbb{E}_{{q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right)}\left[ \log {p}_{} \left( \mathbf{x} \middle| \mathbf{z} \right) \right]
- \beta \cdot D_{\mathrm{KL}}\left( {q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right) \| \log {p}_{} \left( \mathbf{z} \right) \right).$$
This has been given some theoretical justification in \[16\] and \[2\],
where interpretations of $\beta$ are given.

Another possibility is to "weaken" the generative model. For example, in
\[43\], instead of an RNN for text generation, they used a CNN for the
decoder.

Wherever possible, reduce variance
----------------------------------

One of the core motivations for the contributions in \[26\] was methods
for variance reduction. Whenever sampling is performed, variance is
introduced. Before, methods for backpropagating through random samples
involved gradient estimates, like \[42\], \[4\] and \[33\].[^3] The
issue with these gradient estimates is usually high variance. The
reparameterisation trick ameliorates the problem as the variance is
bounded by a constant (More details can be found in the appendix of
\[36\].)

Recall in the previous explanation about the ELBO that the entire
purpose of introducing $q$ in training is variance reduction. Given that
we are using SGD as our training method, the importance of doing this
cannot be understated.

We *may* still be able to further reduce variance during training if we
are able to analytically compute parts of the terms in this lower-bound.
Notice, $$\begin{aligned}
\mathbb{E}_{{q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right)}\left[ \log{p}_{} \left( \mathbf{x} \middle| \mathbf{z} \right) \frac{{p}_{} \left( \mathbf{z} \right)}{{q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right)} \right] 
&= \mathbb{E}_{{q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right)}\left[ \log {p}_{} \left( \mathbf{x} \middle| \mathbf{z} \right) \right] + 
\underbrace{\mathbb{E}_{{q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right)}\left[ \log \frac{{p}_{} \left( \mathbf{z} \right)}{{q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right)} \right]}_{\text{Look familiar?}} \\
&= \mathbb{E}_{{q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right)}\left[ \log {p}_{} \left( \mathbf{x} \middle| \mathbf{z} \right) \right] - 
\underbrace{\mathbb{E}_{{q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right)}\left[ \log \frac{{q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right)}{{p}_{} \left( \mathbf{z} \right)} \right]}_{\text{What about now?}} \\
&= \mathbb{E}_{{q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right)}\left[ \log {p}_{} \left( \mathbf{x} \middle| \mathbf{z} \right) \right] - 
D_{\mathrm{KL}}\left( {q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right) \| \log {p}_{} \left( \mathbf{z} \right) \right)\end{aligned}$$
If both ${q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right)$ and
${p}_{} \left( \mathbf{z} \right)$ are parameterised as Gaussians, for
example, as they are in the original VAE paper, then there is a closed
form for this KL-Divergence term.

\todo{CW: missing new paper ``Reducing Reparameterization Gradient Variance'' by miller NIPS2017}
\appendix
More derivations
================

$$\begin{aligned}
\log {p}_{} \left( \mathbf{x} \right) 
&=  \log \int {p}_{} \left( \mathbf{x} \middle| \mathbf{z} \right) {p}_{} \left( \mathbf{z} \right) \mathrm{d}\mathbf{z}\\
&= \log \int {q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right) \frac{{p}_{} \left( \mathbf{x} \middle| \mathbf{z} \right) {p}_{} \left( \mathbf{z} \right)}{{q}_{} \left( \mathbf{x} \middle| \mathbf{z} \right)} \mathrm{d}\mathbf{z}\\
&= \log \mathbb{E}_{{q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right)}\left[ 
\frac{{p}_{} \left( \mathbf{x} \middle| \mathbf{z} \right){p}_{} \left( \mathbf{z} \right)}{{q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right)} \right] \\
&\geq \mathbb{E}_{{q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right)}\left[ \log 
\frac{{p}_{} \left( \mathbf{x} \middle| \mathbf{z} \right){p}_{} \left( \mathbf{z} \right)}{{q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right)} \right] \\
&= \mathbb{E}_{{q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right)}\left[ \log {p}_{} \left( \mathbf{x} \middle| \mathbf{z} \right) \right] - D_{\mathrm{KL}}\left( {q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right) \| {p}_{} \left( \mathbf{z} \right) \right)\end{aligned}$$

$$\begin{aligned}
D_{\mathrm{KL}}\left( {q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right) \| {p}_{} \left( \mathbf{z} \middle| \mathbf{x} \right) \right)
&= \mathbb{E}_{{q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right)}\left[ \log 
\frac{{q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right) }{{p}_{} \left( \mathbf{z} \middle| \mathbf{x} \right)} \right] \\
&= \mathbb{E}_{{q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right)}\left[ 
    \log {q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right) 
    \frac{{p}_{} \left( \mathbf{x} \right)}{{p}_{} \left( \mathbf{x} \middle| \mathbf{z} \right){p}_{} \left( \mathbf{z} \right)} \right] \\
&= \mathbb{E}_{{q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right)}\left[ 
    \log \frac{{q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right)}{{p}_{} \left( \mathbf{z} \right)} 
    \frac{{p}_{} \left( \mathbf{x} \right)}{{p}_{} \left( \mathbf{x} \middle| \mathbf{z} \right)} \right] \\
&=\mathbb{E}_{{q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right)}\left[ 
    \log \frac{{q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right)}{{p}_{} \left( \mathbf{z} \right)} \right] -
   \mathbb{E}_{{q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right)}\left[ \log {p}_{} \left( \mathbf{x} \middle| \mathbf{z} \right) \right] + \log {p}_{} \left( \mathbf{x} \right) \\
\log {p}_{} \left( \mathbf{x} \right) -
\underbrace{D_{\mathrm{KL}}\left( {q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right) \| {p}_{} \left( \mathbf{z} \middle| \mathbf{x} \right) \right)}_{
    \text{Variational gap}
}
&= \mathbb{E}_{{q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right)}\left[ \log {p}_{} \left( \mathbf{x} \middle| \mathbf{z} \right) \right] - D_{\mathrm{KL}}\left( {q}_{} \left( \mathbf{z} \middle| \mathbf{x} \right) \| {p}_{} \left( \mathbf{z} \right) \right) \end{aligned}$$

#### Glossary

-   VAE: variational autoencoder

-   DLGM: deep latent gaussian model

-   LVM: latent variable model

-   DNN: deep neural networks

#### Notation

-   $\tilde{p}$: unknown distribution

-   $\mathbb{E}_{x\sim p(x)}\left[ f(x) \right]=\int_x p(x)f(x)dx$:
    expected value of $f(x)$, a deterministic feature of $x$, under the
    distribution $p(x)$

-   $D_{\mathrm{KL}}\left( {q}_{} \left( x \right) \| {p}_{} \left( x \right) \right) = \int_x q(x)\log\frac{q(x)}{p(x)}$:
    KL divergence between $q$ and $p$

\bibliographystyle{plainnat}
::: {#refs .references}
::: {#ref-agrawal2016deep}
\[1\] Agrawal, S. and Dukkipati, A. 2016. Deep variational inference
without pixel-wise reconstruction. *arXiv preprint arXiv:1611.05209*.
(2016).
:::

::: {#ref-alemi2017information}
\[2\] Alemi, A.A. et al. 2017. An information-theoretic analysis of deep
latent-variable models. *arXiv preprint arXiv:1711.00464*. (2017).
:::

::: {#ref-bengio2013estimating}
\[3\] Bengio, Y. et al. 2013. Estimating or propagating gradients
through stochastic neurons for conditional computation. *arXiv preprint
arXiv:1308.3432*. (2013).
:::

::: {#ref-bonnet1964transformations}
\[4\] Bonnet, G. 1964. Transformations des signaux aléatoires a travers
les systemes non linéaires sans mémoire. *Annals of Telecommunications*.
19, 9 (1964), 203--220.
:::

::: {#ref-bowman2015generating}
\[5\] Bowman, S.R. et al. 2015. Generating sentences from a continuous
space. *arXiv preprint arXiv:1511.06349*. (2015).
:::

::: {#ref-burda2015importance}
\[6\] Burda, Y. et al. 2015. Importance weighted autoencoders. *arXiv
preprint arXiv:1509.00519*. (2015).
:::

::: {#ref-chen2016variational}
\[7\] Chen, X. et al. 2016. Variational lossy autoencoder. *arXiv
preprint arXiv:1611.02731*. (2016).
:::

::: {#ref-chung2015recurrent}
\[8\] Chung, J. et al. 2015. A recurrent latent variable model for
sequential data. *Advances in neural information processing systems*
(2015), 2980--2988.
:::

::: {#ref-cremer2017inference}
\[9\] Cremer, C. et al. 2017. Inference suboptimality in variational
autoencoders. (2017).
:::

::: {#ref-de2001variational}
\[10\] De Freitas, N. et al. 2001. Variational mcmc. *Proceedings of the
seventeenth conference on uncertainty in artificial intelligence*
(2001), 120--127.
:::

::: {#ref-dinh2016density}
\[11\] Dinh, L. et al. 2016. Density estimation using real nvp. *arXiv
preprint arXiv:1605.08803*. (2016).
:::

::: {#ref-dumoulin2016adversarially}
\[12\] Dumoulin, V. et al. 2016. Adversarially learned inference. *arXiv
preprint arXiv:1606.00704*. (2016).
:::

::: {#ref-duvenaud2016early}
\[13\] Duvenaud, D. et al. 2016. Early stopping as nonparametric
variational inference. *Artificial intelligence and statistics* (2016),
1070--1077.
:::

::: {#ref-edwards2016towards}
\[14\] Edwards, H. and Storkey, A. 2016. Towards a neural statistician.
*arXiv preprint arXiv:1606.02185*. (2016).
:::

::: {#ref-gulrajani2016pixelvae}
\[15\] Gulrajani, I. et al. 2016. Pixelvae: A latent variable model for
natural images. *arXiv preprint arXiv:1611.05013*. (2016).
:::

::: {#ref-higgins2016beta}
\[16\] Higgins, I. et al. 2016. Beta-vae: Learning basic visual concepts
with a constrained variational framework. (2016).
:::

::: {#ref-hoffman2017learning}
\[17\] Hoffman, M.D. 2017. Learning deep latent gaussian models with
markov chain monte carlo. *International conference on machine learning*
(2017), 1510--1519.
:::

::: {#ref-hoffman2016elbo}
\[18\] Hoffman, M.D. and Johnson, M.J. 2016. Elbo surgery: Yet another
way to carve up the variational evidence lower bound. (2016).
:::

::: {#ref-huang2017facilitating}
\[19\] Huang, C.-W. et al. 2017. Facilitating multimodality in
normalizing flows. (2017).
:::

::: {#ref-huang2017led}
\[20\] Huang, C.-W. et al. 2017. Learnable Explicit Density for
Continuous Latent Space and Variational Inference. *ArXiv e-prints*.
(Oct. 2017).
:::

::: {#ref-huszar2017variational}
\[21\] Huszár, F. 2017. Variational inference using implicit
distributions. *arXiv preprint arXiv:1702.08235*. (2017).
:::

::: {#ref-anom2017hali}
\[22\] ICLR 2018), A. (submitted to 2018. Hierarchical adversarially
learned inference. (2018).
:::

::: {#ref-jang2016categorical}
\[23\] Jang, E. et al. 2016. Categorical reparameterization with
gumbel-softmax. *arXiv preprint arXiv:1611.01144*. (2016).
:::

::: {#ref-kingma2014semi}
\[24\] Kingma, D.P. et al. 2014. Semi-supervised learning with deep
generative models. *Advances in neural information processing systems*
(2014), 3581--3589.
:::

::: {#ref-kingma2016improving}
\[25\] Kingma, D.P. et al. 2016. Improving variational inference with
inverse autoregressive flow. *arXiv preprint arXiv:1606.04934*. (2016).
:::

::: {#ref-kingma2013auto}
\[26\] Kingma, D.P. and Welling, M. 2013. Auto-encoding variational
bayes. *arXiv preprint arXiv:1312.6114*. (2013).
:::

::: {#ref-louizos2017causal}
\[27\] Louizos, C. et al. 2017. Causal Effect Inference with Deep
Latent-Variable Models. *ArXiv e-prints*. (May 2017).
:::

::: {#ref-maaloe2016auxiliary}
\[28\] Maaløe, L. et al. 2016. Auxiliary deep generative models. *arXiv
preprint arXiv:1602.05473*. (2016).
:::

::: {#ref-maddison2016concrete}
\[29\] Maddison, C.J. et al. 2016. The concrete distribution: A
continuous relaxation of discrete random variables. *arXiv preprint
arXiv:1611.00712*. (2016).
:::

::: {#ref-mescheder2017adversarial}
\[30\] Mescheder, L. et al. 2017. Adversarial variational bayes:
Unifying variational autoencoders and generative adversarial networks.
*arXiv preprint arXiv:1701.04722*. (2017).
:::

::: {#ref-miller2016variational}
\[31\] Miller, A.C. et al. 2016. Variational boosting: Iteratively
refining posterior approximations. *arXiv preprint arXiv:1611.06585*.
(2016).
:::

::: {#ref-oord2016pixel}
\[32\] Oord, A. van den et al. 2016. Pixel recurrent neural networks.
*arXiv preprint arXiv:1601.06759*. (2016).
:::

::: {#ref-price1958useful}
\[33\] Price, R. 1958. A useful theorem for nonlinear devices having
gaussian inputs. *IRE Transactions on Information Theory*. 4, 2 (1958),
69--72.
:::

::: {#ref-ranganath2016hierarchical}
\[34\] Ranganath, R. et al. 2016. Hierarchical variational models.
*International conference on machine learning* (2016), 324--333.
:::

::: {#ref-rezende2015variational}
\[35\] Rezende, D.J. and Mohamed, S. 2015. Variational inference with
normalizing flows. *arXiv preprint arXiv:1505.05770*. (2015).
:::

::: {#ref-rezende2014stochastic}
\[36\] Rezende, D.J. et al. 2014. Stochastic backpropagation and
approximate inference in deep generative models. *arXiv preprint
arXiv:1401.4082*. (2014).
:::

::: {#ref-salimans2015markov}
\[37\] Salimans, T. et al. 2015. Markov chain monte carlo and
variational inference: Bridging the gap. *Proceedings of the 32nd
international conference on machine learning (icml-15)* (2015),
1218--1226.
:::

::: {#ref-serban2016multi}
\[38\] Serban, I.V. et al. 2016. Multi-modal variational
encoder-decoders. *arXiv preprint arXiv:1612.00377*. (2016).
:::

::: {#ref-shabanian2017variational}
\[39\] Shabanian, S. et al. 2017. Variational bi-lstms. *arXiv preprint
arXiv:1711.05717*. (2017).
:::

::: {#ref-sonderby2016ladder}
\[40\] Sønderby, C.K. et al. 2016. Ladder variational autoencoders.
*Advances in neural information processing systems* (2016), 3738--3746.
:::

::: {#ref-tomczak2017vae}
\[41\] Tomczak, J.M. and Welling, M. 2017. VAE with a vampprior. *arXiv
preprint arXiv:1705.07120*. (2017).
:::

::: {#ref-williams1988use}
\[42\] Williams, R.J. 1988. On the use of backpropagation in associative
reinforcement learning. *Proceedings of the ieee international
conference on neural networks* (1988), 263--270.
:::

::: {#ref-yang2017improved}
\[43\] Yang, Z. et al. 2017. Improved variational autoencoders for text
modeling using dilated convolutions. *arXiv preprint arXiv:1702.08139*.
(2017).
:::
:::

[^1]: See \[18\] or \[20\] for the derivation

[^2]: VAEs do poorly at this\... for now. (circa Jan 2018)

[^3]: As you can see, these go pretty far back.
