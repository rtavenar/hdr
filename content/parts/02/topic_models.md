---
jupyter:
  jupytext:
    formats: md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Temporal Topic Models

Topic models are mixture models that can deal with documents represented as
bags of features (BoF) and extract latent topics (a topic being a distribution
over features) from a corpus of documents.
For these methods, time series are hence seen as bags of timestamped features.
In the methods presented here, the temporal dimension is either
[included in the BoF representation](#Supervised-Hierarchical-Dirichlet-Latent-Semantic-Motifs)
or added
[in a refinement step](#Two-step-Inference-for-Sequences-of-Ornstein-Uhlenbeck-Processes).

## Supervised Hierarchical Dirichlet Latent Semantic Motifs

In this work, we build upon the Hierarchical Dirichlet Latent Semantic Motifs
(HDLSM) topic model that was first introduced in {% cite EmonetCVPR2011 %}.
This generative model relies on the extraction of motifs that encapsulate the
temporal information of the data.
It is able to automatically find both the underlying number of motifs needed to
model a given set of documents and the number of motif occurrences in each
document (which includes their temporal locations), as shown in the following
Figure:

![half-width](../../images/hdlsm.svg)

The HDLSM model takes as input a set of quantized time series (aka temporal
documents).
More specifically, a time series is represented as a table of counts that
informs, for
each pair $(w, t)$, whether word $w$ (typically a quantized feature) was
present in the time series at time index $t$ (in fact, it can also account for
the _amount_ of presence of word $w$ at time $t$).

HDLSM is a generative model whose generative process can be described as
follows:

1. Generate a list of motifs, each motif $k$ being a 2D probability map
indicating how likely it is that word $w$ occurs at relative time $t_r$ after
the beginning of the motif.
2. For each document $j$, generate a list of occurrences, each occurrence having
a starting time $t_o$ and an associated motif $k$.
3. For each observation $i$ in document $j$:
  * Draw an occurrence from the list,
  * Draw a pair $(w, t_r)$ from the associated motif,
  * Generate the observation of word $w$ at time $t = t_o + t_r$.

As stated above, motifs are represented as probabilistic maps.
Each map is drawn from a Dirichlet distribution.
This models makes intensive use of Dirichlet Processes (DP) to model the
possibly infinite number of motifs and occurrences.

To learn the parameters of the model, a Gibbs sampling is applied, in which it
is sufficient to re-sample motif assignments for both observations and
occurrences and starting time for each motif occurrence.
Other variables are either integrated out or deduced, when a deterministic
relation holds.

Our supervised variant relies on the same generative process except that an
extra component is added that maps motifs to classes in a supervised learning
context.
Therefore, this mapping needs to be learned and, once the model is trained,
classifying a new instance $\mathbf{x}$ consists in
(i) extracting motif probabilities $P(z | \mathbf{x})$ and
(ii) deriving class probabilities as:

\begin{equation}
    P(y | \mathbf{x}) = \sum_z P(y | z) P(z | \mathbf{x})
\end{equation}

We have used this model in the context of action recognition in videos
{% cite tavenard:hal-00872048 %}.
Here, our _words_ are quantized spatio-temporal features and each time series
is the encoding of a video in which a single action is performed.
In this context, we show that our
model outperforms standard competitors that operate on the same quantized
features.

## Two-step Inference for Sequences of Ornstein Uhlenbeck Processes

<!-- #region {"tags": ["popout"]} -->
**Note.** This work is part of Pierre Gloaguen's postdoc.
This is joint work with Laetitia Chapel and Chloé Friguet.
<!-- #endregion -->

More recently, I have been involved in a project related to the surveillance of
the maritime traffic.
In this context, a major challenge
is the automatic identification of traffic flows from a set of observed
trajectories, in order to derive good management measures or to detect abnormal
or illegal behaviors for example.

The model we have proposed in this context differs from the one described above
in several aspects:

* we are not in a supervised framework, we have no labelled data at our disposal
and our goal will rather be to extract meaningful trajectory clusters;
* we are not looking for motifs to be localized in time series (with
a possible overlap between motifs, as in the method described above) but rather
in the segmentation of trajectories into homogeneous _movement modes_;
* each movement mode will be described using a continuous time model;
* in order to scale to larger datasets, stochastic variational inference is used
(in place of Gibbs sampling) for inference.

### Use case

The monitoring of maritime traffic relies on several sources of data, in a
rising context of maritime big data {% cite garnier2016exploiting %}.
Among these sources lies the Automatic Identification System (AIS), which
automatically collects messages from vessels around the world, at a high
frequency.
AIS data basically consist in GPS-like data, together with the instantaneous
speed and heading, and some vessel specific static information.
These data are characterized by their diversity as they (1) are collected at
different frequencies (2) have different lengths (3) are not necessarily
regularly sampled (4) represent very different behaviors, but (5) share common
trends or similar subparts (called hereafter _movement modes_).

One major challenge in this context is the extraction of movement patterns
emerging from the observed data, considering trajectories that share similar
movement modes.
This issue can be restated from a machine learning point of view as a
large-scale clustering task involving the definition of clustering methods
that can handle such complex data while being efficient on large databases,
and that both cluster trajectories as a whole and detect common
sub-trajectories.

### The model

We define a parametric framework to model trajectory data,
_i.e._ sequences of geographical positions recorded through time.
The modeling framework aims to account for two levels of heterogeneity possibly
present in trajectory data:

1. heterogeneity of an individual's movement within a single trajectory, and
2. heterogeneity between observed trajectories of several individuals.

Following a common paradigm, we assume that a moving individual's trajectory
is a heterogeneous sequence of patterns that we call _movement modes_.
Different movement modes along a trajectory refer to different ways of moving
in terms of velocity distribution, reflecting different behaviors, activities,
or routes.
It is assumed that a given movement mode can be adopted by several individuals.

As done in {% cite gurarie2017correlated %}, we characterize
movement modes using a specific correlated velocity model, defined in a
continuous-time framework, namely the Ornstein-Uhlenbeck Process
{% cite uhlenbeck1930theory %} (OUP).
One important property of the OUP is that, under mild conditions,
the velocity process is an asymptotically stationary Gaussian Process.

```python tags=["hide_input"]
%config InlineBackend.figure_format = 'svg'
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

plt.ion()
```

```python
def simulate_oup(V0, mu, gamma, sigma, delta_t, n_samples):
    """An Ornstein Uhlenbeck is a solution of:

    dX(t) = \Gamma (X(t)−μ) dt + \Sigma dW(t), X0=x0

    In this case, the solution is a Markov process Markov, with an explicit
    transition law:

    X(t+\Delta) | {X(t)=x_t} \sim \mathcal{N}(m_\Delta,V_\Delta)

    with:

    m_\Delta = \mu + \exp(−\Gamma \Delta)(x_t−\mu)

    V_\Delta = S − \exp(−\Gamma \Delta) S \exp(−\Gamma \Delta)^T

    vec(S)=(\Gamma \oplus \Gamma) − 1 vec(\Sigma \Sigma^T)
    """
    d = mu.shape[0]

    exp_g = scipy.linalg.expm(-delta_t * gamma)

    S = scipy.linalg.solve(
        np.kron(gamma, np.eye(d)) + np.kron(np.eye(d), gamma),
        sigma.dot(sigma.T).reshape((-1, ))
    ).reshape((d, d))
    v_delta = S - exp_g.dot(S).dot(exp_g.T)

    samples = np.empty((n_samples, d))
    for t in range(n_samples):
        V_prev = V0 if t == 0 else samples[t - 1]
        m_delta = mu + np.dot(exp_g, V_prev)
        samples[t] = np.random.multivariate_normal(mean=m_delta,
                                                   cov=v_delta)
    return samples


np.random.seed(0)

dt = .05
n_times = 300  # Number of timestamps
d = 2

# UOP parameters
mu_k = np.array([0., 0.])
gamma_k = np.array([[3., 1.], [-1., 2.]])
sigma_k = np.diag([.5, 2.])

Vt = np.empty((n_times + 1, d))
Vt[0] = np.array([10., 10.])
Vt[1:] = simulate_oup(V0=Vt[0], mu=mu_k, gamma=gamma_k, sigma=sigma_k,
                      delta_t=dt, n_samples=n_times)

plt.figure(figsize=(6, 6))
plt.plot(Vt[:, 0], Vt[:, 1], 'rx-', zorder=0)
plt.scatter([mu_k[0]], [mu_k[1]], color='k', zorder=1)
plt.text(x=mu_k[0] + .3, y=mu_k[1] + .3, s="$\mu$", fontsize=16)
plt.xlabel("X-Velocity")
plt.ylabel("Y-Velocity")
plt.show()
```

### Parameter estimation

In order to perform scalable parameter inference and clustering of both
trajectories and GPS observations (into movement modes), we adopt a pragmatic
two step approach that takes advantage of the inherent properties of the OUP:

1. A first dual clustering is performed based on a simpler independent
Gaussian mixture model, in order to estimate potential movement modes and
trajectory clusters: it allows getting rid of within mode autocorrelation in
the inference, and therefore eases the computations, yet it does not rely on any
temporal or sequential information.
Here again, we use a Hierarchical Dirichlet Process as a model for this
two-level clustering, hence allowing for infinite mixtures of both movement
modes and trajectory clusters.
The Gaussian hypothesis in this case is in line with our choice of the OUP as
our velocity process, since the OUP stationary distribution is Gaussian.
2. Among the estimated movement modes, only those meeting a temporal consistency
constraint are kept.
Parameters of these consistent movement modes are then estimated, and used to
reassign observations that were assigned to inconsistent movement modes (_i.e._
movement modes that do not last long enough to be considered reliable).
It ensures that only trajectory segments for which the stationary distribution
is reached are kept to estimate movement modes.

The resulting consistent movement mode concept allows one to (1) have a good
estimation of OUP parameters within a movement mode (as a consistent sequence
will often be related to a large amount of points) and (2) filter out
"noise" movement modes gathering few observations in a temporally
inconsistent manner.

Parameter estimation for step 1. described above is performed through stochastic
variational inference (SVI) to allow scalability to large datasets of AIS data,
and movement mode parameter estimation is performed using standard tools from
the OUP literature.

Computational complexity of the inference step is dominated by the clustering
step, since the OUP parameter estimation can be performed independently for each
movement mode.
It is quasilinear in the number of
observations and, as stochastic variational inference is used, parts of the
computations involved can be distributed.

### Results

We have provided [a dataset](https://github.com/rtavenar/ushant_ais) of several
millions of observations in the AIS context.
This dataset is used to validate our model qualitatively (through visual
analysis of extracted movement modes and trajectory clusters) and should
allow future competitive methods to compare on a real-world large-scale
trajectory dataset.

**TODO: add ref to online tech report**

## References

{% bibliography --cited %}
