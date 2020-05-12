# Introduction

This document is an attempt to summarize my recent work related to the design of
machine learning methods specifically tailored to handle structured data such
as graphs (in [Sec. 1.3](01/ot.html)) or time series (in the rest of the
document).
However, note that one of my contributions to the field is not developed in
this document (or just marginally in its
[Jupyter notebook](https://rtavenar.github.io/hdr/) form). It concerns open
source software development, especially through the creation and
maintenance of [`tslearn`](https://tslearn.readthedocs.io)
{% cite tslearn %}.[^1]

I realize while writing this document that, over the past few years, I
have treated time series as if they were several different things.
First, from an application point of view, I have worked with video recordings
during my post-doc at Idiap and moved to earth observation time series
(be it pollutant levels in water streams, satellite image time series or ship
trajectories) when I joined LETG (_Littoral, Environnement, Géomatique,
Télédétection_) in 2013.
Most importantly, these diverse applications have lead to different views
over what time series can be and these views are connected to how the temporal
nature of the data is included (or not) in the representation.
In **TODO ref Gloaguen**, for the sake of efficiency, we have relied on a fully
non-temporal pre-clustering of the data so as to be able, in a refinement step,
to model series segments using a continuous-time model (hence re-introducing
temporal information at the sub-segment level).
At the other extreme of the spectrum, we have
{% cite guilleme:hal-02513295 %} and {% cite tavenard:halshs-01561461 %},
in which we have postulated that temporal localization information is key for
prediction.
In these works, we have hence used timestamps as additional features of the
input data.
Elastic alignment-based approaches (such as the well-known Dynamic Time Warping)
somehow belong somewhere in-between those two extremes.
Indeed, they only rely on temporal ordering
(not on timestamps) to assess similarity between series.
Note also that, compared to other approaches considered in this document,
convolutional models presented in [Sec. 2.2](02/shapelets_cnn.html) make an
extra assumption about the regularity of the sampling process.
I have, more recently, turned my focus to other structured data such as graphs,
and it appears that choosing an adequate encoding for the structural information
in these cases is also a very important question.
This study relies on the use of Optimal Transport distances that, surprisingly
or not, use formulations that are very similar in spirit to the one of
Dynamic Time Warping.

Coming back to the current document, my contributions are organized in two
parts, the first one being dedicated to the design of adequate similarity
measures between structured data (_i.e._ graphs and time series), while the
second one focuses on methods that
learn latent representations for temporal data.

## Notations

Throughout this document, the following notations will be used.

A time series is a set of $n$ timestamped features:

\begin{equation}
    \textbf{x} = \{ (x_0, t_0), \dots , (x_{n-1}, t_{n-1}) \}
\end{equation}

where all $x_i$ lie in the same ambient space $\mathbb{R}^{p}$.
Time series datasets will be denoted $(\mathbf{X}, \mathbf{y})$ (or just
$\mathbf{X}$ for unsupervised methods) where
$\mathbf{X} = \left( \mathbf{x}^{(0)}, \cdots, \mathbf{x}^{(N-1)} \right)$ is
a set of $N$ time series (that do not necessarily share the same length) and
$\mathbf{y}$ is a vector of target values.

When subseries have to be considered, we will denote by
$\mathbf{x}_{i \rightarrow j}$ the subseries extracted from $\mathbf{x}$ that
starts at time index $i$ and stops at time index $j$ (excluded), and
$\mathbf{x}_{\rightarrow j}$ will be a shortcut notation for the subseries that
covers indexes 0 to $j-1$.


## References

{% bibliography --cited %}

[^1]: `tslearn` is a general-purpose Python machine learning library
    for time series that offers tools for pre-processing and feature extraction
    as well as dedicated models for clustering, classification and regression,
    and I started this project in 2017.
