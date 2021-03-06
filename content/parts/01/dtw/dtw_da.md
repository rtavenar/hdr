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

# DTW Alignment as an Adaptive Resampling Strategy

<!-- #region {"tags": ["popout"]} -->
**Note.** This work is a part of Rémi Dupas' PhD thesis (in Environment
Sciences).
I was not directly involved in the supervision of Rémi's PhD thesis.
<!-- #endregion -->

In this section, we present a method that uses Dynamic Time Warping (DTW)
on multimodal time series, _i.e._ time series that are made of several
features recorded over time.
The method relies on the assumption that one of the considered modalities
(called
reference modality in the following) can be used as a reference to (temporally)
realign other modalities {% cite dupas:halshs-01228397 %}.
It has been used in the context of hydrological measurements to align pollutant
concentration profiles based on discharge time series.

This approach can be seen as the DTW counterpart of other works that rely on
Optimal Transport for Domain Adaptation {% cite courty:hal-02112785 %}.
One significant difference, however, is that it relies on a reference modality
for
alignment.
This design choice is guided by our application context.

## Motivating Use Case

Phosphorus (P) transfer during storm events represents a significant part of
annual P loads in streams and contributes to eutrophication in downstream water
bodies. To improve understanding of P storm dynamics, automated or
semi-automated methods are needed to extract meaningful information from
ever-growing water quality measurement datasets.

Clustering techniques have proven useful for identifying seasonal storm
patterns and thus for increasing knowledge about seasonal variability in storm
export mechanisms (_e.g._, {% cite aubert:halshs-00906292 %}).
Clustering techniques usually require calculating distances between pairs of
comparable points in multiple time series. For this reason, direct clustering
(without using hysteresis-descriptor variables) of high-frequency storm
concentration time series is usually irrelevant because the lengths of recorded
time series (number of
measurement points) might differ and/or measurement points may have different
positions relative to the hydrograph (flow rise and recession); hence, it is
difficult to calculate a distance between pairs of comparable points.

The aim of this study was to develop a clustering method that overcomes this
limit and test its ability to compare seasonal variability of P storm dynamics
in two headwater watersheds. Both watersheds are ca. 5 km², have similar
climate and geology, but differ in land use and P pressure intensity.

## Alignment-based Resampling Method

In the above-described setting, we have access to one modality (discharge,
commonly denoted $Q$) that is representative of the evolution of the flood.
Temporal realignment based on this modality allows to overcome three
difficulties that can arise when comparing storm-event data.
Indeed, time series can have

1. different starting times due to the discharge threshold at which the
samplers were triggered,
2. different lengths, and
3. differences in phase that yield different temporal localizations of the
discharge peak.

To align time series, we use the path associated with DTW.
This matching path can be viewed as the optimal way to perform point-wise
alignment of time series.

For each discharge time series $\mathbf{x}^{(i)}_\text{Q}$, we compute the
matching path $\pi_\text{Q}$ and use it to find the optimal alignment wrt.
the same reference discharge time series $\mathbf{x}^\text{ref}_\text{Q}$.
The reference discharge time series used in this study is chosen
as a storm event with full coverage of flow rise and flow recession phases.
Alternatively, one could choose a synthetic idealized storm hydrograph.

We then use barycentric mapping based on the obtained matches to realign other
modalities to the timestamps of the reference time series, as shown in the
following Figures:

```python tags=["hide_input"]
%config InlineBackend.figure_format = 'svg'
import matplotlib.pyplot as plt
import numpy
from tslearn.utils import to_time_series

plt.ion()

def plot_matches(ts0, ts1, ts0_resample, path):
    offset = .5
    fig = plt.figure(figsize=(6, 3))

    plt.subplot(1, 2, 1)
    plt.text(x=0.5, y=1.05,
             s="Original discharge time series", fontsize=12, horizontalalignment='center',
             verticalalignment='center', transform=plt.gca().transAxes)
    plt.text(x=0.5, y=-0.05,
             s="Reference discharge time series", fontsize=12, horizontalalignment='center',
             verticalalignment='center', transform=plt.gca().transAxes)

    # Plot series (with pos/neg offset for visu reasons)
    plt.plot(numpy.arange(ts0.shape[0]), ts0.ravel() + offset,
             color='k', linestyle='-', linewidth=2.)
    plt.plot(numpy.arange(ts1.shape[0]), ts1.ravel() - offset,
             color='k', linestyle='-', linewidth=2.)

    # Plot matches
    for (i, j) in path:
        if [pair[0] for pair in path].count(i) > 1:
            plt.plot([i, j], [ts0[i, 0] + offset, ts1[j, 0] - offset],
                     color="blue", marker='o', linestyle="dashed")
        elif [pair[1] for pair in path].count(j) > 1:
            plt.plot([i, j], [ts0[i, 0] + offset, ts1[j, 0] - offset],
                     color="red", marker='o', linestyle="dashed")
        else:
            plt.plot([i, j], [ts0[i, 0] + offset, ts1[j, 0] - offset],
                     color="grey", marker='o', linestyle="dashed")

    plt.xticks([])
    plt.yticks([])
    plt.gca().axis("off")

    plt.subplot(1, 2, 2)
    plt.text(x=0.5, y=1.05,
             s="Resampled discharge time series", fontsize=12, horizontalalignment='center',
             verticalalignment='center', transform=plt.gca().transAxes)
    plt.text(x=0.5, y=-0.05,
             s="Reference discharge time series", fontsize=12, horizontalalignment='center',
             verticalalignment='center', transform=plt.gca().transAxes)

    # Plot series (with pos/neg offset for visu reasons)
    plt.plot(numpy.arange(ts0_resample.shape[0]), ts0_resample.ravel() + offset,
             color='k', linestyle='-', linewidth=2.)
    plt.plot(numpy.arange(ts1.shape[0]), ts1.ravel() - offset,
             color='k', linestyle='-', linewidth=2.)

    # Plot matches
    for j in range(len(ts1)):
        if [pair[1] for pair in path].count(j) > 1:
            plt.plot([j, j], [ts0_resample[j, 0] + offset, ts1[j, 0] - offset],
                     color="red", marker='o', linestyle="dashed")
        else:
            pair = path[[pair[1] for pair in path].index(j)]
            i = pair[0]
            if [pair[0] for pair in path].count(i) > 1:
                plt.plot([j, j], [ts0_resample[j, 0] + offset, ts1[j, 0] - offset],
                         color="blue", marker='o', linestyle="dashed")
            else:
                plt.plot([j, j], [ts0_resample[j, 0] + offset, ts1[j, 0] - offset],
                         color="grey", marker='o', linestyle="dashed")
    plt.xticks([])
    plt.yticks([])
    plt.gca().axis("off")   
    plt.tight_layout()
```

```python
from tslearn.metrics import dtw_path

x_q_ref = to_time_series(
    [0.13991821, 0.16294979, 0.31514145, 0.54636252, 0.69737061, 0.87776431,
     0.95917049, 0.99667355, 0.98113988, 0.87307521, 0.70341944, 0.59648599,
     0.51890249, 0.43674822, 0.38792677, 0.36107532, 0.32893154, 0.30836181,
     0.30146932, 0.27417169]
)

x_prime_q = to_time_series(
    [0.12127299, 0.12750528, 0.14748864, 0.17853797, 0.2815324 , 0.3848446,
     0.51661235, 0.6876372 , 0.83539414, 0.96088103, 1.        , 0.82093283,
     0.70602368, 0.56334187, 0.47268893, 0.41283418, 0.3747808 , 0.34633213,
     0.32026957, 0.30550197]
)

x_prime_srp = to_time_series(
    [0.26215067, 0.14032423, 0.07405513, 0.08556629, 0.07101746, 0.0891955 ,
     0.22119012, 0.32734859, 0.41433779, 0.43256379, 0.56561361, 0.81348724,
     0.93016563, 0.92843896, 0.71375583, 0.55979408, 0.43102897, 0.32704483,
     0.27554838, 0.26154313]
)

path, dist = dtw_path(x_prime_q, x_q_ref)

# The resampling happens here:
list_indices = [[ii for (ii, jj) in path if jj == j]
                for j in range(len(x_q_ref))]
x_prime_q_resample = to_time_series(
    [x_prime_q[indices].mean(axis=0) for indices in list_indices]
)
x_prime_srp_resample = to_time_series(
    [x_prime_srp[indices].mean(axis=0) for indices in list_indices]
)

plot_matches(x_prime_q, x_q_ref, x_prime_q_resample, path)
```

```python tags=["hide_input"]
fig = plt.figure(figsize=(6, 2))

plt.subplot(1, 2, 1)
plt.text(x=0.5, y=1.05,
         s="Original SRP time series", fontsize=12,
         horizontalalignment='center',
         verticalalignment='center', transform=plt.gca().transAxes)

# Plot series (with pos/neg offset for visu reasons)
plt.plot(numpy.arange(x_prime_srp.shape[0]), x_prime_srp.ravel(),
         color='k', linestyle='-', linewidth=2.)

# Plot matches
for (i, j) in path:
    if [pair[0] for pair in path].count(i) > 1:
        plt.plot([i, i], [x_prime_srp[i, 0], x_prime_srp[i, 0]],
                 color="blue", marker='o', linestyle="dashed")
    elif [pair[1] for pair in path].count(j) > 1:
        plt.plot([i, i], [x_prime_srp[i, 0], x_prime_srp[i, 0]],
                 color="red", marker='o', linestyle="dashed")
    else:
        plt.plot([i, i], [x_prime_srp[i, 0], x_prime_srp[i, 0]],
                 color="grey", marker='o', linestyle="dashed")

plt.xticks([])
plt.yticks([])
plt.gca().axis("off")

plt.subplot(1, 2, 2)
plt.text(x=0.5, y=1.05,
         s="Resampled SRP time series", fontsize=12,
         horizontalalignment='center',
         verticalalignment='center', transform=plt.gca().transAxes)

# Plot series (with pos/neg offset for visu reasons)
plt.plot(numpy.arange(x_prime_srp_resample.shape[0]),
         x_prime_srp_resample.ravel(),
         color='k', linestyle='-', linewidth=2.)

# Plot matches
for j in range(len(x_q_ref)):
    if [pair[1] for pair in path].count(j) > 1:
        plt.plot([j, j],
                 [x_prime_srp_resample[j, 0], x_prime_srp_resample[j, 0]],
                 color="red", marker='o', linestyle="dashed")
    else:
        pair = path[[pair[1] for pair in path].index(j)]
        i = pair[0]
        if [pair[0] for pair in path].count(i) > 1:
            plt.plot([j, j],
                     [x_prime_srp_resample[j, 0], x_prime_srp_resample[j, 0]],
                     color="blue", marker='o', linestyle="dashed")
        else:
            plt.plot([j, j],
                     [x_prime_srp_resample[j, 0], x_prime_srp_resample[j, 0]],
                     color="grey", marker='o', linestyle="dashed")
plt.xticks([])
plt.yticks([])
plt.gca().axis("off")   
plt.tight_layout()
```

At this point, each time series is transformed to series of $n$
$p$-dimensional measurements, where $n$ is the length of the
reference discharge time series and $p$ is the number of water quality
parameters considered in the study (_i.e._ all modalities except the discharge).
In a second step, a standard $k$-means algorithm is used to cluster
realigned time series.
Note that a Euclidean distance can be used for clustering since time series
have already been temporally realigned; hence, time-sensitive metrics (such as
DTW) are no longer needed.

This method proved useful to extract meaningful clusters and an _a posteriori_
analysis of the clusters enabled to identify the export dynamics of pollutants
in different geographical areas of the study sites, which then led to management
recommendations, as detailed in {% cite dupas:halshs-01228397 %}.


## References

{% bibliography --cited %}
