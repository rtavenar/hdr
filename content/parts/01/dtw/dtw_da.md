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

# DTW alignment as a temporal domain adaptation strategy

<!-- #region {"tags": ["popout"]} -->
**Note.** This work was a part of Rémi Dupas' PhD thesis (in Environment
Sciences).
I was not involved in Rémi's PhD supervision.
<!-- #endregion -->

In this section, we present a method that uses Dynamic Time Warping (DTW)
on multimodal data.
The method relies on the assumption that one of the modalities at stake (called
reference modality in the following) can be used as a reference to (temporally)
realign other modalities {% cite dupas:halshs-01228397 %}.
It has been used in the context of hydrological measurements to align pollutant
concentration profiles based on discharge time series.

This approach can be seen as the DTW counterpart of other works that rely on
Optimal Transport for Domain Adaptation {% cite courty:hal-02112785 %}.
One significant difference however is that we rely on a reference modality for
alignment, which is guided by our application context.

## Use case

Phosphorus (P) transfer during storm events represents a significant part of
annual P loads in streams and contributes to eutrophication in downstream water
bodies. To improve understanding of P storm dynamics, automated or
semi-automated methods are needed to extract meaningful information from
ever-growing water quality measurement datasets.

Clustering techniques have proven useful for identifying seasonal storm
patterns and thus for increasing knowledge about seasonal variability in storm
export mechanisms (e.g. {% cite aubert:halshs-00906292 %}).
Clustering techniques usually require calculating a distance between pairs of
comparable points in several time series. For this reason, direct clustering
(without using hysteresis-descriptor variables) of high frequency storm
concentration time series is usually irrelevant because their length (number of
measurement points) may differ and/or measurement points may have different
positions relative to the hydrograph (flow rise and recession); hence, it is
difficult to calculate a distance between pairs of comparable points.

The aim of this study was to develop a clustering method that overcomes this
limit and test its ability to compare seasonal variability of P storm dynamics
in two headwater watersheds. Both watersheds are ca. 5 km², have similar
climate and geology, but differ in land use and P pressure intensity.

## Method

In the above-described setting, we have access to one modality (discharge,
commonly denoted $Q$) that is representative of the evolution of the flood.
Temporal realignment based on this modality allows to overcome three
difficulties that may arise when comparing storm-event data: time series may
have

1. different starting times due to the discharge threshold at which the
autosamplers were triggered,
2. different lengths  and
3. differences in phase that yield different positions of the discharge peak
and of concentration data points relative to the hydrograph.

To align time series, we used path associated to DTW.
This matching path can be viewed as the optimal way to perform point-wise
alignment of time series.

We used the matching path $\pi^\text{Q}$ to align each discharge time series to
the same reference discharge time series $\mathbf{x}_\text{ref}^\text{Q}$.
The reference discharge time series used in this study was chosen
as a storm event with full coverage of flow rise and flow recession phases.
Alternatively, one could choose a synthetic idealized storm hydrograph.

As stated above, the continuity condition imposed on admissible paths results
in each element of reference time series $\mathbf{x}_\text{ref}^\text{Q}$ being
matched with at least one element in each discharge time series from the
dataset.
We then used barycentric mapping based on obtained matches to realign other
modalities to the timestamps of the reference time series, as shown in the
following Figure:

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
    plt.text(x=0.5, y=1.0,
             s="Original discharge time series", fontsize=12, horizontalalignment='center',
             verticalalignment='center', transform=plt.gca().transAxes)
    plt.text(x=0.5, y=0.0,
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
    plt.text(x=0.5, y=1.0,
             s="Resampled discharge time series", fontsize=12, horizontalalignment='center',
             verticalalignment='center', transform=plt.gca().transAxes)
    plt.text(x=0.5, y=0.0,
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
plt.text(x=0.5, y=1.0,
         s="Original SRP time series", fontsize=12,
         horizontalalignment='center',
         verticalalignment='center', transform=plt.gca().transAxes)

# Plot series (with pos/neg offset for visu reasons)
plt.plot(numpy.arange(x_prime_srp.shape[0]), y_srp.ravel(),
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
plt.text(x=0.5, y=1.0,
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
parameters considered in the study (_i.e._ all modalities except discharge).
In a second step, a standard $k$-means algorithm was used to cluster
realigned time series.
Note that a Euclidean distance can be used for clustering since time series
had already been temporally realigned; hence, no time-sensitive metric (such as
DTW) was needed anymore.

This method proved useful to extract meaningful clusters and an _a posteriori_
analysis of the clusters enabled to identify the export dynamics of pollutants
in different geographical areas of the study sites, which then led to management
recommendations, as detailed in {% cite dupas:halshs-01228397 %}.


## References

{% bibliography --cited %}
