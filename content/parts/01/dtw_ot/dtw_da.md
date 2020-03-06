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
the same reference discharge time series $x_\text{ref}^\text{Q}$.
The reference discharge time series used in this study was chosen
as a storm event with full coverage of flow rise and flow recession phases.
Alternatively, one could choose a synthetic idealized storm hydrograph.

As stated above, the continuity condition imposed on admissible paths results
in each element of reference time series $x_\text{ref}^\text{Q}$ being matched
with at least one element in each discharge time series from the dataset.
We then used barycentric mapping based on obtained matches to realign other
modalities to the timestamps of the reference time series, as shown in the
following Figure:
 **TODO Figure 5 in the notebook**

At this point, each time series was transformed to series of $T_\text{ref}$
$d$-dimensional measurements, where $T_\text{ref}$ is the length of the
reference discharge time series and $d$ is the number of water quality
parameters considered in the study (_i.e._ all modalities except discharge).
In a second step, a standard $k$-means algorithm was used to cluster
realigned time series.
Note that a Euclidean distance can be used for clustering since time series
had already been temporally realigned; hence, no time-sensitive metric (such as
DTW) was needed anymore.

This method proved useful to extract meaningful clusters and an _a posteriori_
analysis of the clusters enabled to identify the export dynamics of pollutants
in different geographical areas of the study sites, which then led to management
recommendations, as described in {% cite dupas:halshs-01228397 %}.

## References

{% bibliography --cited %}
