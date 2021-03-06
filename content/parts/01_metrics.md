# Defining Adequate Metrics for Structured Data

The definition of adequate metrics between objects to be compared is at the
core of many machine learning methods (_e.g._, nearest neighbors, kernel
machines, _etc._).
When complex objects are involved, such metrics have to be carefully designed
in order to leverage on desired notions of similarity.

This section covers my works related to the definition of new metrics for
structured data such as time series or graphs.
Three tracks are investigated.
First, in [Sec. 1.1](01/temporal_kernel.html), time series are seen as discrete
distributions over the feature-time product space and a kernel that
efficiently compares such representations is defined.
Second, in [Sec. 1.2](01/dtw.html), time series are seen as sequences, which
means that only ordering is of importance (time delay between observations
is ignored) and variants of the Dynamic Time Warping algorithm are used.
Finally, in [Sec. 1.3](01/ot.html), undirected labeled graphs are seen as
discrete distributions over the feature-structure product space and
optimal transport distances are used.
