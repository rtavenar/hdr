---
jupyter:
  jupytext:
    formats: ipynb,md,Rmd,py:percent
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

# Temporal Kernel for Time Series Classification

The method presented in this section consists in defining a kernel between
sets of timestamped objects (typically features).
Hence, time series considered in this section are supposed to be of the
following form:

\begin{equation}
    \textbf{x} = \{ (x_0, t_0), \dots , (x_{n-1}, t_{n-1}) \}
\end{equation}

(each observation in the sequence is timestamped).
This allows, in particular, to consider the case of irregular sampling.

<!-- #region {"tags": ["popout"]} -->
**Note.** This work was part of Adeline Bailly's PhD thesis.
<!-- #endregion -->

## Match kernel and Signature Quadratic Form Distance

Our method relies on a user-chosen kernel $k(\cdot,\cdot)$ between local
features.
Based on this local kernel, one can compute the match kernel
{% cite NIPS2009_3874 %} between sets of local features as:

\begin{equation}
    K(\textbf{x}, \textbf{y}) = \sum_i \sum_j k(x_i, y_j).
\end{equation}

And the Signature Quadratic Form Distance (SQFD,
{% cite 10.1145/1631272.1631391 %}) is the distance
between feature sets embedded in the Reproducing Kernel Hilbert Space (RKHS)
associated with $K$:

\begin{equation}
    SQFD(\textbf{x}, \textbf{y})^2 = K(\textbf{x}, \textbf{x}) +
        K(\textbf{y}, \textbf{y}) - 2 K(\textbf{x}, \textbf{y}).
\end{equation}

## Local temporal kernel

We introduce a time-sensitive local kernel defined as:

\begin{equation}
    k_t((x_i, t_i), (y_j, t_j)) = e^{\gamma_t (t_j, - t_i)^2} k(x_i, y_j).
\end{equation}

This kernel is positive semi definite (psd), as the product of two psd kernels
and, if $k$ is the RBF kernel, it can be written as:

\begin{equation}
    k_t((x_i, t_i), (y_j, t_j)) = k(g(x_i, t_i), g(y_j, t_j)).
\end{equation}
with
\begin{equation}
g(x_i, t_i) = \left( x_{i,0}, \dots , x_{i, d-1},
                            \sqrt{\frac{\gamma_t}{\gamma_f}} t_i \right)
\end{equation}

The code below illustrates the impact of the ratio
$\sqrt{\frac{\gamma_t}{\gamma_f}}$ on the kernel matrix (larger $\gamma_t$
leads to ignoring off-diagonal elements):

```python tags=["hide_input"]
import numpy
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

plt.ion()

# Define time series
s_x = numpy.array(
    [-0.790, -0.765, -0.734, -0.700, -0.668, -0.639, -0.612, -0.587, -0.564,
     -0.544, -0.529, -0.518, -0.509, -0.502, -0.494, -0.488, -0.482, -0.475,
     -0.472, -0.470, -0.465, -0.464, -0.461, -0.458, -0.459, -0.460, -0.459,
     -0.458, -0.448, -0.431, -0.408, -0.375, -0.333, -0.277, -0.196, -0.090,
     0.047, 0.220, 0.426, 0.671, 0.962, 1.300, 1.683, 2.096, 2.510, 2.895,
     3.219, 3.463, 3.621, 3.700, 3.713, 3.677, 3.606, 3.510, 3.400, 3.280,
     3.158, 3.038, 2.919, 2.801, 2.676, 2.538, 2.382, 2.206, 2.016, 1.821,
     1.627, 1.439, 1.260, 1.085, 0.917, 0.758, 0.608, 0.476, 0.361, 0.259,
     0.173, 0.096, 0.027, -0.032, -0.087, -0.137, -0.179, -0.221, -0.260,
     -0.293, -0.328, -0.359, -0.385, -0.413, -0.437, -0.458, -0.480, -0.498,
     -0.512, -0.526, -0.536, -0.544, -0.552, -0.556, -0.561, -0.565, -0.568,
     -0.570, -0.570, -0.566, -0.560, -0.549, -0.532, -0.510, -0.480, -0.443,
     -0.402, -0.357, -0.308, -0.256, -0.200, -0.139, -0.073, -0.003, 0.066,
     0.131, 0.186, 0.229, 0.259, 0.276, 0.280, 0.272, 0.256, 0.234, 0.209,
     0.186, 0.162, 0.139, 0.112, 0.081, 0.046, 0.008, -0.032, -0.071, -0.110,
     -0.147, -0.180, -0.210, -0.235, -0.256, -0.275, -0.292, -0.307, -0.320,
     -0.332, -0.344, -0.355, -0.363, -0.367, -0.364, -0.351, -0.330, -0.299,
     -0.260, -0.217, -0.172, -0.128, -0.091, -0.060, -0.036, -0.022, -0.016,
     -0.020, -0.037, -0.065, -0.104, -0.151, -0.201, -0.253, -0.302, -0.347,
     -0.388, -0.426, -0.460, -0.491, -0.517, -0.539, -0.558, -0.575, -0.588,
     -0.600, -0.606, -0.607, -0.604, -0.598, -0.589, -0.577, -0.558, -0.531,
     -0.496, -0.454, -0.410, -0.364, -0.318, -0.276, -0.237, -0.203, -0.176,
     -0.157, -0.145, -0.142, -0.145, -0.154, -0.168, -0.185, -0.206, -0.230,
     -0.256, -0.286, -0.318, -0.351, -0.383, -0.414, -0.442, -0.467, -0.489,
     -0.508, -0.523, -0.535, -0.544, -0.552, -0.557, -0.560, -0.560, -0.557,
     -0.551, -0.542, -0.531, -0.519, -0.507, -0.494, -0.484, -0.476, -0.469,
     -0.463, -0.456, -0.449, -0.442, -0.435, -0.431, -0.429, -0.430, -0.435,
     -0.442, -0.452, -0.465, -0.479, -0.493, -0.506, -0.517, -0.526, -0.535,
     -0.548, -0.567, -0.592, -0.622, -0.655, -0.690, -0.728, -0.764, -0.795,
     -0.815, -0.823, -0.821])

s_y1 = numpy.concatenate((s_x, s_x)).reshape((-1, 1))
s_y2 = numpy.concatenate((s_x, s_x[::-1])).reshape((-1, 1))

# Figure details
left, bottom = 0.01, 0.1
w_ts = h_ts = 0.2
left_h = left + w_ts + 0.02
width = height = 0.65
bottom_h = bottom + height + 0.02

rect_s_y = [left, bottom, w_ts, height]
rect_gram = [left_h, bottom, width, height]
rect_s_x = [left_h, bottom_h, width, h_ts]
```

```python
def g(x, ratio):
    sz = x.shape[0]
    return numpy.hstack(
        (x, numpy.linspace(0., 1., sz).reshape((-1, 1)))
    )

gamma_f = 10.
gamma_t = 1000 * 1000.
ratio = numpy.sqrt(gamma_t / gamma_f)

# Build augmented representations ($g$ function)
s_y1_t = g(s_y1, ratio)
s_y2_t = g(s_y2, ratio)

# Plotting stuff
plt.figure(figsize=(8, 8))
ax_gram = plt.axes(rect_gram)
ax_s_x = plt.axes(rect_s_x)
ax_s_y = plt.axes(rect_s_y)
ax_gram.axis("off")
ax_s_x.axis("off")
ax_s_x.set_xlim((0, s_y2.shape[0] - 1))
ax_s_y.axis("off")
ax_s_y.set_ylim((0, s_y1.shape[0] - 1))

# Show kernel matrix and series on the same plot
gram = numpy.exp(-gamma_f * cdist(s_y1_t, s_y2_t, "sqeuclidean"))
ax_gram.imshow(gram)

ax_s_x.plot(numpy.arange(s_y2.shape[0]), s_y2,
            "b-", linewidth=3.)

ax_s_y.plot(- s_y1, numpy.arange(s_y1.shape[0])[::-1],
            "b-", linewidth=3.);
```

$k_t$ is then a RBF kernel itself, and kernel approximation techniques can be
used in order to approximate it with a linear kernel {% cite NIPS2007_3182 %}.

Let us assume that we have a feature map $\phi$ such that

\begin{equation}
k(x_{i}, y_{j}) \approx \left\langle\phi(x_{i}), \phi(y_{j})\right\rangle,
\end{equation}
then we have:

\begin{equation}
SQFD(\mathbf{x}, \mathbf{y}) \approx \left\|
    \underbrace{\frac{1}{n}\sum_i \phi(x_{i})}_{b_\phi(\mathbf{x})} -
    \underbrace{\frac{1}{m}\sum_j \phi(y_{j})}_{b_\phi(\mathbf{y})}
    \right\|.
\end{equation}

In other words, once feature sets are projected in this finite-dimensional
space, approximate SQFD computation is performed through (i) a barycenter
computation $b_\phi(\cdot)$ in the feature space (which can be done offline)
followed by (ii) a Euclidean distance computation in $O(D)$ time, where $D$ is
the dimension of the feature map $\phi(x)$.
Note that SQFD then corresponds to a biased estimator of the squared
difference between the mean of the samples $\mathbf{x}$ and $\mathbf{y}$ which
is classically used to test the difference between distributions
{% cite NIPS2006_3110 %}.

Overall, we have a distance between timestamped feature sets whose
complexity can be tuned via the map dimensionality $D$.

## Evaluation

### Temporal SIFT features

<!-- #region {"tags": ["popout"]} -->
**Note** that the use of such handcrafted features was already outdated in the
computer vision community at the time of this work.
However, in our small data context, they proved useful for the task at hand.
<!-- #endregion -->

In order to evaluate the classifier presented above, we used the UCR Time
Series Classification archive, which, at the time, was made of monodimensional
data only.
We decided not to work on raw data but rather extract local features on which
we could run the classification algorithm.
We chose to rely on temporal SIFT features, that we had introduced in
{% cite bailly:halshs-01184900 bailly:hal-01252726 %}.
These features are straight-forward 1D extensions of the Scale-Invariant
Feature Transform (SIFT) framework introduced in Computer Vision
{% cite Lowe:2004:DIF:993451.996342 %}.

### Results

Our experiments show that kernel approximation leads to lower computational
complexity than a pre-processing of the feature sets that would consist in
performing a $k$-means clustering of each set and using centroids as
representative features.
We also show that the obtained distance, once embedded in a Support Vector
Machine with Gaussian kernel, leads to classification performance that is
competitive with the state-of-the-art.

## References

{% bibliography --cited %}
