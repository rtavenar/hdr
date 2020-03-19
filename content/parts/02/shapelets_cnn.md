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

# Shapelet-based Representations and Convolutional Models

In this section, we will cover works that either relate to the Shapelet
representation for time series or to the family of (1d) Convolutional Neural
Networks, since these two families of methods are very similar in spirit
{% cite lods:hal-01565207 %}.

## Data Augmentation for Time Series Classification

<!-- #region {"tags": ["popout"]} -->
**Note.** This work is part of Arthur Le Guennec's Master internship.
We were co-supervising Arthur together with Simon Malinowski.
<!-- #endregion -->

We have shown in {% cite leguennec:halshs-01357973 %} that augmenting time
series classification datasets was an efficient way to improve generalization
for shallow Convolutional Neural Networks.
The data augmentation strategies that were investigated in this work are
local warping and window slicing and both lead to improvements.

**TODO: code snippets for these two simple DA techniques**

## Learning to Mimic a Target Distance

<!-- #region {"tags": ["popout"]} -->
**Note.** This work is part of Arnaud Lods' Master internship.
We were co-supervising Arnaud together with Simon Malinowski.
<!-- #endregion -->

Another track of research we have lead concerns unsupervised representation
learning for time series.
In this context, our approach has consisted in learning a representation in
order to mimic a target distance.

As presented in [Sec. 1.2](../01/dtw.html), Dynamic Time Warping is a widely
used similarity measure for time series.
However, it suffers from its non differentiability and the fact that it does
not satisfy metric properties.
Our goal in {% cite lods:hal-01565207 %} was to introduce a Shapelet model that
extracts latent representations such that Euclidean distance in the latent
space is as close as possible to Dynamic Time Warping between original time
series.
The resulting model is an instance of a Siamese Network:

![](../../images/siamese_ldps.png)

```python  tags=["hide_input"]
%config InlineBackend.figure_format = 'svg'
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import numpy
from torch.autograd import Variable
from sklearn.linear_model import LinearRegression
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import dtw

plt.ion()

def _kmeans_init_shapelets(X, n_shapelets, shp_len, n_draw=10000):
    n_ts, sz, d = X.shape
    indices_ts = numpy.random.choice(n_ts, size=n_draw, replace=True)
    indices_time = numpy.random.choice(sz - shp_len + 1,
                                       size=n_draw,
                                       replace=True)
    subseries = numpy.zeros((n_draw, shp_len, d))
    for i in range(n_draw):
        subseries[i] = X[indices_ts[i],
                         indices_time[i]:indices_time[i] + shp_len]
    return TimeSeriesKMeans(n_clusters=n_shapelets,
                            metric="euclidean",
                            verbose=False).fit(subseries).cluster_centers_


def tslearn2torch(X):
    X_ = torch.Tensor(numpy.transpose(X, (0, 2, 1)))
    return X_


class MinPool1d(nn.Module):
    """ Simple Hack for 1D min pooling. Input size = (N, C, L_in)
        Output size = (N, C, L_out) where N = Batch Size, C = No. Channels
        L_in = size of 1D channel, L_out = output size after pooling.
        This implementation does not support custom strides, padding or dilation
        Input shape compatibilty by kernel_size needs to be ensured

        This code comes from:
        https://github.com/reachtarunhere/pytorch-snippets/blob/master/min_pool1d.py
        (under MIT license)
        """

    def __init__(self, kernel_size=3):
        super(MinPool1d, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, l):
        N, C, L = [l.size(i) for i in range(3)]
        l = l.view(N, C, int(L / self.kernel_size), self.kernel_size)
        return l.min(dim=3)[0].view(N, C, -1)


class ShapeletLayer(nn.Module):
    """Shapelet layer.

    Computes sliding window distances between a set of time series and a set
    of shapelets.

    Parameters
    ----------
    in_channels : int
        Number of input channels (modalities of the time series)
    out_channels: int
        Number of output channels (number of shapelets)
    kernel_size: int
        Shapelet length

    Examples
    --------
    >>> time_series = torch.Tensor([[1., 2., 3., 4., 5.], [-4., -5., -6., -7., -8.]]).view(2, 1, 5)
    >>> shapelets = torch.Tensor([[1., 2.], [3., 4.], [5., 6.]])
    >>> layer = ShapeletLayer(in_channels=1, out_channels=3, kernel_size=2)
    >>> layer.weight.data = shapelets
    >>> dists = layer.forward(time_series)
    >>> dists.shape
    torch.Size([2, 3, 4])
    >>> dists[0]
    tensor([[ 0.,  1.,  4.,  9.],
            [ 4.,  1.,  0.,  1.],
            [16.,  9.,  4.,  1.]], grad_fn=<SelectBackward>)
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ShapeletLayer, self).__init__()

        self.kernel_size = kernel_size
        self.out_channels = out_channels

        self.false_conv_layer = nn.Conv1d(in_channels=in_channels,
                                          out_channels=in_channels,
                                          kernel_size=kernel_size,
                                          bias=False)
        data = torch.Tensor(numpy.eye(kernel_size))
        self.false_conv_layer.weight.data = data.view(kernel_size,
                                                      1,
                                                      kernel_size)
        for p in self.false_conv_layer.parameters():
            p.requires_grad = False
        self.weight = nn.Parameter(torch.Tensor(out_channels,
                                                kernel_size))

    def forward(self, x):
        reshaped_x = self.false_conv_layer(x)
        reshaped_x = torch.transpose(reshaped_x, 1, 2)
        reshaped_x = reshaped_x.contiguous().view(-1, self.kernel_size)
        distances = self.pairwise_distances(reshaped_x, self.weight)
        distances = distances.view(x.size(0), -1, self.out_channels)
        return torch.transpose(distances, 1, 2)

    @classmethod
    def pairwise_distances(cls, x, y):
        """Computes pairwise distances between vectors in x and those in y.

        Computed distances are normalized (i.e. divided) by the dimension of
        the space in which vectors lie.
        Assumes x is 2d (n, d) and y is 2d (l, d) and returns
        a tensor of shape (n, l).

        Parameters
        ----------
        x : Tensor of shape=(n, d)
        y : Tensor of shape=(l, d)

        Returns
        -------
            A 2d Tensor of shape (n, l)
        """
        L = y.size(-1)
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        return torch.clamp(dist, 0.0, numpy.inf) / L
```

Here is a `PyTorch` implementation of this model:

```python
class LDPSModel(nn.Module):
    """Learning DTW-Preserving Shapelets (LDPS) model.

    Parameters
    ----------
    n_shapelets_per_size: dict (optional, default: None)
        Dictionary giving, for each shapelet size (key),
        the number of such shapelets to be trained (value)
        None should be used only if `load_from_disk` is set
    ts_dim: int (optional, default: None)
        Dimensionality (number of modalities) of the time series considered
        None should be used only if `load_from_disk` is set
    lr: float (optional, default: 0.01)
        Learning rate
    epochs: int (optional, default: 500)
        Number of training epochs
    batch_size: int (optional, default: 64)
        Batch size for training procedure
    verbose: boolean (optional, default: True)
        Should verbose mode be activated

    Note
    ----
        This implementation requires a dataset of equal-sized time series.
    """
    def __init__(self,
                 n_shapelets_per_size=None,  # dict sz_shp -> n_shp
                 ts_dim=1,
                 lr=.01,
                 epochs=500,
                 batch_size=64,
                 verbose=True):
        super(LDPSModel, self).__init__()

        self.n_shapelets_per_size = n_shapelets_per_size
        self.ts_dim = ts_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

        self._set_layers_and_optim()

    def _set_layers_and_optim(self):
        self.shapelet_sizes = sorted(self.n_shapelets_per_size.keys())
        self.shapelet_blocks = self._get_shapelet_blocks()
        self.scaling_layer = nn.Linear(1, 1, bias=False)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def _get_shapelet_blocks(self):
        return nn.ModuleList([
            ShapeletLayer(
                in_channels=self.ts_dim,
                out_channels=self.n_shapelets_per_size[shapelet_size],
                kernel_size=shapelet_size
            ) for shapelet_size in self.shapelet_sizes
        ])

    def _temporal_pooling(self, x):
        pool_size = x.size(-1)
        pooled_x = MinPool1d(kernel_size=pool_size)(x)
        return pooled_x.view(pooled_x.size(0), -1)

    def _features(self, x):
        features_maxpooled = []
        for shp_sz, block in zip(self.shapelet_sizes, self.shapelet_blocks):
            f = block(x)
            f_maxpooled = self._temporal_pooling(f)
            features_maxpooled.append(f_maxpooled)
        return torch.cat(features_maxpooled, dim=-1)

    def _init_params(self):
        for m in self.shapelet_blocks:
            self._shapelet_initializer(m.weight)

        # Initialize scaling using linear regression
        pair, targets = self.get_batch(self.X_fit_)
        nn.init.constant_(self.scaling_layer.weight, 1.)  # Start without scale
        output = self(pair)
        reg_model = LinearRegression(fit_intercept=False)
        reg_model.fit(output.detach().numpy(), targets.detach().numpy())
        nn.init.constant_(self.scaling_layer.weight, reg_model.coef_[0, 0])

    def _shapelet_initializer(self, w):
        X_npy = numpy.transpose(self.X_fit_, axes=(0, 2, 1))
        shapelets_npy = _kmeans_init_shapelets(X=X_npy,
                                               n_shapelets=w.size(0),
                                               shp_len=w.size(1))
        w.data = torch.Tensor(shapelets_npy.reshape((w.size(0), w.size(1))))

    def forward(self, x):
        xi, xj = x
        emb_xi = self._features(xi)
        emb_xj = self._features(xj)
        norm_ij = torch.norm(emb_xi - emb_xj, p=2, dim=1, keepdim=True)
        scaled_norm_ij = self.scaling_layer(norm_ij)
        return scaled_norm_ij

    def get_batch(self, X):
        n_samples = X.size(0)
        batch_indices1 = numpy.random.choice(n_samples, size=self.batch_size)
        batch_indices2 = numpy.random.choice(n_samples, size=self.batch_size)
        X1 = Variable(X[batch_indices1].type(torch.float32),
                      requires_grad=False)
        X2 = Variable(X[batch_indices2].type(torch.float32),
                      requires_grad=False)
        targets_tensor = torch.Tensor([dtw(X1[i].T, X2[i].T)
                                       # NOTE: tslearn dim ordering is reverse
                                       for i in range(self.batch_size)])
        targets = Variable(targets_tensor.view(-1, 1), requires_grad=False)
        return (X1, X2), targets

    def fit(self, X):
        """Learn shapelets and weights for a given dataset.

        Parameters
        ----------
        X : numpy array of shape=(n_ts, sz, 1)
            Time series dataset

        Returns
        -------
        LDPSModel
            The fitted model
        """
        X_ = tslearn2torch(X)
        self.X_fit_ = X_

        self._init_params()
        loss_fun = nn.MSELoss()

        n_batch_per_epoch = max(X_.size(0) // self.batch_size, 1)

        for epoch in range(self.epochs):
            running_loss = 0.0
            for _ in range(n_batch_per_epoch):
                inputs, targets = self.get_batch(X_)

                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                preds = self(inputs)
                loss = loss_fun(preds, targets)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if self.verbose and (epoch + 1) % 50 == 0:
                    print('[%d] loss: %.3f' %
                          (epoch + 1, running_loss / n_batch_per_epoch)
                         )
        return self
```

and a simple example usage:

```python
from tslearn.datasets import UCR_UEA_datasets
X, _, X_test, _ = UCR_UEA_datasets().load_dataset("SwedishLeaf")

numpy.random.seed(0)
torch.manual_seed(0)
model = LDPSModel(n_shapelets_per_size={30: 5},
                  lr=.001,
                  epochs=1000,
                  batch_size=16)
model.fit(X[:10])

pairs, targets = model.get_batch(tslearn2torch(X_test[:10]))
outputs = model(pairs)
plt.scatter(outputs.detach().numpy(),
            targets.detach().numpy(),
            color="b", marker="x")
max_val = max(outputs.max(), targets.max())
plt.plot([0., max_val], [0., max_val], "r-")
plt.xlabel("Distance in the Shapelet Transform space")
plt.ylabel("Dynamic Time Warping distance");
```

We have shown that such a model could be used as a feature extractor on top of
which a $k$-means clustering could operate efficiently.
We have also shown in {% cite carlinisperandio:hal-01841995 %} that this
representation is useful for time series indexing tasks.

## Including Localization Information



## References

{% bibliography --cited %}
