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

# Optimal Transport for Structured Data

This section covers my works related to Optimal Transport distances for
structured data such as graphs.
In order to compare graphs, we have introduced the Fused Gromov Wasserstein
distance that interpolates between Wasserstein distance between node feature
distributions and Gromov-Wasserstein distance between structures.

<!-- #region {"tags": ["popout"]} -->
**Note.** This work was part of Titouan Vayer's PhD thesis.
We were co-supervising Titouan together with Laetitia Chapel and Nicolas Courty.
<!-- #endregion -->

Here, we first introduce both Wasserstein and Gromov-Wasserstein distances and
some of our results concerning computational considerations related to the
latter.

## Wasserstein and Gromov-Wasserstein distances

Let $\mu = \sum_i h_i \delta_{x_i}$ and $\nu = \sum_j g_j \delta_{y_j}$ be two
discrete distributions lying in the same metric space $(\Omega, d)$.
Then, the $p$-Wasserstein distance is defined as:

<!-- #region {"tags": ["popout"]} -->
**Note** the close connection between this definition and that of the Dynamic
Time Warping in [Sec. 1.2](dtw.html)
<!-- #endregion -->

\begin{equation}
    W_p(\mu, \nu) = \left(
        \min_{\pi \in \Pi} \sum_{i,j} d(x_i, y_j)^p \pi_{i,j}
        \right)^{\frac{1}{p}}
    \label{eq:wass}
\end{equation}

where $\Pi$ is the set of all admissible couplings between $\mu$ and $\nu$
(_ie._ the set of all matrices with maginals $h$ and $g$).

This distance is illustrated in the following Figure:

![](../../images/wass.png)

When distributions $\mu$ and $\nu$ do not lie in the same ambiant space,
however, one cannot compute their Wasserstein distance. An alternative that was
introduced in {% cite memoli2011gromov %} relies on matching intra-domain
distances, as illustrated below:

![](../../images/gw.png)

The corresponding distance is the Gromov-Wasserstein distance, defined as:

\begin{equation}
    GW_p(\mu, \nu) = \left(
        \min_{\pi \in \Pi}
            \sum_{i,j,k,l}
            \left| d_X(x_i, x_k) - d_Y(y_j, y_l) \right|^p \pi_{i,j} \pi_{k,l}
        \right)^{\frac{1}{p}}
    \label{eq:gw}
\end{equation}

where $d_X$ (resp. $d_Y$) is the metric associated to the space in which
$\mu$ (resp. $\nu$) lies.

### Sliced Gromov-Wasserstein

Computational complexity associated to the optimization problem in
Equation \eqref{eq:gw} is high in general.
However, we have shown in {% cite vayer:hal-02174309 %} that in the
mono-dimensional case, this problem can be seen as an instance of the Quadratic
Assignment Problem {% cite koopmans1957assignment %}.
We have provided closed form solution for this instance.
In a nutshell, our solution consists in sorting mono-dimensional distributions
and either matching elements from both distributions in order or in reverse
order, leading to a $O(n \log n)$ algorithm that exactly solves this problem.

Based on this closed-form solution, we were able to introduce a Sliced
Gromov-Wasserstein distance that, similarly to the Sliced Wasserstein distance
{% cite rabin2011wasserstein %}, computes similarity between distributions
through projections on random lines.

**TODO: add a summary of Titouan's last findings about GW when they are
stabilized.**

## Fused Gromov-Wasserstein

Here, we focus on comparing structured data which combine a feature
**and** a structure information.
More formally, we consider undirected labeled graphs as tuples of the form $\mathcal{G}=(\mathcal{V},\mathcal{E},\ell_f,\ell_s)$ where
$(\mathcal{V},\mathcal{E})$ are the set of vertices and edges of the graph.
$\ell_f: \mathcal{V} \rightarrow \Omega_f$ is a labelling function which
associates each vertex $v_{i} \in \mathcal{V}$ with a feature
$a_{i}\stackrel{\text{def}}{=}\ell_f(v_{i})$ in some feature metric space
$(\Omega_f,d)$.
We will denote by _feature information_ the set of all the features
$\{a_{i}\}_{i}$ of the graph.
Similarly, $\ell_s: \mathcal{V} \rightarrow \Omega_s$ maps a vertex $v_i$ from
the graph to its structure representation
$x_{i} \stackrel{\text{def}}{=} \ell_s(v_{i})$ in some structure space
$(\Omega_s,C)$ specific to each graph.
$C : \Omega_s \times \Omega_s \rightarrow \mathbb{R_{+}}$ is a symmetric
application which aims at measuring the similarity between the nodes in the
graph.
Unlike the feature space however, $\Omega_s$ is implicit and in practice,
knowing the similarity measure $C$ will be sufficient. With a slight abuse of
notation, $C$ will be used in the following to denote both the structure
similarity measure and the matrix that encodes this similarity between pairs of
nodes in the graph $\{C(i,k) = C(x_i, x_k)\}_{i,k}$.
Depending on the context, $C$ can either encode the neighborhood information of
the nodes, the edge information of the graph or more generally it can model a
distance between the nodes such as the shortest path distance.
When $C$ is a metric, such as the shortest-path
distance, we naturally endow the structure with the metric space $(\Omega_s,C)$.
We will denote by _structure information_ the set of all the structure
embeddings $\{x_{i}\}_i$ of the graph.
We propose to enrich the previously described graph with a histogram which
serves the purpose of signaling the relative importance of the vertices in the
graph.
To do so, we equip graph vertices with weights $\{h_{i}\}_{i}$ that sum to $1$.

All in all, we define _structured data_ as a
tuple $\mathcal{S}=(\mathcal{G},h_{\mathcal{G}})$ where $\mathcal{G}$ is a
graph as described previously and $h_{\mathcal{G}}$ is a function that
associates a weight to each vertex. This definition allows the graph to be
represented by a fully supported probability measure over the product space
feature/structure $\mu= \sum_{i=1}^{n} h_{i} \delta_{(x_{i},a_{i})}$ which
describes the entire structured data:

![](../../images/graph_as_distrib.svg)

### Distance definition and properties

Let $\mathcal{G}_1$ and $\mathcal{G}_2$ be two graphs, described respectively
by their probability measure $\mu= \sum_{i=1}^{n} h_{i} \delta_{(x_{i},a_{i})}$
and $\nu= \sum_{i=1}^{m} g_{j} \delta_{(y_{j},b_{j})}$.
Their structure matrices are denoted $C_{1}$ and $C_{2}$, respectively.


We define a novel Optimal Transport discrepancy called the
Fused Gromov-Wasserstein distance.
It is defined, for a trade-off parameter  $\alpha \in [0,1]$, as

\begin{equation}
\label{discretefgw}
FGW_{q, \alpha} (\mu, \nu) = \min_\pi E_{q}(\mathcal{G}_1, \mathcal{G}_2, \pi)
\end{equation}

where $\pi$ is a transport map (_i.e._ it has marginals $h$ and $g$) and

\begin{equation}
E_{q}(\mathcal{G}_1, \mathcal{G}_2, \pi) =
    \sum_{i,j,k,l} (1-\alpha) d(a_{i},b_{j})^{q}
                    +\alpha |C_{1}(i,k)-C_{2}(j,l)|^{q} \pi_{i,j}\pi_{k,l} .
\end{equation}

The FGW distance looks for the coupling $\pi$ between vertices of the
graphs that minimizes the cost $E_{q}$ which is a linear combination of a cost
$d(a_{i},b_{j})$ of transporting one feature $a_{i}$ to a feature $b_{j}$ and a
cost $|C_{1}(i,k)-C_{2}(j,l)|$ of transporting pairs of nodes in each structure.
As such, the optimal coupling tends to associate pairs of feature and
structure points with similar distances within each structure pair and with
similar features.
As an important feature of FGW, by relying on a sum of
(inter- and intra-)vertex-to-vertex distances, it can handle structured data
with continuous attributed or discrete labeled nodes
(depending on the definition of $d$) and can also be computed even if the graphs
have different numbers of nodes.

We have shown in {% cite vayer:hal-02174322 %} that FGW retains the following
properties:

* it defines a metric for $q=1$ and a semi-metric for $q >1$;
* varying $\alpha$ between 0 and 1 allows to interpolate between the
Wasserstein distance between the features and the Gromov-Wasserstein distance
between the structures;

We also define a continuous counterpart for FGW which comes with a
concentration inequality in {% cite vayer:hal-02174316 %}.

We have presented a Conditional Gradient algorithm for optimization on the
above-defined loss.
We have also exposed a Block Coordinate Descent algorithm to compute graph
barycenters _w.r.t._ FGW.

### Results

We show that FGW allows to extract meaningful barycenters:

```python tags=["hide_input"]
import networkx as nx
import matplotlib.pyplot as plt
plt.ion()


def build_noisy_circular_graph(n_nodes=20, mu_features=0., sigma_features=0.3,
                               with_noise=False, structure_noise=False, p=None):
    g=Graph()
    g.add_nodes(list(range(N)))
    for i in range(N):
        noise=float(np.random.normal(mu,sigma,1))
        if with_noise:
            g.add_one_attribute(i,math.sin((2*i*math.pi/N))+noise)
        else:
            g.add_one_attribute(i,math.sin(2*i*math.pi/N))
        g.add_edge((i,i+1))
        if structure_noise:
            randomint=np.random.randint(0,p)
            if randomint==0:
                if i<=N-3:
                    g.add_edge((i,i+2))
                if i==N-2:
                    g.add_edge((i,0))
                if i==N-1:
                    g.add_edge((i,1))
    g.add_edge((N,0))
    noise=float(np.random.normal(mu,sigma,1))
    if with_noise:
        g.add_one_attribute(N,math.sin((2*N*math.pi/N))+noise)
    else:
        g.add_one_attribute(N,math.sin(2*N*math.pi/N))
    return g


def draw_graph(g):
    pos = nx.kamada_kawai_layout(g.nx_graph)
    nx.draw(g.nx_graph, pos=pos,
            node_color=graph_colors(g.nx_graph, vmin=-1, vmax=1),
            with_labels=False)

```

```python
import numpy

np.random.seed(0)

n_graphs = 10
n_graphs_shown = 5

dataset = []
for k in range(n_graphs):
    dataset.append(
        build_noisy_circular_graph(np.random.randint(15,25),
                                   with_noise=True,
                                   structure_noise=True,
                                   p=3)
    )

Cs=[x.distance_matrix(force_recompute=True,method='shortest_path') for x in X0]
ps=[np.ones(len(x.nodes()))/len(x.nodes()) for x in X0]
Ys=[x.values() for x in X0]
lambdas=np.array([np.ones(len(Ys))/len(Ys)]).ravel()
sizebary=15 # we choose a barycenter with 15 nodes
init_X=np.repeat(sizebary,sizebary)

features, C_matrix, log = fgw_barycenters(sizebary,
                                          Ys, Cs, ps, lambdas,
                                          alpha=0.95,
                                          init_X=init_X)

# Build graph from barycenter parts (C1 and D1)
barycenter = nx.from_numpy_matrix(
    sp_to_adjency(C_matrix,
                  threshinf=0,
                  threshsup=find_thresh(C1,sup=100,step=100)[0])
)
for i in range(len(features)):
    barycenter.add_node(i, attr_name=float(features[i]))

# Plot stuff
for i in range(len(dataset)):
    plt.subplot(1, n_graphs_shown + 1, i + 1)
    g = dataset[i]
    draw_graph(g)
    plt.title('Sample %d' % (i + 1))
plt.subplot(1, n_graphs_shown + 1, n_graphs_shown + 1)
draw_graph(barycenter)
plt.title('FGW Barycenter')
```

**TODO toy barycenters**
Use that:
<https://pot.readthedocs.io/en/stable/all.html#ot.gromov.fgw_barycenters>

We also show that these barycenters can be used for graph clustering.

We also exhibit classification results for FGW embedded in a Gaussian kernel
SVM which leads to state-of-the-art performance (even outperforming graph
neural network approaches) on a wide range of graph classification problems.

## References

{% bibliography --cited %}
