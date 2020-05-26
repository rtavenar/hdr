# Perspectives

In this part, I will first describe some current and future works that we plan
to investigate.
Finally, I will discuss some more fundamental and general questions that
I expect to be of importance to the future of machine learning for time series.

## Current and Future Works

### Dealing with Sequences of Arbitrary Objects

As described in [Sec 1.2.3](01/dtw/dtw_gi.html), we have started to investigate
the design of invariant alignment similarity measures.
This work can be seen as a first attempt to accommodate time series alignments
(such as Dynamic Time Warping) and optimal transport distances
(and more specifically the work presented in {% cite alvarez2018towards %}).

One step forward in this direction is to take direct inspiration from
the Gromov-Wasserstein distance presented in [Sec. 1.3](01/ot.html) for designing
novel time series alignment strategies.
While DTW-GI can deal with series of features that do not have the
same dimension, this formulation would allow the comparison of
sequences of arbitrary objects that lie in different metric spaces (not
necessarily of the form $\mathbb{R}^p$), like, for example, graphs evolving
over time.

Though this extension seems appealing, it would come with additional
computing costs since the Bellmann recursion, which is at the core of the
Dynamic
Time Warping algorithm, cannot be used anymore.
It is likely that approximate solvers will have to be used in this case.
Also, one typical use-case for such a similarity measure would be to serve as
a loss function in a forecasting setting, in which case the
computational complexity would be an even higher concern which could necessitate
to train dedicated Siamese networks (_e.g._ by taking inspiration from the
method presented in
[Sec. 2.2](02/shapelets_cnn.html#Learning-to-Mimic-a-Target-Distance)).

### Temporal Domain Adaptation

Another track of research that I am considering at the moment concerns
temporal domain adaptation, that is the task of temporally realigning time
series datasets in order to be able to transfer knowledge (_e.g._ a trained
classifier) from one domain to the other.

In this context, and in close relation with application needs, several settings
can be considered:

1. Time series can be matched with no particular constraint on temporal
alignments (_i.e._ individual alignments are considered independent);
2. Time series are matched with the strong constraint that a single temporal
alignment map is used for all time series comparison;
3. There exists a finite number of different temporal alignment patterns and
one should extract these patterns, the matching between series of source
to target datasets and the pattern used for each match.

In the **first
case**, matching can be performed using optimal transport and DTW as the ground
metric, and the method from {% cite courty:hal-02112785 %} can be used.
One straightforward approach for the **second case** can be to alternate
between (i) an optimal transport
problem (finding time series pairs) for a fixed temporal realignment and (ii) a
Dynamic Time Warping between synthetic series (that are built from the source
and target datasets respectively) given a fixed series matching.
The **latter case** is probably the most ambitious one, yet it is of prime
importance in real-world settings such as the classification of satellite image
time series.
Indeed, in this context, images can contain pixels representing different land
cover classes, which have different temporal responses to a given input
(_e.g._ change in meteorological conditions).
Hence each cluster of temporal response could be assigned a different temporal
alignment pattern.

## Broader Questions Related to Learning from Time Series

### Learning the Notion of Similarity

As illustrated in this document, learning from time series can take very diverse
forms depending on the invariants involved in the data.
In case these invariants are known, dedicated methods can be used, yet it
can be that very limited expert knowledge is available or that knowledge cannot
easily guide the choice of a learning method.
At the moment, this is handled through the use of ensemble techniques that
cover a wide range of similarity notions {% cite lines2018time %},
yet this is at the cost of a significantly augmented complexity.
More principled approaches are yet to be designed that could learn the notion
of similarity from the data.

### Structure as a Guide for Weakly-supervised Learning

Finally, learning meaningful representations in weakly supervised settings is
probably one of the major challenges for the community in the coming years.
Unsupervised representation learning has been overlooked in the
literature up to now, despite recent advances such as
{% cite franceschi2019unsupervised %}, which relies on contrastive learning.

In this context, I believe structure can
be used as a guide.
Typically, in the time series context, learning intermediate representations
that are suited for structured prediction (_i.e._, predicting future
observations together with their emission times) is likely to capture the
intrinsics of the data.
Such approaches could rely on the recent revival of time series forecasting
models, such as in
{% cite vincent2019shape %} and {% cite rubanova2019latent %}.
A first step in this direction is the SOM-VAE model presented in
{% cite fortuin2019som %}, which relies on a Markov assumption to model
transitions between quantized latent representations.

Note that the great potential of structured prediction to learn useful
representations from unsupervised datasets is not restricted to the time series
context, it also holds for graphs and other kinds of structured data.
Such a representation could then be used for various tasks with limited amount
of supervision, in a few-shot learning fashion.

We have started investigating an instance of this paradigm in François
Painblanc's PhD thesis that deals with the use of forecasting models for a
better estimation of possible futures in the context of early classification.



## References

{% bibliography --cited %}
