# Learning sensible representations for time series

Another track of research I have been following over the past years is the
learning of latent representations for time series.
These latent representations can either be mixture coefficients
(_cf._ [Sec 2.1](02/topic_models.html)) -- in which case time series are
represented as multinomial distributions over latent topics -- or intermediate
neural networks feature maps (as in [Sec 2.2](02/shapelets_cnn.html) and
[Sec 2.3](02/early.html)) -- and then time series are represented through
filter activations they trigger.

More specifically, in [Sec 2.3](02/early.html), we focus on the task of early
classification of time series. In this context, a method is introduced that
learns an intermediate representation from which both the decision of
triggering classification and the classification itself can be computed.
