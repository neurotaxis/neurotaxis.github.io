# Gauss-Hermite quadrature: a principled approximation of Gaussian integrals
**Rich Pang**

2026-01-21

**TL;DR: A principled, deterministic approximation of a Gaussian integral when only $n$ function calls are allowed.**

Consider a function of the form

$$
f = \int_{-\infty}^{\infty}dx \mathcal{N}(x) g(x)
$$

where $\mathcal{N}(x)$ is the standard Gaussian distribution and $g$ is such that we cannot compute the integral analytically. Computing $f$ numerically requires evaluating $g$ at a set of $x$'s and then combining the results to approximate the integral. Where are the "best" $x$ at which to evaluate $g(x)$ and how do we combine these to estimate $f$?

## Trapezoid and Monte-Carlo approaches

One approach is to truncate the integration bounds at $\pm L$, evaluate $x$ at a grid of $n$ evenly spaced points and use the trapezoid rule. This requires $n$ function evaluations, and when $L$ and $n$ are large we will generally observe good results as long as $g$ is sufficiently well-behaved. Yet when it comes time to write the code it is not clear how to pick $L$ nor whether $n$ evenly spaced grid points are where we should evaluate $g$.

Alternatively, since $f$ is itself an expectation value, $f = \text{E}_{\mathcal{N}(x)}[g(x)]$, we could take the Monte Carlo approach of sampling $x_1 \dots x_n$wfrom $N(x)$ and approximate $f$ using the law of large numbers:

$$
f \approx \sum_{i=1}^n g(x_i).
$$
Now we don't have to worry about $L$. However, the MC approach introduces variance in the approximation due to variation in how the $x_i$ are sampled. One sample of $x_i$ might yield a very good approximation and another a rather poor one.

Suppose $g$ is a rather expensive to evaluate. If we allow ourselves only $n$ evaluations of $g$, what $x_1 \dots x_n$ should we pick and how should we combine these to approximate $f$?

## Gauss-Hermite quadrature

Gauss-Hermite quadrature provides a principled, deterministic solution for exactly how we should distribute our $n$ function evaluations. Specifically, G-H quadrature approximates the integral as
$$
f \approx \sum_{i=1}^n w_i g(x_i)
$$
where $x_i$ are the roots of the [physicist's Hermite polynomial](https://en.wikipedia.org/wiki/Hermite_polynomials) $H_n(x)$ and the weights are given by

$$
w_j = \frac{2^{n-1}n!\sqrt{pi}}{n^2 H_{n-1}(x_i)^2}.
$$

Unlike the trapezoid rule, we no longer have to choose a truncation, and unlike the MC approach, $x_i$ are purely deterministic.

## Example

![Plot of Gauss-Hermite quadrature approximation of Gaussian expectation of sin(x)*x**3](gauss_hermite.png)

## Python Code

The Python code is quite simple, since we can use numpy's `hermgauss` function:

```
import numpy as np
from numpy.polynomial.hermite import hermgauss

n = 16
x_i, w_i = hermgauss(n)

def g(x):
    return sin(x)*x**3

# evaluate (note the additional scaling factors needed)
f = (1. / np.sqrt(np.pi)) * np.sum(w_i * g(np.sqrt(2.0) * x_i))
```

### Generalization to non-standard Gaussian

To approximate

$$
f = \int_{-\infty}^{\infty}dx \frac{1}{\sqrt{2\pi \sigma^2}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right) g(x)
$$
we first change variables via $y = (x - \mu)/\sigma$, $dy = dx/\sigma$:
$$
f = \int_{-\infty}^{\infty}dy \frac{1}{\sqrt{2\pi}}\exp\left(-\frac{y^2}{2}\right) g(\sigma y + \mu) = \int_{-\infty}^{\infty} dy \mathcal{N} h(y)
$$
where $h(y) = g(\sigma y + \mu)$.

The code is

```
import numpy as np
from numpy.polynomial.hermite import hermgauss

mu = 1
sd = 2

n = 16
y_i, w_i = hermgauss(n)

def g(x):
    return sin(x)*x**3

def h(y):
    return g(sd*y + \mu)

f = (1. / np.sqrt(np.pi)) * np.sum(w_i * h(np.sqrt(2.0) * y_i))
```

## Remark

In general, the G-H approximation is not necessarily optimal, and its accuracy will depend on $g$. For instance, if $g$ is a sum of $n$ delta functions, the optimal function evaluations should take place at the locations of the delta functions. In practice, however, it can be quite accurate for well-behaved $g$ and notably its accuracy is determined only by a single parameter $n$, the number of function evaluations.

## Further reading

* [Wikipedia page](https://en.wikipedia.org/wiki/Gauss–Hermite_quadrature)