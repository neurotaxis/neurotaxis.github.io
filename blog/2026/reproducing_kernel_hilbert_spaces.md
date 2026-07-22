# What are reproducing kernel Hilbert spaces all about?
**Rich Pang**

2026-07-22

Reproducing kernel Hilbert spaces (RKHS) are the indispensable spaces underlying the formal foundations of kernel methods.

What are they and what is interesting about them?

## What is a RKHS?

A RKHS is the space of functions along with a specific inner product, where that space of functions corresponds to the functions that can be learned with a kernel machine.
A kernel machine is a model that generates predictions about a test input $x$ by comparing it to a set of training inputs $\{x_c\}$ where comparison is formalized via a kernel function,
$$k(x, x_c),$$
typically symmetric and positive definite in RKHS theory, which returns a value quantifying how similar $x$ and $x_c$ are.
A common kernel is the squared-exponential kernel, $k(x, x_c) = \exp(-(x-x_c)^2/(2\lambda))$.
If we hold $x_c$ fixed and let
$$
K_{x_c}(x) \equiv k(x, x_c)
$$
we call $K_{x_c}(x)$ the kernel function with center $x_c$.

Kernel machines produce predictions of the target value associated with a test input $x$ via a weighted sum of $k(x, x_c)$ for all the training data points $x_c$.
$$
f(x) = \sum_{x_c} \alpha_{x_c} k(x, x_c) = 
\sum_{x_c} \alpha_{x_c} K_{x_c}(x)
$$
where the $\{\alpha_{x_c}\}$ are determined by an optimization process and the sum runs over all the training data or centers.

A RKHS for a kernel $k$ is, roughly, the span of all the $K_{x_c}(x)$, i.e. the space of functions that can be produced as a linear combination of the kernels with different centers. (Strictly speaking, one must deal with a continuous analog of this statement.)
If any continuous function $f$ can be reproduced to arbitrary accuracy by choosing sufficient centers $\{x_c\}$ and weights $\{\alpha_{x_c}\}$, the kernel $k$ is called universal; i.e. the RKHS is dense in the space $C$ of continuous functions.

## An interesting inner product

A Hilbert space is a complete inner product space.
To be a Hilbert space, the RKHS therefore requires an inner product,
$$\langle f_1, f_2 \rangle$$
that takes in two functions in the RKHS as arguments are returns some notion of how similar they are.

If you are used to inner products equipped to function spaces, you may be familiar with the following inner product,
$$
\langle f_1, f_2 \rangle \equiv \int f_1(x) f_2(x) dx,
$$
which is in some sense the simplest continous generalization of the finite-dimensional dot product, $\langle \mathbf{v}, \mathbf{w}\rangle \equiv \sum_i v_iw_i$.
This, however, is not the inner product of a RHKS.

The inner product of a RKHS, applied to two functions $K_{x_c}$ and $K_{y_c}$ is
not $\langle K_{x_c}, K_{y_c} \rangle = \int K_{x_c}(x) K_{y_c}(x)dx$, but rather
$$
\langle K_{x_c}, K_{y_c} \rangle = k(x_c, y_c).
$$
Note that no integral is involved, which is also computationally advantageous.
What is the inner product between $K_{y_c}$ and some arbitrary function $f(x)$ in the RKHS?
By linearity of the inner product we have
$$
\langle f, K_{y_c} \rangle = 
\langle \sum_{x_c} \alpha_{x_c} K_{x_c}, K_{y_c} \rangle = 
$$
$$
\sum_{x_c} \alpha_{x_c} \langle K_{x_c}, K_{y_c} \rangle = 
\sum_{x_c} \alpha_{x_c} k(x_c, y_c) = f(y_c).
$$
This is called the reproducing property, the essential mathematical feature of an RKHS.
The reproducing property is the property that the inner product of a function $f$ in the RKHS, with a kernel centered at $x$ is simply $f(x)$.

Similarly, the inner product of an arbitrary pair of functions in the RKHS is
$$
\langle f, g \rangle = 
\sum_{x_c, y_c} \alpha_{x_c} \alpha_{y_c} k(x_c, y_c)
$$
where we have let $g(x) = \sum_{y_c} \alpha_{y_c} K_{y_c}(x).$

Thus, the inner product of a RKHS is not the ordinary continuous generalization of the dot product, but deeply related to function evaluation.

## Function evaluation as a linear operation

It is at first glance very strange that one can perform function evaluation in a RKHS via a linear operation, namely the inner product. 
How can function evaluation be linear and emerge from an inner product operation?
In fact, this idea is not as strange as it sounds and emerges even with simpler inner products.
If we consider again the standard inner product 
$$
\langle f, g \rangle \equiv \int f(x)g(x)dx,
$$
and let $\delta_y \equiv \delta(x-y)$ we in turn find that
$$
\langle f, \delta_y \rangle = \int f(x)\delta(x-y)dx = f(y).
$$
Hence, inner products and function evaluation are quite closely related even without RKHS.
The RKHS, however, represents a very powerful generalization of this idea to spaces produced from kernel functions.

The RKHS, moreover, supports the rigorous foundation underlying the famous ``kernel trick'', which is based on the observation that given a feature map $\psi(x) \in \mathbb{R}^N$, where $N \to \infty$ possibly, one can naturally defines $k(x, y) \equiv \psi(x)^T\psi(y)$, i.e. the kernel is the inner product of two feature maps.
This in turn enables solving a possibly infinite dimensional problem (finding the infinite-dimensional $w$ such that $f(x) \approx \psi(x)^T$) in finite dimensions, by finding the $\{\alpha_{x_c}\}$ described above instead---a notable advantage of kernel methods.

## Further reading/watching

[Kernel method (Wikipedia)](https://en.wikipedia.org/wiki/Kernel_method)

[Representer Theorem (Wikipedia)](https://en.wikipedia.org/wiki/Representer_theorem)

[Kernel methods in machine learning lectures by Julien Mairal and Jean-Philippe Vert](https://www.youtube.com/watch?v=IzGS8uKc5E4)

[Bishop, 2006 (Ch. 6).](https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)

[Aronszajn, 1950. "Theory of Reproducing Kernels"](https://www.jstor.org/stable/1990404)

[Steinwert, 2001. "On the influence of the kernel on the consistency of Support Vector Machines](https://www.jmlr.org/papers/volume2/steinwart01a/steinwart01a.pdf)