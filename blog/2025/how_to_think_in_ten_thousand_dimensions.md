# How to think in ten thousand dimensions
**Rich Pang**

2025-12-30

Many neuroscience and AI models operate in high-dimensional (HD) spaces, each dimension corresponding to the activation of a neuron, for instance.
HD spaces have many interesting and useful properties that can appear rather counterintuitive at first glance.

One can encode sets, for instance, by averaging HD codewords for their elements.
In low dimensions averaging throws away substantial information:
if one replaced the set $\{3, 10, 2\}$ with its average, 5, it would be hard to decode the original three elements from the number 5.
In HD spaces such averaging can in fact preserve the original elements, which may seem fairly unexpected.
But seen in a certain way, it's quite obvious.

Below are two tricks to help make HD vectors more intuitive to think about.

## Imagine high-dimensional vectors as images

Thinking of HD vectors as images makes certain properties quite clear. For instance, that sets can be represented as averages of their high-dimensional (HD) codewords can be derived either mathematically or understood intuitively by thinking of the codewords as images.

The mathematical explanation goes as follows. First, random HD vectors are almost orthogonal with very high probability:

$$
\mathbf{r}_1 \cdot \mathbf{r}_2 \approx 0
$$

when the individual elements of $\mathbf{r}_1$ and $\mathbf{r}_2$ are i.i.d.
This can be shown with a bit of algebra.
It can also be shown that it is possible to sample an exponentially large number $\exp(cN)$ of such codewords, where $N$ is the dimensionality of $\mathbf{r}_i$.
This is quite qualitatively different from the number of exactly orthogonal codewords, which is just $N$.
There are many more than $N$ HD codewords if near-orthogonality suits one's purposes.

Construct a set of codewords in this way, $\mathbf{r}_1 \dots \mathbf{r}_K$, corresponding to some set of items 1 through K, and with the codewords normalized such that $\mathbf{r}_i \cdot \mathbf{r}_i \approx 1$. Pick items $k, l, m$ and average their codewords:

$$
\mathbf{r}_{avg} = \frac{1}{3}\left(\mathbf{r}_k + \mathbf{r}_l + \mathbf{r}_m \right).
$$
To verify that this quantity stores the set $\{k, l, m\}$ we check codeword $\mathbf{r}_{k'}$ for $k' \in \{k, l, m\}$ and $k' \notin \{k, l, m\}$. We perform the check by taking the dot product $\mathbf{r}_{k'} \cdot \mathbf{r}_{avg}$.

In the former case, suppose $k' = k$. Then we have

$$
\begin{split}
\mathbf{r}_{k} \cdot \mathbf{r}_{avg} 
& = \frac{1}{3}\left(\mathbf{r}_{k} \cdot \mathbf{r}_k + \mathbf{r}_{k} \cdot \mathbf{r}_l + \mathbf{r}_{k} \cdot \mathbf{r}_m \right) \\
& \approx \frac{1}{3} + 0 + 0 = \frac{1}{3} \\
\end{split}
$$

In the latter case:

$$
\begin{split}
\mathbf{r}_{k'} \cdot \mathbf{r}_{avg} 
& = \frac{1}{3} \left(\mathbf{r}_{k'} \cdot \mathbf{r}_k + \mathbf{r}_{k'} \cdot \mathbf{r}_l + \mathbf{r}_{k'} \cdot \mathbf{r}_m \right)\\
& \approx 0 + 0 + 0 = 0 \\
\end{split}
$$
Hence, whether the dot product is different from zero reveals whether any test element is in the subset represented by $\mathbf{r}_{avg}$, with an accuracy that depends on how many elements are in the set.

While the algebra above is simple and elegant, consider the following image, which is itself the superposition of three famous images.
We have decreased each image's opacity such that each final pixel is an average of the corresponding pixels in each image.

![Superposition of three famous images: Einstein sticking his tongue out, Pink Floyd's Dark Side of the Moon album cover, and Van Gogh's Starry Night](how_to_think_in_ten_thousand_dimensions_image_superposition.png)

The average clearly retains the identities of the original images in some meaningful sense.
Images that are not in the superposition have low overlap with it, and images that are in the superposition have high overlap.
Curiously, whereas decoding the elements from the image in the algebraic manner described above requires checking each element against $\mathbf{r}_{avg}$, our brains seem to be able to decode the superimposed images quite quickly.

## Think of sparse binary vectors as one-hot vectors in an even higher dimensional space

Sparse binary vectors are useful when non-negativity and finite resolution needed, for instance when mapping codewords to binary memory elements in a computer. 
They can be intuitied about by imagining them as one-hot vectors in an even higher dimensional space.
In both cases, all elements are positive, different codewords are orthogonal (or nearly orthogonal), and the codewords are sparse, with most elements zero.

For example, this perspective makes the basic action of a Bloom filter quite natural.
Bloom filters are probabilistic data structures used for set membership querying (e.g. to test whether a URL is in a set of previously visited URLs). 
They operate by mapping each element (URL) to a collection of $Q$ bits in an array of size $B$ using a hash function, such that the bits assigned to each element are a pseudorandom deterministic function of the element.
Elements are added to the BF by setting their bits in the array to 1.
Elements are decoded by comparing the codeword for an element with the BF itself.
If fewer than $Q$ bits match, the element is not in the BF.
If all $Q$ bits match, the element may be in the BF, with a false positive rate that depends on how many elements have been stored.

The analogy lets us think of the element codewords, which are $Q$-hot vectors in a $B$-dimensional space, as 1-hot vectors in an $\sim \exp{cB}$ dimensional space.
In turn, storing elements corresponds to looking up the index "assigned" to the element in the $\sim \exp{cB}$ dimensional space and setting its bit to 1.
Querying the BF then amounts to computing the codeword for a test element and checking whether the bit at its address in the ultra high-dimensional space is set to 1.

![bloom_filter](how_to_think_in_ten_thousand_dimensions_bloom_filter.png)

Thus, sparse binary HD vectors receive the benefits of dimensionality yet act in many ways like 1-hot vectors, which may be easier to intuit about. The correspondence breaks down when the BF inevitably "fills up", but does so gracefully. The image analogy discussed above can also be relevant to Bloom filters. 

Many brain areas are also thought to exhibit sparse HD neural codes, for instance in the hippocampus or insect mushroom body, which are thought to reflect the results of pattern-separation or decorrelation operations. As we have seen, such codes can behave similarly to 1-hot codes in higher dimensional spaces.

## Further reading

* [Kanerva 2009](https://rctn.org/vs265/kanerva09-hyperdimensional.pdf)
* [Musco lecture notes](https://www.chrismusco.com/amlds2025/lectures/lec4.pdf)
* [Wegner lecture notes](https://arxiv.org/pdf/2101.05841)
* [Bloom filter Wikipedia Page](https://en.wikipedia.org/wiki/Bloom_filter)