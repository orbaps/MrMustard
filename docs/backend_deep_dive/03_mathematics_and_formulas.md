# 3. Mathematics and Formulas

MrMustard performs rigorous calculations spanning Phase space, Fock space, and Bargmann representations. The backends (`backend_numpy.py` and `backend_jax.py`) are strictly responsible for implementing the heavy mathematical lifting for these transitions.

This document breaks down the advanced mathematical operations, their mathematical formulations, and how they are computed by the backends.

## 1. Complex Multidimensional Gaussian Integrals

To contract tensors representing quantum states and gates (such as projecting a state onto a measurement), MrMustard frequently integrates over complex multivariate Gaussian distributions.

The integration measure is given by:
$$ d\mu(z) = \exp(-|z|^2) \frac{d^{2m}z}{\pi^m} = \frac{1}{\pi^m}\exp(-|z|^2) d\mathrm{Re}(z) d\mathrm{Im}(z) $$

### `complex_gaussian_integral_1`
This method solves integrals of the form:
$$ \int_{\mathbb{C}^m} d\mu(z) \exp\left( \frac{1}{2}(z, z^*, \beta)^T A (z, z^*, \beta) + (z, z^*, \beta)^T b \right) $$
Where $z \in \mathbb{C}^m$, $\beta \in \mathbb{C}^N$.

The backend partitions $A$ and $b$ into blocks associated with the integration variables $z$ and $z^*$ (indices `idx12`), and the remaining free variables $\beta$.
$$ A = \begin{pmatrix} B & C^T \\ C & D \end{pmatrix},\quad b = \begin{pmatrix} g \\ h \end{pmatrix} $$

**The Backend Solution:**
The result of the integral yields a new Gaussian parameterized by:
$$ A_{\mathrm{out}} = D - C M^{-1} C^T $$
$$ b_{\mathrm{out}} = h - C M^{-1} g $$
$$ \log c_{\mathrm{out}} = -\frac{1}{2} g^T M^{-1} g + \frac{1}{2} \log(\det(iM^{-1})) $$
Where $M = B - X$, and $X$ is a block matrix with zeros on the diagonal and identities on the off-diagonal.

### `complex_gaussian_integral_2`
This handles the integration over the product of *two* Gaussian exponentials (e.g., when merging a State with a Transformation):
$$ \int_{\mathbb{C}^m} d\mu(z) \exp\left(\frac{1}{2}(z,\beta)^T A_1 (z,\beta) + (z,\beta)^T b_1\right) \exp\left(\frac{1}{2}(z^*,\gamma)^T A_2 (z^*,\gamma) + (z^*,\gamma)^T b_2\right) $$

**The Backend Solution:**
Partitioning $A_1, b_1$ and $A_2, b_2$:
$$ A_1 = \begin{pmatrix} A & C^T \\ C & B \end{pmatrix},\quad b_1 = \begin{pmatrix} g \\ h \end{pmatrix},\qquad A_2 = \begin{pmatrix} D & F^T \\ F & E \end{pmatrix},\quad b_2 = \begin{pmatrix} i \\ j \end{pmatrix} $$

The exact output parameterized by the backend is:
$$ L = (A D - I)^{-1} $$
$$ A_{\mathrm{out}} = \begin{pmatrix} B - C D L C^T & -F L C^T \\ -F L C^T & E - F L A F^T \end{pmatrix} $$
$$ b_{\mathrm{out}} = \begin{pmatrix} h - C(D L^T g + L i) \\ j - F(A L i + L^T g) \end{pmatrix} $$
$$ \log c_{\mathrm{out}} = -\frac{1}{2}\left[g^T D L^T g + 2 g^T L i + i^T A L i\right] + \frac{1}{2} \log(\det(-L)) $$

These matrix inversions ($M^{-1}$ and $(AD - I)^{-1}$) are computed using `math.solve` and `math.inv` directly inside the backend.

---

## 2. Multidimensional Hermite Polynomials (Fock Space Representation)

To convert Bargmann/Phase space representations into standard Fock space (photon number amplitudes), MrMustard uses multidimensional Hermite polynomials. The Fock amplitudes are given by the Taylor series of:
$$ \exp\left(c + b x + \frac{1}{2}x^T A x\right) $$
evaluated at $x=0$, but with a normalization factor of $\sqrt{n!}$ in the denominator (instead of $n!$).

### `hermite_renormalized`
The backends implement a recurrence relation to calculate the tensor of Fock amplitudes up to a specific shape/cutoff.

**Standard vs Stable execution:**
Because calculating large photon number cutoffs can lead to numerical instability, the backends accept a `stable=True|False` parameter.
- If `False`: Uses a fast, vanilla recurrence.
- If `True`: Uses a numerically stable (but slower) algorithm to prevent precision loss at high photon numbers.

### Specialized Fock Variations
- **`hermite_renormalized_diagonal`**: Calculates only the diagonal of the Fock density matrix. This is heavily optimized because it corresponds physically to Photon Number Resolving (PNR) detection probabilities. The equation changes to $\exp(c + b x - A x^2)$ and a specialized loop is executed.
- **`hermite_renormalized_binomial`**: Calculates the polynomials up to a global L2 norm constraint rather than a strict cubic shape tensor.

---

## 3. Manifolds and Riemannian Gradients

During optimization, variables might be constrained to specific physical groups (e.g., Symplectic transformations or Unitary matrices). Standard Euclidean gradient descent will pull the matrices off these manifolds. The backend handles the metric projections.

### Euclidean to Symplectic Gradient (`euclidean_to_symplectic`)
Given an ordinary Euclidean gradient tensor $dS_{\text{euclidean}}$, the backend computes the Riemannian gradient on the Symplectic manifold $Sp(2N)$:
$$ Z = S^T \cdot dS_{\text{euclidean}} $$
$$ \text{Symplectic Gradient} = \frac{1}{2} \left( Z + J \cdot Z^T \cdot J \right) $$
Where $J$ is the canonical Symplectic form:
$$ J = \begin{pmatrix} 0 & I \\ -I & 0 \end{pmatrix} $$

### Euclidean to Unitary Gradient (`euclidean_to_unitary`)
For a unitary matrix $U \in U(N)$:
$$ Z = U^\dagger \cdot dU_{\text{euclidean}} $$
$$ \text{Unitary Gradient} = \frac{1}{2} \left( Z - Z^\dagger \right) $$

### Euclidean to Siegel Disk Gradient (`euclidean_to_siegel`)
For optimization in the open Siegel disk $\mathcal{D}_g$ (the space of complex symmetric matrices where $I - Z^* Z \succ 0$), MrMustard employs the Bergman metric.
$$ \mathrm{grad} f(Z) = \frac{1}{2} \left( M + M^T \right) $$
Where $M = (I - Z Z^*) \cdot dZ_{\text{euclidean}} \cdot (I - Z^* Z)$.