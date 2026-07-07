# MrMustard Backend Details

This document provides an exhaustive dive into how backends are architected, managed, and implemented in the `MrMustard` codebase.

## Overview of the Backend Architecture

The `MrMustard` library operates on an abstract mathematical backend, which allows the core physics and quantum optics logic to be written in an agnostic way. This is achieved via a **Backend Manager** architecture located in the `mrmustard.math` module.

When you `import mrmustard.math as math`, you are actually interacting with a `BackendManager` singleton object instance. This object dynamically binds and routes function calls to the active underlying backend (either NumPy or JAX).

### How `BackendManager` Works
- **Dynamic Method Resolution:** The `BackendManager` class implements methods like `math.matmul()`, `math.exp()`, and `math.complex_gaussian_integral_1()`. Instead of executing logic directly, it delegates execution to the currently active backend object via a helper `_apply()` method (`getattr(backend, fn)(*args, **kwargs)`).
- **Lazy Loading:** Backends are lazy-loaded to prevent importing heavy dependencies (like `jax`) until they are strictly needed. This happens through the `lazy_import()` function and a dictionary `all_modules`.
- **Default Backend:** By default, `MrMustard` uses the NumPy backend (`BackendNumpy`).
- **Singleton Pattern:** Overriding `__new__`, `BackendManager` ensures only a single instance manages state across the entire process. The module replacement hack `sys.modules[__name__] = BackendManager()` in `mrmustard/math/__init__.py` makes the instance masquerade as a module.

### Switching Backends
Switching the mathematical backend is achieved simply by calling `change_backend()`:
```python
from mrmustard import math

# Currently using numpy
math.cos(0.1)

# Switch to JAX
math.change_backend("jax")
math.cos(0.1) # Now executed by JAX
```

## NumPy vs. JAX: Algorithmic and Structural Differences

`MrMustard` currently supports two backends extending from the abstract `BackendBase`:
1. **NumPy (`mrmustard.math.backend_numpy.BackendNumpy`)**
2. **JAX (`mrmustard.math.backend_jax.BackendJax`)**

### 1. The NumPy Backend
- **Use Case:** Ideal for standard simulation, numerical accuracy checks, fast single-threaded execution, and scenarios where gradients/optimization are not required.
- **Under the Hood:** Relies directly on Python's `numpy`, `scipy` (for functions like `expm`, `sqrtm`), and standard Python execution. It employs `Numba` JIT compilation (via `mrmustard.math.numba.compactFock~`) under the hood to highly optimize heavy operations like multidimensional Hermite polynomial calculations (`hermite_renormalized`).

### 2. The JAX Backend
- **Use Case:** Necessary for optimization, training of trainable parameters, and automatic differentiation. Allows calculating `math.value_and_gradients()` over a circuit parameterization.
- **Under the Hood:** Replaces NumPy operations with `jax.numpy` and `jax.scipy` equivalents. It is structurally built to handle JAX’s immutable arrays, random key state management, and PyTree structures.
- **Specific Differences:**
  - JAX uses XLA (Accelerated Linear Algebra) compiling for JIT.
  - The JAX backend heavily uses JAX's `jax.jit` decorators on the specialized math operations like Gaussian integrals and Hermite polynomials to make them performant during Autodiff sweeps.
  - In JAX, the `complex_gaussian_integral` handles batched tensor operations differently by taking advantage of `jax.vmap` internally where appropriate, ensuring seamless batched differentiation.

## Exhaustive List of Backend Operations

Every backend must implement a rigorous set of operations which are then mirrored in `BackendManager`.

### Basic Tensor Attributes and Creation
- `abs`, `angle`, `real`, `imag`, `conj`: Complex number interactions.
- `arange`, `ones`, `zeros`, `eye`, `full`, `ones_like`, `zeros_like`, `infinity_like`, `eye_like`: Array initialization.
- `astensor`, `asnumpy`, `cast`: Typing conversions and boundary traversals.
- `make_complex`: Creating complex tensors from two real tensors.
- `shape`: Getting array dimensions.

### Array Manipulation and Reductions
- `reshape`, `expand_dims`, `squeeze`, `transpose`, `swapaxes`, `moveaxis`: Dimension manipulations.
- `concat`, `stack`, `block`: Joining arrays.
- `gather`, `pad`, `tile`, `broadcast_to`, `broadcast_arrays`: Resizing and extracting.
- `atleast_nd`: Enforcing minimum rank.
- `diag`, `diag_part`, `diagonal`: Diagonal interactions.
- `sum`, `prod`, `max`, `min`, `mean`, `norm`, `trace`: Reduction operations.
- `argmax`, `argmin`, `argsort`, `sort`: Sorting and indexing.
- `update_add_tensor`: In-place like addition (handled purely functionally in JAX).

### Logical Operations
- `all`, `any`, `allclose`, `equal`, `maximum`, `minimum`, `isnan`, `iscomplexobj`, `issubdtype`.
- `conditional`: Execute one of two functions based on a condition (JAX relies on `jax.lax.cond`).
- `error_if`: Asserts conditions at runtime.

### Standard Mathematical and Linear Algebra Operations
- `cos`, `sin`, `tan`, `cosh`, `sinh`, `tanh`: Trigonometry.
- `exp`, `log`, `pow`, `sqrt`, `lgamma`: Exponents and Logs.
- `xlogy`: Computes `x * log(y)`, safely returning 0 if `x == 0`.
- `matmul`, `matvec`, `tensordot`, `kron`, `outer`: Multiplications.
- `einsum`: Einstein summation (optimized via `opt_einsum`).
- `det`, `inv`, `pinv`, `solve`: Matrix operations.
- `eigvals`, `eigh`, `expm`, `sqrtm`: Matrix eigensolving and calculus.

### Advanced Math / Quantum Optics Specifics
The backends provide domain-specific operations optimized for phase and Fock space calculations.

- **`complex_gaussian_integral_1` / `_2`**:
  - Calculates complex multidimensional Gaussian integrals. Crucial for transitioning components between different phase-space representations.
  - Both backends implement separated implementations for single and batched execution (`_single` vs `_batched`) to strictly optimize overhead.

- **`hermite_renormalized` (and variants)**:
  - Generates multidimensional Hermite polynomials given by exponential Taylor series. It solves the exact Fock amplitudes down to machine precision.
  - Variants include:
    - `hermite_renormalized_batched`
    - `hermite_renormalized_diagonal`: Solves the diagonal of the Fock representation (PNR detection probabilities).
    - `hermite_renormalized_binomial`
    - `hermite_renormalized_1leftoverMode`: Calculates conditional density matrices.

- **Fock Lattice Strategies**:
  Used to construct gates directly in the Fock basis using sophisticated recurrence relations:
  - `displacement(alpha, shape)`
  - `squeezed(r, phi, shape)`
  - `squeezer(r, phi, shape)`
  - `beamsplitter(theta, phi, shape, method)`
  - `homodyne_projector(fock_dim, A, b, c)`

### Symplectic / Manifold Optimizations
The backend manages Riemannian geometric operations to ensure optimizations of symplectic matrices remain on the correct geometric manifolds.
- `unitary_to_orthogonal`: Embeds unitary matrices into larger orthogonal ones.
- `random_symplectic`, `random_orthogonal`, `random_unitary`, `random_siegel`: Generators for math components.
- `euclidean_to_symplectic`: Converts Euclidean gradient to Riemannian gradient on the Symplectic manifold.
- `euclidean_to_unitary`: Riemannian gradient conversion on the Unitary manifold.
- `euclidean_to_siegel`: Riemannian gradient conversion on the Siegel disk (Bergman metric).

### Constants & Metaprogramming
- The backend caches and creates commonly used matrices in Quantum Optics: `Xmat` (Position-momentum swaps), `Zmat`, `J` (Symplectic form), `rotmat` (Quadrature to complex amplitudes).
- `map_fn`: Applies functions over an unstacked tensor axis (`jax.lax.map` in JAX).
- `value_and_gradients`: Exclusively used by JAX to differentiate the `cost_fn` relative to a set of trainable parameters using `jax.value_and_grad`. (In NumPy, this fails or is not implemented).
