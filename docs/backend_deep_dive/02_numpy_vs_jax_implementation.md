# 2. NumPy vs. JAX: Implementation Deep Dive

MrMustard supports two independent backends located in `mrmustard/math/backend_numpy.py` and `mrmustard/math/backend_jax.py`. This document explores how they functionally differ, how operations are mapped, and how advanced topics like JIT compilation, XLA, and Batched computations are handled.

## The `BackendNumpy` Implementation

`BackendNumpy` is designed for CPU-bound, single-threaded numerical evaluation. It provides the highest baseline compatibility.

### Core Mappings
Most operations in `BackendNumpy` are direct wrappers around `numpy` or `scipy.linalg`.
- `math.cos` -> `np.cos`
- `math.matmul` -> `np.matmul`
- `math.expm` -> `scipy.linalg.expm`

### Handling Optimization
NumPy does not inherently support Automatic Differentiation. When a user tries to call `math.value_and_gradients()` while the NumPy backend is active, it will fail. `BackendNumpy` is purely a forward-pass execution framework.

### Performance via Numba JIT
Even though NumPy does not support autograd or XLA compilation, MrMustard accelerates certain bottleneck algorithms in NumPy using **Numba**.
In `mrmustard/math/numba/`, the heavy operations for constructing Fock representations via recurrence relations are written in Python but compiled down to LLVM machine code via `@numba.njit`.

When `math.hermite_renormalized()` is called in the NumPy backend, it internally routes to the Numba-compiled `compactFock` logic to achieve near C-level speed for generating Fock amplitudes.

## The `BackendJax` Implementation

`BackendJax` leverages Google's JAX library. It maps operations to `jax.numpy` and `jax.scipy`, which function almost identically to their NumPy counterparts but operate on immutable JAX arrays.

### JIT Compilation and XLA
JAX uses XLA (Accelerated Linear Algebra) to compile entire sequences of tensor operations into highly optimized kernels for CPU, GPU, or TPU.
In `backend_jax.py`, performance-critical functions are explicitly decorated with `@jax.jit`.

```python
@jax.jit
def complex_gaussian_integral_1_batched(...):
    # This entire block is fused into a single XLA operation graph when called
```

### Automatic Differentiation
The true power of `BackendJax` is autodiff. MrMustard components (like states and gates) can have their properties defined as `Trainable` `Variable` parameters.

JAX uses a functional tracing mechanism. When an optimization step is run, `BackendJax` provides `value_and_gradients(cost_fn, parameters)`. This method wraps JAX's native `jax.value_and_grad`:
1. It unrolls the `cost_fn` across the circuit parameters.
2. It tracks the reverse-mode gradient tape.
3. It returns both the forward evaluation of the cost function (the scalar loss) and the gradients of that loss with respect to all the input parameters.

### Functional Immutability & Updates
JAX arrays are immutable. Therefore, `update_add_tensor` (which updates values at specific indices) is implemented completely differently across the two backends.
- **NumPy:** Modifies the array in-place (`tensor[indices] += values`).
- **JAX:** Must use `tensor.at[indices].add(values)` which returns a brand new tensor object representing the updated array without mutating the original.

### JAX PyTrees and Flattening
To allow JAX to differentiate through MrMustard's complex objects (like `Ansatz` or `Parameter`), `BackendJax` registers them as JAX PyTrees. This is handled by defining `_tree_flatten` and `_tree_unflatten` methods. This allows JAX to unravel complex Python objects into a flat list of arrays that the XLA compiler can differentiate, and then re-assemble them afterward.

## Batched Computations (`vmap`)

Quantum optics often requires simulating many circuits simultaneously (e.g., Monte Carlo simulations or scanning parameter spaces).

- **NumPy approach:** Handles batching via standard NumPy broadcasting rules (`np.broadcast_to`).
- **JAX approach:** The `backend_jax.py` utilizes `jax.vmap` (Vectorizing Map). When operations like Gaussian Integrals are called, if the input tensors have an extra batch dimension, `jax.vmap` is used to functionally parallelize the calculation across the batch axis without needing a slow Python `for` loop.

For example, `complex_gaussian_integral_1_batched` in JAX structurally wraps the integral calculation inside a `vmap` call, allowing batched quantum states to be evaluated natively by the XLA compiler.