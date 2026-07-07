# 1. Architecture and Backend Manager

This document explains the overarching architectural design of MrMustard's backend integration. It covers how the abstract mathematics layer works, how `BackendManager` routes executions, and the lazy-loading process for the NumPy and JAX backends.

## The Goal of the Architecture

Quantum optics simulations and differentiable programming involve heavy tensor operations, Riemannian manifold optimizations, and linear algebra. To maintain code flexibility and prevent user lock-in to a single framework, MrMustard employs an agnostic backend architecture.

Instead of calling `numpy` or `jax` directly in the core physics code, MrMustard calls `mrmustard.math`. The `math` module acts as a smart router that forwards the call to the active backend.

## The `BackendManager` Singleton

At the heart of `mrmustard.math` is the `BackendManager` class (located in `mrmustard/math/backend_manager.py`).

### Initialization and Metaprogramming
When you run `import mrmustard.math as math`, a python script in `mrmustard/math/__init__.py` overwrites the loaded module with a singleton instance of the `BackendManager`:

```python
# mrmustard/math/__init__.py
from .backend_manager import BackendManager
sys.modules[__name__] = BackendManager()
```
This metaprogramming trick means that every time you call `math.cos()`, you are actually calling a method on the `BackendManager` singleton instance.

### `BackendBase` Interface
Both `BackendNumpy` and `BackendJax` inherit from `BackendBase`. This ensures they implement a unified interface of exact methods. The `BackendManager` acts as the interface facade. If an operation is requested (e.g., `math.exp()`), `BackendManager` delegates the execution via an internal `_apply` method.

```python
def _apply(self, fn: str, args: Sequence[Any] = (), kwargs: dict | None = None, backend_name: str | None = None) -> Any:
    # Resolves the backend and uses `getattr` to execute the function on it
    kwargs = kwargs or {}
    backend = self.get_backend(backend_name) if backend_name else self.backend
    try:
        attr = getattr(backend, fn)
    except AttributeError:
        raise NotImplementedError(...)
    return attr(*args, **kwargs)
```

## Lazy Loading of Dependencies

Libraries like `jax` are extremely heavy to load into memory. To ensure fast startup times for users only doing NumPy numerical work, MrMustard lazy-loads backends.

The `backend_manager.py` defines a `lazy_import()` function utilizing Python's `importlib.util`.
```python
def lazy_import(module_name: str):
    spec = importlib.util.find_spec(module_name)
    module = importlib.util.module_from_spec(spec)
    loader = importlib.util.LazyLoader(spec.loader)
    return module, loader
```

The dictionary `all_modules` keeps references to these lazy modules. Only when `math.change_backend("jax")` is called does the loader actually execute the module code and load `jax` into memory.

## Switching Backends dynamically

A core feature is the ability to swap backends globally at runtime:

```python
math.change_backend("jax")
```

When this is called:
1. `BackendManager` checks if the requested backend is different from the current one.
2. It fetches the backend object (triggering the lazy load if it's the first time).
3. It calls `self._bind()`, which re-binds data types (like `math.float64`, `math.complex128`) to the specific primitives used by the newly active backend (e.g., `jax.numpy.float64`).

## Handling Tensors

The concept of a "Tensor" is abstracted. In MrMustard, a tensor might be a `numpy.ndarray` or a `jax.Array`. To seamlessly integrate these, `BackendManager` provides typecasting utilities:
- `math.astensor(array)`: Forces the input into the native tensor format of the active backend.
- `math.asnumpy(tensor)`: A guarantee that the returned value is specifically a standard `numpy.ndarray`, which is necessary when writing data to disk, generating plots with Matplotlib, or interfacing with legacy SciPy functions.

### The Ecosystem Flow
1. **User code** -> calls `math.complex_gaussian_integral(...)`.
2. **`BackendManager`** -> catches the call, identifies the active backend (e.g., `BackendJax`).
3. **`_apply()`** -> calls `BackendJax.complex_gaussian_integral_1_batched`.
4. **Backend Implementation** -> executes using JAX tracing, XLA compiling, and autodiff support.