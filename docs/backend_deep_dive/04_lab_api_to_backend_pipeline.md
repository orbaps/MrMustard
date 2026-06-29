# 4. Lab API to Backend Pipeline

MrMustard separates its user-facing interface (`mrmustard.lab`) from its physics engine (`mrmustard.physics`), which in turn relies on the math backends (`mrmustard.math`). This document traces the execution lifecycle of a standard quantum optics command to show exactly how the layers interact.

## The API Trace

Consider the following user script:
```python
from mrmustard.lab import Coherent, Sgate
from mrmustard import math

state = Coherent(mode=0, alpha=1.0)
gate = Sgate(mode=0, r=0.5)

output_state = state >> gate
```

Here is the step-by-step trace of how this code is executed.

### Step 1: Instantiation (`mrmustard.lab`)
When `Coherent(mode=0, alpha=1.0)` is called:
1. The `Coherent` class inherits from `Ket` (which inherits from `State`).
2. Inside `Coherent.__init__`, it instantiates an `AnsatzFactory`.
3. The `AnsatzFactory` is populated with a `ansatz_dict` linking to the `coherent_state` generator function, noting that its primary representation is `ReprEnum.BARGMANN`.
4. It initializes the `Parameter` object for `alpha`, casting `1.0` into the active backend's tensor type via `math.astensor`.

### Step 2: The `>>` Operator (Circuit Application)
When the user executes `state >> gate`, Python invokes the `__rshift__` operator.
1. In MrMustard, `__rshift__` on a `State` object checks the right-hand operand (the `Sgate`).
2. It calls an internal application method, passing the state into the transformation.
3. The actual combination of these two objects does not happen in `lab`. The `lab` classes act as wrappers for the underlying `physics` objects. The combination is deferred to the underlying `Ansatz` objects.

### Step 3: Resolution via `Ansatz` (`mrmustard.physics`)
MrMustard uses `Ansatz` classes (like `PolyExpAnsatz`) to store the raw tensors $(A, b, c)$ that define the state/gate in phase space.
1. When combining the `Coherent` state and the `Sgate`, the physics engine looks at their wires (modes).
2. It determines that they need to be contracted (i.e., a matrix multiplication in phase space or an integration in Bargmann space).
3. The `Ansatz` class uses a method like `_contract` or `__matmul__`.

### Step 4: The Math Backend Execution (`mrmustard.math`)
The physics engine now requires a tensor contraction.
1. It calls the `BackendManager` singleton via `math.complex_gaussian_integral_2(...)` passing in the $A$ matrices, $b$ vectors, and $c$ scalars of both the State and the Gate.
2. The `BackendManager` routes this to the active backend (e.g., `BackendNumpy.complex_gaussian_integral_2_single`).
3. The backend executes the matrix inversions, transposes, and dot products using `np.linalg.solve`, `np.matmul`, etc.
4. The backend returns the new contracted $A_{out}$, $b_{out}$, and $c_{out}$ tensors.

### Step 5: Returning to the User
1. The new tensors are wrapped inside a new `PolyExpAnsatz` object.
2. This `Ansatz` is wrapped inside a new `State` object (representing the final transformed state).
3. This object is assigned to the user's `output_state` variable.

## Switching Representations
If the user then calls `output_state.fock_array(shape=(10))`:
1. The `lab` module sees the request for a Fock representation.
2. It calls the `physics` module to convert the Bargmann `Ansatz` into a Fock representation.
3. The `physics` module calls `math.hermite_renormalized(...)`.
4. The backend runs the Numba-compiled or JAX-jitted multidimensional Hermite polynomial recurrence relation.
5. The resulting Fock tensor is returned to the user.