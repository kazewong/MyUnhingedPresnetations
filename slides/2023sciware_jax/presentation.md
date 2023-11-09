# Welcome to Slides


---

# Jax basic - your normal python

```python
import jax.numpy as jnp
from jax import grad, jit, vmap

def predict(params, inputs):
    for W, b in params:
        outputs = jnp.dot(inputs, W) + b
        inputs = jnp.tanh(outputs)
    return outputs

```

---

# Jax basic - grad
    
```python

```

---

# Jax basic - vmap
    
```python

```

---

# Jax basic - jit
    
```python

```

---

# Jax basic - EZ GPU

```python

```

---

# Multi-GPU same node

```python

```

---

# Multi-GPU multiple node

```python

```

---