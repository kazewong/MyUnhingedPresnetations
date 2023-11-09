import jax
import jax.numpy as jnp
from IPython import get_ipython
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding
ipython = get_ipython()

print(jax.process_count())
print(jax.devices())
print(jax.local_device_count())

@jax.jit
def f(x,y):
    return x@y + x

x = jnp.zeros((8000,8000))

print(x.device())

ipython.run_line_magic("timeit", "f(x,x)")

sharding = PositionalSharding(mesh_utils.create_device_mesh((4,)))
y = jax.device_put(x, sharding.reshape(4,1))

print(y.devices())

ipython.run_line_magic("timeit", "f(y,y)")
