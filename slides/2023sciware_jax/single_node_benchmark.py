import jax
import jax.numpy as jnp
from IPython import get_ipython
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding
import argparse
import time
ipython = get_ipython()

print("Number of processes:" , jax.process_count())
print("Number of devices:", jax.local_device_count())

# parse command line arguments of how many numbers to process

parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, default=8000)
args = parser.parse_args()

@jax.jit
def f(x,y):
    return x@y + x

x = jnp.zeros((args.size, args.size))

sharding = PositionalSharding(mesh_utils.create_device_mesh((4,)))
y = jax.device_put(x, sharding.reshape(2,2))

print("Number of devices on tensor y:", len(y.devices()))

f(y,y).block_until_ready() # Precompile

current_time = time.time()

f(y,y).block_until_ready()

multi_gpu_time = time.time() - current_time

print("Time to compute on 4 devices:", multi_gpu_time)

print("Number of devices on tensor x:", len(x.devices()))

f(x,x).block_until_ready() # Precompile

current_time = time.time()

f(x,x).block_until_ready()

single_gpu_time = time.time() - current_time

print("Time to compute on 1 device:", single_gpu_time)

print("Speedup:", single_gpu_time / multi_gpu_time)

