# Do salloc -p gpu --nodes N --ntasks-per-node G --gpus-per-node=G --cpus-per-task C -C a100
# Note that ntasks-per-node and gpus-per-node must be the same. Otherwise jax won't recognize the number of devices.
# Then srun bash "submit_script", which load the environment correctly.
# BURN BABY BURN!
import os
import time

import jax
from jax import sharding
from jax.sharding import Mesh
import numpy as np
from jax.sharding import PartitionSpec as P
import math
from jax._src.distributed import initialize
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, default=8000)
args = parser.parse_args()

initialize()
global_mesh = Mesh(np.array(jax.devices()), ('b'))

if jax.process_index() == 0:
    for k in [
        'SLURM_JOB_ID',
        'SLURM_NTASKS',
        'SLURM_NODELIST',
        'SLURM_STEP_NODELIST',
        'SLURM_STEP_GPUS',
        'SLURM_GPUS',
    ]:
        print(f'{k}: {os.getenv(k,"")}')
    print("Total number of process: ", jax.process_count())
    print("List of devices: ",jax.devices())
    print("Number of device on this process: ",jax.local_device_count())


local_shape = (args.size // jax.process_count(), args.size)
global_shape = (jax.process_count() * local_shape[0], ) + local_shape[1:]
local_array = np.arange(math.prod(local_shape)).reshape(local_shape) + jax.process_index()*1.0
arrays = jax.device_put(
    np.split(local_array, len(global_mesh.local_devices), axis = 0), global_mesh.local_devices)

sharding = jax.sharding.NamedSharding(global_mesh, P(('b'), ))
arr = jax.make_array_from_single_device_arrays(global_shape, sharding, arrays)

print(arrays[0].devices())
print(arrays[0].shape)
print(arr.shape)
print(arr.devices())

@jax.jit
def f(x, y):
    return x@y + x


f(arr, arr).block_until_ready() # Precompile

current_time = time.time()

f(arr, arr).block_until_ready()

multi_gpu_time = time.time() - current_time

if jax.process_index() == 0:
    print("Time to compute on 4 devices:", multi_gpu_time)
