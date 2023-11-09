# Do salloc -p gpu --nodes N --ntasks-per-node G --gpus-per-node=G --cpus-per-task C -C a100
# Note that ntasks-per-node and gpus-per-node must be the same. Otherwise jax won't recognize the number of devices.
# Then srun bash "submit_script", which load the environment correctly.
# BURN BABY BURN!
import os
import time


import jax
import jax.numpy as jnp
from jax import sharding
from jax.sharding import Mesh
import numpy as np
from jax.sharding import PartitionSpec as P
from jax.experimental.pjit import pjit
from jax.experimental import mesh_utils
from jax.experimental import multihost_utils
import math
from jax._src.distributed import initialize

initialize()
global_mesh = Mesh(np.array(jax.devices()), ('b'))

if jax.process_index() == 0:
    os.system("echo SLURM_ID: $SLURM_JOB_ID")
    os.system("echo SLURM_NTASKS: $SLURM_NTASKS")
    os.system("echo SLURM_NODELIST: $SLURM_NODELIST")
    os.system("echo SLURM_STEP_NODELIST: $SLURM_STEP_NODELIST")
    os.system("echo SLURM_STEP_GPUS: $SLURM_STEP_GPUS")
    os.system("echo SLURM_GPUS: $SLURM_GPUS")
    print(jax.process_count())
    print(jax.devices())
    print(jax.local_device_count())

loss_values = 0.

local_shape = (8, 2)
global_shape = (jax.process_count() * local_shape[0], ) + local_shape[1:]
local_array = np.arange(math.prod(local_shape)).reshape(local_shape) + jax.process_index()*1.0
arrays = jax.device_put(
    np.split(local_array, len(global_mesh.local_devices), axis = 0), global_mesh.local_devices)
sharding = jax.sharding.NamedSharding(global_mesh, P(('b'), ))
arr = jax.make_array_from_single_device_arrays(global_shape, sharding, arrays)
print(jnp.sum(multihost_utils.process_allgather(arr))+loss_values)
print(local_array.shape,  arr.shape)
print(arrays)


@jax.jit
def f(x):
    return jnp.sum(x*x)

# print(multihost_utils.process_allgather(jax.value_and_grad(f)(arr)))