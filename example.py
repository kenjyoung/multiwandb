from multiwandb import multiwandb
import jax
from jax import jit, vmap
import jax.numpy as jnp

num_runs = 10

#initialize multiwandb with a list of configs one for each wandb instance/run (these can all be the same if desired)
configs = [{"index":i} for i in range(num_runs)]
logger = multiwandb(configs, project='multiproc_test', group=None)

key = jax.random.PRNGKey(0)

def func_1(x,key):
	return x/2+2*(2*jax.random.uniform(key)-1)
func_1 = jit(vmap(func_1,in_axes=(None,0)))

def func_2(x,key):
	return x+(2*jax.random.uniform(key)-1)
func_2 = jit(vmap(func_2,in_axes=(None,0)))

def func_3(x,key):
	return jnp.sin(2*jnp.pi*x/10)+0.5*(2*jax.random.uniform(key)-1)

# loop for 10 steps
for i in range(10):
	# generate some data from vmapped functions
	key, subkey = jax.random.split(key)
	subkeys = jax.random.split(subkey, num=num_runs)
	values_1 = func_1(i,subkeys)
	subkeys = jax.random.split(subkey, num=num_runs)
	values_2 = func_2(i,subkeys)
	data = [{"data_1":v_1, "data_2":v_2, "time_step": i} for v_1,v_2 in zip(list(values_1),list(values_2))]

	# log expects a list of values the same length as the number of runs (number of configs passed at intialization)
	logger.log(data)

	# you can also log data to only a single run at once with log at index
	for j in range(num_runs):
		key, subkey = jax.random.split(key)
		v_3 = func_3(i,subkey)
		data = {"data_3":v_3, "time_step": i}
		logger.log_at_index(data, j)
logger.end()
