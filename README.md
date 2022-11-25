Multiwandb
==========
A very simple (hacky) work around for a limitation of wandb which makes it awkward to include multiple runs in a single python script. 

Automatic batching in JAX makes it fairly straight forward to run multiple random seeds for a given experiment on a single GPU (see for example https://willwhitney.com/parallel-training-jax.html). However, there is no straightforward way to tell wandb to treat these as seperate runs. The multiwandb class gets around this by simply using multiprocessing to spin up multiple wandb runs in a single script which you can then log data to. See example.py for basic usage (this example requires JAX).