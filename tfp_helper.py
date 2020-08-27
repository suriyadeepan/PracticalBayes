import arviz as az
import numpy as np
from tqdm import tqdm

from colors import colors
from random import choice
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


def evaluate(tensors, sess=None):
  if tf.executing_eagerly():
    return tf.nest.pack_sequence_as(tensors,
        [t.numpy() if tf.is_tensor(t) else t for t in tf.nest.flatten(tensors)])

  if sess is None:
    return tensors

  return sess.run(tensors)


def infer(joint_log_prob, data, variables, initial_chain_state,
          nsteps=2000, burn_in_ratio=0.5,
          step_size=0.1, num_leapfrog_steps=3,
          bijectors={}, exclude_burn_in=True,
          return_raw=False, tqdm_steps=10):
    
  # create closure of joint_log_prob function
  def log_posterior(*args):
    return joint_log_prob(data, *args)
  
  # wrap the mcmc sampling call in a @tf.function to speed it up
  @tf.function(autograph=False)
  def graph_sample_chain(*args, **kwargs):
    return tfp.mcmc.sample_chain(*args, **kwargs)
  
  unconstraining_bijectors = []
  for var in variables:
    if var in bijectors:
      unconstraining_bijectors.append(bijectors[var])
    else:
      unconstraining_bijectors.append(tfb.Identity())
          
  # prepare kernel
  kernel=tfp.mcmc.TransformedTransitionKernel(
      inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=log_posterior,
          num_leapfrog_steps=num_leapfrog_steps,
          step_size=step_size,
          state_gradients_are_stopped=True),
      bijector=unconstraining_bijectors)

  # prepare metakernel
  num_burnin_steps = int(nsteps * burn_in_ratio)
  kernel = tfp.mcmc.SimpleStepSizeAdaptation(inner_kernel=kernel,
                     num_adaptation_steps=int(num_burnin_steps * 0.8))

  # prepare chain
  initial_chain_state = [
      t * tf.ones([], dtype=tf.float32, name='init_{}'.format(var_name))
      for t, var_name in zip(initial_chain_state, variables) ]
    
  # sample from chain
  posterior = run_inference_in_steps(sample_fn=graph_sample_chain,
                                     nsteps=nsteps,
                                     burn_in_ratio=burn_in_ratio,
                                     initial_state=initial_chain_state,
                                     variables=variables,
                                     kernel=kernel, isteps=tqdm_steps)
  # evaluate posterior
  post_vars = evaluate([ post_var for post_var in posterior ])

  # exclude burn_in samples
  if exclude_burn_in:
    post_vars = [ v[num_burnin_steps:] for v in post_vars ]

  # return samples in raw form
  if return_raw:
    return post_vars
  
  # convert to arviz.InferenceData
  inference_dict = { var_name : np.expand_dims(var_trace, axis=0)
      for var_name, var_trace in zip(variables, post_vars) }
  inference_data = az.from_dict(inference_dict)
  
  return inference_data


def run_inference_in_steps(sample_fn, kernel, nsteps, burn_in_ratio,
    initial_state, variables, isteps=10):
  # calculate steps
  msteps = nsteps // 10
  mburn_in_steps = int(msteps * burn_in_ratio)
  # create metakernel
  meta_kernel = tfp.mcmc.SimpleStepSizeAdaptation(inner_kernel=kernel,
                     num_adaptation_steps=int(mburn_in_steps * 0.8))
  # set chain state
  chain_state = initial_state
  # posterior samples
  posterior = [ [] for _ in variables ]
  # sample in steps
  for i in tqdm(range(isteps)):
    samples = sample_fn(num_results=msteps,
                        num_burnin_steps=mburn_in_steps,
                        current_state=chain_state,
                        kernel=meta_kernel, trace_fn=None)
    # set chain state
    chain_state = [
      tf.constant(samples[j][-1], name='init_{}_{}'.format(var_name, i+1))
      for j, var_name in enumerate(variables) ]
    # add samples to posterior
    for j in range(len(variables)):
      posterior[j].extend(samples[j])

  return posterior


def az_to_numpy(azdata, flatten=False):
  keys = azdata.posterior.all()
  if not flatten:
    return [ azdata.posterior[key].data for key in keys ]
  else:
    return [ azdata.posterior[key].data.flatten() for key in keys ]


def az_to_dict(azdata):
  return { k : azdata.posterior[k].data for k in azdata.posterior.all() }


def plot_posterior_hist(trace, var_name):
  # convert to dictionary of samples
  samples = trace.posterior[var_name].data
  # check dimensionality of variable
  if len(samples.shape) == 2:
    samples = np.expand_dims(samples, axis=-1)
  # get dims
  dims = samples.shape[-1]

  for i in range(dims):
    # set figure size
    fig = plt.figure(figsize=(12.5, 5))
    # set title
    # plot hist
    plt.hist(samples[:, :, i].flatten(), histtype='stepfilled', bins=50,
        alpha=0.65, label="posterior of {}_{}".format(var_name, i),
        color=choice(colors), density=True)
    # set x label
    plt.xlabel("{}_{} value".format(var_name, i))
    plt.yticks([])
    # set legend
    plt.legend(loc="upper left")
    # set y limits
    # plt.ylim([0., 1.])
    # set y label
    # plt.ylabel("probability");
  plt.title("Posterior distribution")
  plt.show()
