import arviz as az
import numpy as np
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
          return_raw=False):
    
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
  posterior = graph_sample_chain(num_results=nsteps,
                                 num_burnin_steps=num_burnin_steps,
                                 current_state=initial_chain_state,
                                 kernel = kernel, trace_fn=None)
  
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


def az_to_numpy(azdata):
  keys = azdata.posterior.all()
  return [ azdata.posterior[key].data for key in keys ]
