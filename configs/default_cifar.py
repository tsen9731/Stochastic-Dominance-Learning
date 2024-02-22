"""Default Hyperparameter configuration."""

import ml_collections


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()
  config.dataset_path = 'datasets/'
  config.result_path = 'results/'
  config.learning_rate = 0.1
  config.momentum = 0.9
  config.batch_size = 128
  config.num_epochs = 200
  
  loss_list = ['standard', 'sd_2nd_cdf', 'mean_risk']
  config.loss = loss_list[1]

  config.buffer_args = {'max_length': 10*config.batch_size, 'min_length': config.batch_size, 'sample_batch_size': config.batch_size, 'add_batches': True}

  config.seed = 0
  config.save_log = False
  config.save_sample_loss = False
  config.task = 'cifar'
  return config


def metrics():
  return []
