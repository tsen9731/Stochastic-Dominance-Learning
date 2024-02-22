# Copyright 2023 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Default Hyperparameter configuration."""

from ml_collections import config_dict


def get_config():
  """Get the default hyperparameter configuration."""
  config = config_dict.ConfigDict()
  config.dataset_path = 'datasets/'
  config.result_path = 'results/'
  config.learning_rate = 0.1
  config.momentum = 0.9
  config.batch_size = 128
  config.num_epochs = 10

  loss_list = ['standard', 'sd_2nd_cdf', 'mean_risk']
  config.loss = loss_list[1]

  config.buffer_args = {'max_length': 10*config.batch_size, 'min_length': config.batch_size, 'sample_batch_size': config.batch_size, 'add_batches': True}

  config.seed = 0
  config.save_log = False
  config.save_sample_loss = False
  config.task = 'mnist'
  return config


def metrics():
  return []
