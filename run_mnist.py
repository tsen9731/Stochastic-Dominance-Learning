from absl import app
from absl import flags
from absl import logging
# from clu import platform

import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

import jax
from ml_collections import config_flags
from flax.training import train_state
from flax import linen as nn
import optax
import jax.numpy as jnp
from functools import partial
import train
from utils import TrainableModel, SDTrainState
from sd_loss import sd_2nd_cdf, mean_risk
# replay buffer
import flashbax as fbx

flags.DEFINE_string('workdir', '/tmp/mnist', 'Directory to store model data.')
config_flags.DEFINE_config_file(
    'config',
    'configs/default_mnist.py',
    'File path to the training hyperparameter configuration.',
    lock_config=True,
)
FLAGS = flags.FLAGS

import numpy as np
import torch
import torch.utils.data as data
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms

# Transformations applied on each image => bring them into a numpy array and normalize between -1 and 1
def image_to_numpy(img):
    img = np.array(img, dtype=np.float32)
    img = (img / 255. - 0.5) / 0.5
    return img

# We need to stack the batch elements as numpy arrays
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

def get_dataloader(config):
  train_dataset = MNIST(root=config.dataset_path, train=True, transform=image_to_numpy, download=True)

  test_set = MNIST(root=config.dataset_path, train=False, transform=image_to_numpy, download=True)

  train_loader = data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, collate_fn=numpy_collate, pin_memory=True)

  test_loader  = data.DataLoader(test_set, batch_size=config.batch_size, shuffle=False, drop_last=False, collate_fn=numpy_collate, pin_memory=True)
  
  return train_loader, test_loader


class CNN(nn.Module):
  """A simple CNN model."""

  @nn.compact
  def __call__(self, x):
    x = jnp.expand_dims(x, axis=3)
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)
    return x

class Trainer(TrainableModel):
  def __init__(self, config):
    super(Trainer, self).__init__(config)
    self.rng = jax.random.key(config.seed)
    self.model=CNN()
    self.state = None
    self.buffer = fbx.make_item_buffer(**config.buffer_args)
    self.create_fn()

  def create_train_state(self, batch):
    """Creates initial `TrainState`."""
    self.rng, init_rng = jax.random.split(self.rng)
    imgs, labels = batch
    params = self.model.init(init_rng, imgs)['params']
    tx = optax.sgd(self.config.learning_rate, self.config.momentum)
    state = SDTrainState.create(apply_fn=self.model.apply, params=params, tx=tx)

    loss, _ = self.eval_step(state, batch)
    buffer_state = self.buffer.init(loss[0])
    buffer_state = self.buffer.add(buffer_state, loss[1:])
    state = state.replace(buffer_state=buffer_state)

    self.state = state

  def create_fn(self):

    def calc_batch_loss(params, batch):
      imgs, labels = batch
      logits = self.model.apply({'params': params}, imgs)
      one_hot = jax.nn.one_hot(labels, 10)
      batch_loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot)
      acc = jnp.mean(jnp.argmax(logits, -1) == labels)
      metrics = {'ce_loss': jnp.mean(batch_loss), 'accuracy': acc, 'batch_loss': batch_loss}
      return batch_loss, metrics
    
    def calc_final_loss(batch_loss, batch_ref=None):
      ce_loss = jnp.mean(batch_loss)
      if self.config.loss == 'standard':
        loss = ce_loss
      elif self.config.loss == 'mean_risk':
        loss = mean_risk(batch_loss)
      elif self.config.loss == 'sd_2nd_cdf':
        loss = sd_2nd_cdf(-batch_loss, -batch_ref)
      return loss

    # Training function
    def train_step(state, batch, rng=None):

      def loss_fn(params):
        batch_loss, metrics = calc_batch_loss(params, batch)
        batch_ref = None
        if self.config.loss == 'sd_2nd_cdf':
          batch_ref = self.buffer.sample(state.buffer_state, rng)['experience']
        final_loss = calc_final_loss(batch_loss, batch_ref)
        return final_loss, metrics

      (final_loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

      state = state.apply_gradients(grads=grads)
      new_buffer_state = self.buffer.add(state.buffer_state, metrics['batch_loss'])
      state = state.replace(buffer_state=new_buffer_state)
      return state, metrics
    
    def eval_step(state, batch):
      batch_loss, metrics = calc_batch_loss(state.params, batch)
      return batch_loss, metrics
    
    self.train_step = jax.jit(train_step)
    self.eval_step = jax.jit(eval_step)

  @staticmethod
  def format_log(epoch, train_metrics, test_metrics=None):
    if test_metrics is not None:
      return 'epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f, test_accuracy: %.2f' \
      % (epoch, train_metrics['ce_loss'], train_metrics['accuracy'] * 100, test_metrics['ce_loss'], test_metrics['accuracy'] * 100)
    else:
       return 'epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f' \
      % (epoch, train_metrics['ce_loss'], train_metrics['accuracy'] * 100)

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  
  logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX local devices: %r', jax.local_devices())

  DATASET_PATH = FLAGS.workdir
  config = FLAGS.config

  seed = 0
  config.seed = seed
  torch.manual_seed(config.seed)
  train.train_and_evaluate(config, Trainer(config), get_dataloader, FLAGS.workdir)

  # for seed in range(10):
  #   config.seed = seed
  #   torch.manual_seed(config.seed)
  #   for loss in ['standard', 'sd_2nd_cdf']:
  #     config.loss = loss
  #     train.train_and_evaluate(config, Trainer(config), get_dataloader, FLAGS.workdir)

if __name__ == '__main__':
  flags.mark_flags_as_required(['config', 'workdir'])
  app.run(main)
