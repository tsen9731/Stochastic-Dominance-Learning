from absl import app
from absl import flags
from absl import logging

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

flags.DEFINE_string('workdir', '/tmp/cifar', 'Directory to store model data.')
config_flags.DEFINE_config_file(
    'config',
    'configs/default_cifar.py',
    'File path to the training hyperparameter configuration.',
    lock_config=True,
)
FLAGS = flags.FLAGS

import numpy as np
import torch
import torch.utils.data as data
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms
# import tensorflow as tf

data_means = jnp.array([0.49139968,0.48215841,0.44653091])
data_std = jnp.array([0.24703223,0.24348513,0.26158784])

# Transformations applied on each image => bring them into a numpy array and normalize between -1 and 1
def image_to_numpy(img):
    img = np.array(img, dtype=np.float32)
    img = (img / 255. - data_means) / data_std
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

test_transform = image_to_numpy
# For training, we add some augmentation. Networks are too powerful and would overfit.
train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.RandomResizedCrop((32,32),scale=(0.8,1.0),ratio=(0.9,1.1)),
                                      image_to_numpy
                                     ])

def get_dataloader(config):
  train_dataset = CIFAR10(root=config.dataset_path, train=True, transform=train_transform, download=True)

  test_set = CIFAR10(root=config.dataset_path, train=False, transform=test_transform, download=True)

  train_loader = data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, collate_fn=numpy_collate, pin_memory=True)

  test_loader  = data.DataLoader(test_set, batch_size=config.batch_size, shuffle=False, drop_last=False, collate_fn=numpy_collate, pin_memory=True)
  
  return train_loader, test_loader


# Conv initialized with kaiming int, but uses fan-out instead of fan-in mode
# Fan-out focuses on the gradient distribution, and is commonly used in ResNets
resnet_kernel_init = nn.initializers.variance_scaling(2.0, mode='fan_out', distribution='normal')

class ResNetBlock(nn.Module):
    act_fn : callable  # Activation function
    c_out : int   # Output feature size
    subsample : bool = False  # If True, we apply a stride inside F

    @nn.compact
    def __call__(self, x, train=True):
        # Network representing F
        z = nn.Conv(self.c_out, kernel_size=(3, 3),
                    strides=(1, 1) if not self.subsample else (2, 2),
                    kernel_init=resnet_kernel_init,
                    use_bias=False)(x)
        z = nn.BatchNorm()(z, use_running_average=not train)
        z = self.act_fn(z)
        z = nn.Conv(self.c_out, kernel_size=(3, 3),
                    kernel_init=resnet_kernel_init,
                    use_bias=False)(z)
        z = nn.BatchNorm()(z, use_running_average=not train)

        if self.subsample:
            x = nn.Conv(self.c_out, kernel_size=(1, 1), strides=(2, 2), kernel_init=resnet_kernel_init)(x)

        x_out = self.act_fn(z + x)
        return x_out
    
class PreActResNetBlock(ResNetBlock):

    @nn.compact
    def __call__(self, x, train=True):
        # Network representing F
        z = nn.BatchNorm()(x, use_running_average=not train)
        z = self.act_fn(z)
        z = nn.Conv(self.c_out, kernel_size=(3, 3),
                    strides=(1, 1) if not self.subsample else (2, 2),
                    kernel_init=resnet_kernel_init,
                    use_bias=False)(z)
        z = nn.BatchNorm()(z, use_running_average=not train)
        z = self.act_fn(z)
        z = nn.Conv(self.c_out, kernel_size=(3, 3),
                    kernel_init=resnet_kernel_init,
                    use_bias=False)(z)

        if self.subsample:
            x = nn.BatchNorm()(x, use_running_average=not train)
            x = self.act_fn(x)
            x = nn.Conv(self.c_out,
                        kernel_size=(1, 1),
                        strides=(2, 2),
                        kernel_init=resnet_kernel_init,
                        use_bias=False)(x)

        x_out = z + x
        return x_out

class ResNet(nn.Module):
    num_classes : int
    act_fn : callable
    block_class : nn.Module
    num_blocks : tuple = (3, 3, 3)
    c_hidden : tuple = (16, 32, 64)

    @nn.compact
    def __call__(self, x, train=True):
        # A first convolution on the original image to scale up the channel size
        x = nn.Conv(self.c_hidden[0], kernel_size=(3, 3), kernel_init=resnet_kernel_init, use_bias=False)(x)
        if self.block_class == ResNetBlock:  # If pre-activation block, we do not apply non-linearities yet
            x = nn.BatchNorm()(x, use_running_average=not train)
            x = self.act_fn(x)

        # Creating the ResNet blocks
        for block_idx, block_count in enumerate(self.num_blocks):
            for bc in range(block_count):
                # Subsample the first block of each group, except the very first one.
                subsample = (bc == 0 and block_idx > 0)
                # ResNet block
                x = self.block_class(c_out=self.c_hidden[block_idx],
                                     act_fn=self.act_fn,
                                     subsample=subsample)(x, train=train)

        # Mapping to classification output
        x = x.mean(axis=(1, 2))
        x = nn.Dense(self.num_classes)(x)
        return x

class Trainer(TrainableModel):
  def __init__(self, config):
    super(Trainer, self).__init__(config)
    self.rng = jax.random.key(config.seed)
    self.model=ResNet(num_classes=10, act_fn=nn.relu, block_class=ResNetBlock)
    self.state = None
    self.buffer = fbx.make_item_buffer(**config.buffer_args)
    self.create_fn()

  def create_train_state(self, batch):
    """Creates initial `TrainState`."""
    self.rng, init_rng = jax.random.split(self.rng)
    imgs, labels = batch
    variables = self.model.init(init_rng, imgs, train=True)
    params = variables['params']

    num_steps_per_epoch = int(50000/self.config.batch_size)
    lr_schedule = optax.piecewise_constant_schedule(
      init_value=self.config.learning_rate,
      boundaries_and_scales=
        {int(num_steps_per_epoch*self.config.num_epochs*0.6): 0.1,
        int(num_steps_per_epoch*self.config.num_epochs*0.85): 0.1}
    )
    tx = optax.chain(optax.clip(1.0), optax.add_decayed_weights(1e-4), optax.sgd(lr_schedule, self.config.momentum))
    state = SDTrainState.create(apply_fn=self.model.apply, params=params, tx=tx, batch_stats = variables['batch_stats'])
    loss, _ = self.eval_step(state, batch)
    buffer_state = self.buffer.init(loss[0])
    buffer_state = self.buffer.add(buffer_state, loss[1:])
    state = state.replace(buffer_state=buffer_state)
    self.state = state

  def create_fn(self):

    def calc_batch_loss(params, batch_stats, batch, train=True):
      imgs, labels = batch
      outs = self.model.apply({'params': params, 'batch_stats': batch_stats}, imgs, train, mutable=['batch_stats'] if train else False)
      logits, new_state = outs if train else (outs, None)
      one_hot = jax.nn.one_hot(labels, 10)
      batch_loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot)
      acc = jnp.mean(jnp.argmax(logits, -1) == labels)
      metrics = {'ce_loss': jnp.mean(batch_loss), 'accuracy': acc, 'batch_loss': batch_loss}
      if train:
        return batch_loss, (metrics, new_state)
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
        batch_loss, (metrics, new_state) = calc_batch_loss(params, state.batch_stats, batch, True)
        batch_ref = None
        if self.config.loss == 'sd_2nd_cdf':
          batch_ref = self.buffer.sample(state.buffer_state, rng)['experience']

        final_loss = calc_final_loss(batch_loss, batch_ref)
        return final_loss, (metrics, new_state)

      outs, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
      final_loss, (metrics, new_state) = outs
      state = state.apply_gradients(grads=grads, batch_stats=new_state['batch_stats'])
      new_buffer_state = self.buffer.add(state.buffer_state, metrics['batch_loss'])
      state = state.replace(buffer_state=new_buffer_state)
      return state, metrics
    
    def eval_step(state, batch):
      batch_loss, metrics = calc_batch_loss(state.params, state.batch_stats, batch, False)
      return batch_loss, metrics
    
    # self.calc_batch_loss = jax.jit(calc_batch_loss)
    self.train_step = jax.jit(train_step)
    self.eval_step = jax.jit(eval_step)

  @staticmethod
  def format_log(epoch, train_metrics, test_metrics=None):
    if test_metrics is not None:
      return 'epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f, test_accuracy: %.2f' \
      % (
      epoch,
      train_metrics['ce_loss'],
      train_metrics['accuracy'] * 100,
      test_metrics['ce_loss'],
      test_metrics['accuracy'] * 100,
      )
    else:
       return 'epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f' \
      % (
      epoch,
      train_metrics['ce_loss'],
      train_metrics['accuracy'] * 100,
      )

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX local devices: %r', jax.local_devices())

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
