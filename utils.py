from typing import Any

from flax.training import train_state

class SDTrainState(train_state.TrainState):
    batch_stats: Any = None
    buffer_state: Any = None

class TrainableModel(object):
    def __init__(self, config):
        self.config = config