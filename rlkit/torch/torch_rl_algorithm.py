import abc
from collections import OrderedDict

from typing import Iterable
from torch import nn as nn

from rlkit.core.batch_rl_algorithm import BatchRLAlgorithm
from rlkit.core.online_rl_algorithm import OnlineRLAlgorithm
from rlkit.core.trainer import Trainer
from rlkit.torch.core import np_to_pytorch_batch


class TorchOnlineRLAlgorithm(OnlineRLAlgorithm):
    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)


class TorchBatchRLAlgorithm(BatchRLAlgorithm):
    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)


class TorchTrainer(Trainer, metaclass=abc.ABCMeta):
    def __init__(self):
        self._num_train_steps = 0

    def train(self, np_batch, np_batch_gen=None, slac_representation = False, rad_aug = False):
        self._num_train_steps += 1
        if slac_representation or rad_aug:
            batch = np_batch # assume already tensor            
            if np_batch_gen is not None:
                batch_gen = np_batch_gen
            else:
                batch_gen=None
        else:
            batch = np_to_pytorch_batch(np_batch)
            if np_batch_gen is not None:                
                batch_gen = np_to_pytorch_batch(np_batch_gen)
            else:
                batch_gen = None

        self.train_from_torch(batch, batch_gen=batch_gen)

    def get_diagnostics(self):
        return OrderedDict([
            ('num train calls', self._num_train_steps),
        ])

    @abc.abstractmethod
    def train_from_torch(self, batch):
        pass

    @property
    @abc.abstractmethod
    def networks(self) -> Iterable[nn.Module]:
        pass
