# Code adapted from https://github.com/facebookresearch/xR-EgoPose

"""
Base Transform class

Adapted from original
@author: Denis Tome'

"""
from abc import ABC, abstractmethod
from utils import console_logger, ConsoleLogger


class BaseTransform(ABC):
    """BaseTrasnform class"""

    def __init__(self):
        super().__init__()
        self.logger = ConsoleLogger(self.__class__.__name__)

    @abstractmethod
    def __call__(self, data):
        """Perform transformation

        Arguments:
            data {dict} -- frame data
        """
        raise NotImplementedError