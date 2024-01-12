# ----------------------------------------------------------- #
#  This is code confidential, for peer-review purposes only   #
#  and protected under conference code of ethics              #
# ----------------------------------------------------------- #

# Code adapted from https://github.com/facebookresearch/xR-EgoPose authored by Denis Tome

"""
Base Transform class

Adapted from original

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