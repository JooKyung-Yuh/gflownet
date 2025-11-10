import abc

class BaseRule(abc.ABC):
  @abc.abstractmethod
  def is_valid(self, sequence):
    """
    Check if the given sequence is valid according to the rule.
    """
    pass