from abc import ABC, abstractmethod

class BaseRule(ABC):
  """
  Abstract base class for defining rules that validate binary sequences.
  All subclasses must implement the abstract methods defined here.
  """

  @abstractmethod
  def is_valid(self, sequence) -> bool :
    """
    Check if the given sequence is valid according to the rule.

    Args:
        sequence: Binary sequence to validate

    Returns:
        bool: True if valid, False otherwise
    """
    pass

  @abstractmethod
  def get_dimension(self) -> int:
    """
    Get the dimension (length) of input sequences for this rule.

    Returns:
        int: Dimension of the rule
    """
    pass

  @abstractmethod
  def get_name(self) -> str:
    """
    Get the name of the rule.

    Returns:
        str: Rule name
    """
    pass

  @abstractmethod
  def get_description(self) -> str:
    """
    Get a human-readable description of the rule.

    Returns:
        str: Rule description
    """
    pass
  
  @abstractmethod
  def generate_candidate(self) -> list[int]:
    """
    Generate a candidate sample that likely satisfies this rule.
    
    Returns:
        A binary sequence of length equal to rule dimension.
    """
    pass
  
  @abstractmethod
  def generate_violating_candidate(self) -> list[int]:
    """
    Generate a violating candidate sample that likely not satisfies this rule.
    
    Returns:
        A binary sequence of length equal to rule dimension.
    """
    pass