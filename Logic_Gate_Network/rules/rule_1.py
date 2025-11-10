from .base import BaseRule
class Rule1_NoConsecutive1s(BaseRule):
    """
    Rule 1: No two consecutive 1s are allowed in the input sequence.
    """
    dimension = 10
    name = "No Consecutive 1s"
    description = "Valid if no two adjacent positions are both 1"
    
    def is_valid(self, sequence):
      for i in range(len(sequence) - 1):
          if sequence[i] == 1 and sequence[i + 1] == 1:
              return False
      return True
    
    def get_dimension(self):
        return self.dimension
      
    def get_name(self):
        return self.name
      
    def get_description(self):
        return self.description