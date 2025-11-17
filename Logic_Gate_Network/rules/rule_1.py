from .base import BaseRule
import random
class Rule1_NoConsecutive1s(BaseRule):
    """
    Rule 1: No two consecutive 1s are allowed in the input sequence.
    """
    dimension = 25
    name = "No Consecutive 1s"
    description = "Valid if no two adjacent positions are both 1"
    
    def is_valid(self, sequence:list[int]):
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
    
    def generate_candidate(self) -> list[int]:
        """Generate a candidate sample using constraint satisfaction for Rule 1."""
        rule_dimension = self.get_dimension()
        sequence = []
        for i in range(rule_dimension):
            if i == 0:
                sequence.append(random.choice([0, 1]))
            else:
                if sequence[i-1] == 1:
                    sequence.append(0)
                else:
                    sequence.append(random.choice([0, 1]))
        
        return sequence
    
    def generate_violating_candidate(self) -> list[int]:
        """Generate violating candidate by forcing consecutive 1s for Rule 1"""
        rule_dimension = self.get_dimension()
        sequence = [random.randint(0, 1) for _ in range(rule_dimension)]
        
        # Force at least one violation (consecutive 1s)
        pos = random.randint(0, rule_dimension - 2)
        sequence[pos] = 1
        sequence[pos + 1] = 1
        
        return sequence