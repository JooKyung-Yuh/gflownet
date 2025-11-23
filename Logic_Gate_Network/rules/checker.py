

class RuleChecker :
  """
  RuleChecker: Manages and validates multiple rules.

  This class stores multiple Rule objects and provides functionality to:
  - Add new rules to the collection
  - Check if a sequence satisfies all rules
  - Get information about all registered rules
  """
  def __init__(self) -> None:
    """Initialize RuleChecker with an empty rules list."""
    self.rules = []
    
  def add_rule(self, rule) -> None:
    """Add a rule to the checker."""
    self.rules.append(rule)
    
  def validate_all(self, sequence) -> bool:
    """
    Check if a sequence satisfies all registered rules.

    Args:
        sequence: The sequence to validate.

    Returns:
        True if the sequence passes all rules, False otherwise.
    """
    for rule in self.rules :
      if(rule.is_valid(sequence) == False) :
        return False
    
    return True
  
  def get_rules(self) -> list:
    """Get the list of all registered rules."""
    return self.rules

