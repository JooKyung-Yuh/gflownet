# 여러 규칙들을 관리하는 RuleChecker 클래스
# RuleChecker는 여러 개의 Rule 객체들을 모아서 관리하는 클래스
# Rule 1, Rule 2, Rule 3... 여러 규칙들을 저장해두고
# 주어진 sequence가 모든 규칙을 만족하는지 한 번에 확인가능


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

