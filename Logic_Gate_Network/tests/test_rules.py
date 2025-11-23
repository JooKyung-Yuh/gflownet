import pytest # Python 테스트 프레임워크
from rules import Rule1_NoConsecutive1s
from rules import RuleChecker

def test_rule1_valid_cases():
  rule1 = Rule1_NoConsecutive1s()
  sequence1 = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
  sequence2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  sequence3 = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  sequence4 = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
  assert rule1.is_valid(sequence1)
  assert rule1.is_valid(sequence2)
  assert rule1.is_valid(sequence3)
  assert rule1.is_valid(sequence4)
  
def test_rule1_invalid_cases():
  rule1 = Rule1_NoConsecutive1s()
  sequence1 = [0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  sequence2 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  sequence3 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  sequence4 = [0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  
  assert rule1.is_valid(sequence1) == False
  assert rule1.is_valid(sequence2) == False
  assert rule1.is_valid(sequence3) == False
  assert rule1.is_valid(sequence4) == False

def test_rule1_metadata():
  rule1 = Rule1_NoConsecutive1s()
  assert rule1.get_dimension() == 10
  assert rule1.get_name() == "No Consecutive 1s"
  assert rule1.get_description() == "Valid if no two adjacent positions are both 1"
  
def test_rulechecker_add_rule():
  rulechecker = RuleChecker()
  rule1 = Rule1_NoConsecutive1s()
  rulechecker.add_rule(rule1)
  
  assert len(rulechecker.get_rules()) == 1
  
def test_rulechecker_validate_all_pass():
  rulechecker = RuleChecker()
  rule1 = Rule1_NoConsecutive1s()
  
  rulechecker.add_rule(rule1)
  sequence = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
  
  assert rulechecker.validate_all(sequence) == True
  
def test_rulechecker_validate_all_fail():
  rulechecker = RuleChecker()
  rule1 = Rule1_NoConsecutive1s()
  
  rulechecker.add_rule(rule1)
  sequence = [0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  
  assert rulechecker.validate_all(sequence) == False
  