# rules 폴더를 Python 패키지로 만들고, 
# BaseRule과 Rules을 외부에서 import할 수 있게 하기 작업
from .base import BaseRule
from .rule_1 import Rule1_NoConsecutive1s

__all__ = [
  "BaseRule", 
  "Rule1_NoConsecutive1s",
  ]