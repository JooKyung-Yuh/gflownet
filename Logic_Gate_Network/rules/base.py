from abc import ABC, abstractmethod

class BaseRule(ABC): # 추상 클래스, 이후에 class Rule1(BaseRule)처럼 상속받아 사용  
  @abstractmethod # 추상 메서드, 하위 클래스에서 반드시 구현해야 함
  def is_valid(self, sequence) -> bool :
    """
    Check if the given sequence is valid according to the rule.
    메서드 이름: is_valid
    파라미터: self, sequence (시퀀스를 받아서 검사합니다)
    기능: 주어진 sequence가 규칙을 따르는지 확인
    반환: True 또는 False
    내용: pass (추상 메서드는 구현 없이 pass만 씁니다)
    """
    pass

  @abstractmethod
  def get_dimension(self) -> int:
    """
    Get the dimension of the rule.
    메서드 이름: get_dimension
    파라미터: self만 받습니다 (sequence는 필요 없음)
    기능: 입력 데이터의 차원을 반환 (Rule 1의 경우 10)
    반환: 정수 (integer)
    내용: pass
    """
    pass
  
  @abstractmethod
  def get_name(self) -> str:
    """
    Get the name of the rule.
    메서드 이름: get_name
    파라미터: self만 받습니다 (sequence는 필요 없음)
    기능: 규칙의 이름을 반환 (예: "Rule 1")
    반환: 문자열 (string)
    내용: pass
    """
    pass
  
  @abstractmethod
  def get_description(self) -> str:
    """
    Get the description of the rule.
    메서드 이름: get_description
    파라미터: self만 받습니다 (sequence는 필요 없음)
    기능: 규칙에 대한 설명을 반환 (예: "Valid if no two adjacent positions are both 1")
    반환: 문자열 (string)
    내용: pass
    """
    pass