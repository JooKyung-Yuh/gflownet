from .network import LGNState
from .gates import GateType, is_valid_arity

class ActionSpace:
  def __init__(self, lgn_state:LGNState) -> None:
    self.lgn_state = lgn_state

  def get_available_inputs(self) -> None:
    pass
  
  def get_valid_actions(self) -> None:
    pass
  
  def is_valid_action(self, action) -> None:
    pass