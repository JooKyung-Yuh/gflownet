from .gates import apply_gate
from .network import LGNState


class LGNEvaluator:
  """
  Evaluator for Logic Gate Networks (LGN).
  
  This class performs forward pass evaluation of LGN states on binary inputs,
  supporting hybrid gate architecture (variadic AND, binary other gates).
  """
  def evaluate(self, lgn_state:LGNState, binary_input:list[int]) -> int:
    """
    Evaluate Logic Gate Network on a single binary input via forward pass.

    This method executes all gates in the network sequentially, computing outputs
    by applying gate operations on inputs and previous gate outputs.

    Args:
        lgn_state (LGNState): The Logic Gate Network structure to evaluate.
        binary_input (list[int]): Binary input vector (0s and 1s).
            Length must match lgn_state.num_inputs.

    Returns:
        int: Final output (0 or 1) from the last gate in the network.

    Raises:
        AssertionError: If binary_input length doesn't match num_inputs.

    Example:
        >>> lgn = LGNState(num_inputs=10, max_gates=15)
        >>> lgn.add_gate(GateType.AND, [0, 1, 2])
        >>> lgn.add_gate(GateType.OR, [3, 10])
        >>> evaluator = LGNEvaluator()
        >>> evaluator.evaluate(lgn, [1,1,1,0,1,0,1,0,1,0])
        1
    """
    assert len(binary_input) == lgn_state.num_inputs  # Validate input dimension matches network configuration
    
    outputs = binary_input.copy() # Initialize outputs with inputs (indices 0~num_inputs-1) This list will grow as gates are executed (indices num_inputs~)
    
    for gate in lgn_state.gates:  # Execute each gate sequentially in the order they were added
      input_values = [outputs[idx] for idx in gate.inputs]  # Gather input values from outputs list using gate's input indices
      result = apply_gate(gate.gate_type, input_values)   # Execute gate operation (handles hybrid arity: variadic AND, binary others)
      outputs.append(result)  # Append gate output to outputs list (becomes available for subsequent gates)
    
    return outputs[-1]  # Return final gate's output as network result
  
  def evaluate_batch(self, lgn_state:LGNState, binary_inputs:list[list[int]]) -> list[int]:
    """
    Evaluate Logic Gate Network on multiple binary inputs (batch processing).

    This method applies the same LGN to multiple inputs independently,
    collecting all results for efficient evaluation of datasets.

    Args:
        lgn_state (LGNState): The Logic Gate Network structure to evaluate.
        binary_inputs (list[list[int]]): List of binary input vectors.
            Each input must have length matching lgn_state.num_inputs.

    Returns:
        list[int]: List of outputs (0 or 1) for each input, in the same order.

    Example:
        >>> lgn = LGNState(num_inputs=10, max_gates=15)
        >>> lgn.add_gate(GateType.AND, [0, 1, 2])
        >>> lgn.add_gate(GateType.OR, [3, 10])
        >>> evaluator = LGNEvaluator()
        >>> inputs = [[1,1,1,0,1,0,1,0,1,0], [0,0,0,1,0,1,0,1,0,1]]
        >>> evaluator.evaluate_batch(lgn, inputs)
        [1, 1]
    """
    output = [self.evaluate(lgn_state, inp) for inp in binary_inputs] # Evaluate each input independently and collect results
    return output

  def evaluate_final(self, lgn_state: LGNState, binary_input: list[int]) -> int:
    """
    Evaluate LGN using AND combination of all root gates' outputs.

    Unlike evaluate() which returns only the last gate's output, this method
    finds all root gates (gates whose outputs are not used by other gates)
    and returns the AND of their outputs.

    This ensures all sub-LGNs contribute to the final output, preventing
    disconnected subgraphs from being ignored in reward calculation.

    Args:
        lgn_state (LGNState): The Logic Gate Network structure to evaluate.
        binary_input (list[int]): Binary input vector (0s and 1s).

    Returns:
        int: 1 if ALL root gates output 1, 0 otherwise.
             Returns 1 if no gates exist (empty network).

    Example:
        >>> lgn = LGNState(num_inputs=5, max_gates=10)
        >>> lgn.add_gate(GateType.AND, [0, 1])    # Gate 0, root
        >>> lgn.add_gate(GateType.OR, [2, 3])     # Gate 1, root (disconnected)
        >>> evaluator = LGNEvaluator()
        >>> evaluator.evaluate(lgn, [1, 1, 0, 0, 0])
        0  # Only Gate 1's output (OR of 0,0 = 0)
        >>> evaluator.evaluate_final(lgn, [1, 1, 0, 0, 0])
        0  # AND(Gate0=1, Gate1=0) = 0
    """
    if len(lgn_state.gates) == 0:
      return 1  # Empty network defaults to 1

    assert len(binary_input) == lgn_state.num_inputs

    # Compute all gate outputs
    outputs = binary_input.copy()
    for gate in lgn_state.gates:
      input_values = [outputs[idx] for idx in gate.inputs]
      result = apply_gate(gate.gate_type, input_values)
      outputs.append(result)

    # Get root gates and AND their outputs
    root_gates = lgn_state.get_root_gates()
    if len(root_gates) == 0:
      return outputs[-1]  # Fallback to last gate if no roots found

    root_outputs = [outputs[lgn_state.num_inputs + gate_idx] for gate_idx in root_gates]
    return int(all(root_outputs))  # AND combination

  def evaluate_final_batch(self, lgn_state: LGNState, binary_inputs: list[list[int]]) -> list[int]:
    """
    Batch version of evaluate_final().

    Args:
        lgn_state (LGNState): The Logic Gate Network structure to evaluate.
        binary_inputs (list[list[int]]): List of binary input vectors.

    Returns:
        list[int]: List of final outputs for each input.
    """
    return [self.evaluate_final(lgn_state, inp) for inp in binary_inputs]