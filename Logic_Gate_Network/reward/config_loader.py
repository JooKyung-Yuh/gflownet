"""
Configuration Loader for Reward Function.

This module provides utilities for loading and validating YAML configuration files
that contain hyperparameters for the Reward Function and related components.

Key Features:
-------------
1. Load YAML config files into Python dictionaries
2. Validate configuration values (type checking, range checking)
3. Provide helpful error messages for invalid configurations
4. Support for default values and optional parameters

Usage:
------
>>> from Logic_Gate_Network.reward.config_loader import load_reward_config
>>>
>>> # Load configuration from YAML file
>>> config = load_reward_config("configs/reward_config.yaml")
>>>
>>> # Access values using dictionary keys
>>> print(config['reward_function']['C'])  # 1.0
>>> print(config['lgn_network']['num_inputs'])  # 10
>>>
>>> # Initialize RewardFunction with loaded config
>>> reward_fn = RewardFunction(
...     C=config['reward_function']['C'],
...     epsilon=config['reward_function']['epsilon'],
...     lambda_complexity=config['reward_function']['lambda_complexity']
... )
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_reward_config(config_path: str, validate: bool = True) -> Dict[str, Any]:
    """
    Load reward function configuration from a YAML file.

    This function reads a YAML configuration file, parses it into a Python dictionary,
    and optionally validates the values to ensure they are within acceptable ranges.

    Args:
        config_path (str):
            Path to the YAML configuration file (absolute or relative).
            Examples: "configs/reward_config.yaml", "./my_config.yaml"

        validate (bool, optional):
            Whether to validate the loaded configuration values.
            If True, checks that all required fields exist and values are valid.
            If False, returns the raw dictionary without validation.
            Default: True (recommended for safety)

    Returns:
        Dict[str, Any]: Configuration dictionary with nested structure:
            {
                'experiment': {...},          # Metadata
                'reward_function': {...},     # RewardFunction hyperparameters
                'lgn_network': {...},         # LGNState configuration
                'data': {...},                # Optional data info
                'presets': {...}              # Optional preset configs
            }

    Raises:
        FileNotFoundError:
            If the config file doesn't exist at the specified path.

        yaml.YAMLError:
            If the YAML file is malformed (syntax errors, invalid structure).

        ValueError:
            If validate=True and configuration values are invalid:
            - Missing required fields
            - Values outside acceptable ranges
            - Invalid data types

    Example:
        >>> # Load and use configuration
        >>> config = load_reward_config("configs/reward_config.yaml")
        >>>
        >>> # Access nested values
        >>> C = config['reward_function']['C']
        >>> epsilon = config['reward_function']['epsilon']
        >>> lambda_complexity = config['reward_function']['lambda_complexity']
        >>>
        >>> # Initialize components
        >>> from Logic_Gate_Network.reward.reward_fn import RewardFunction
        >>> from Logic_Gate_Network.lgn import LGNState
        >>>
        >>> reward_fn = RewardFunction(C=C, epsilon=epsilon, lambda_complexity=lambda_complexity)
        >>> lgn = LGNState(
        ...     num_inputs=config['lgn_network']['num_inputs'],
        ...     max_gates=config['lgn_network']['max_gates']
        ... )

    Notes:
        - YAML file must follow the structure defined in configs/reward_config.yaml
        - Validation checks are designed to catch common mistakes early
        - If validation fails, the error message will indicate the problematic field
        - Use validate=False only if you're confident the config is correct

    See Also:
        - validate_reward_config(): Standalone validation function
        - configs/reward_config.yaml: Example configuration file with documentation
    """
    # Convert to Path object for better path handling
    config_file = Path(config_path)

    # Check if file exists
    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Absolute path: {config_file.absolute()}\n"
            f"Please ensure the file exists and the path is correct."
        )

    # Load YAML file
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(
            f"Failed to parse YAML configuration file: {config_path}\n"
            f"Error details: {str(e)}\n"
            f"Please check the YAML syntax (indentation, special characters, etc.)"
        )

    # Validate configuration if requested
    if validate:
        validate_reward_config(config, config_path=config_path)

    return config


def validate_reward_config(config: Dict[str, Any], config_path: Optional[str] = None) -> None:
    """
    Validate reward function configuration values.

    This function checks that all required fields exist and that values are within
    acceptable ranges. It provides detailed error messages to help debug configuration issues.

    Args:
        config (Dict[str, Any]):
            Configuration dictionary to validate (from load_reward_config or manual creation).

        config_path (str, optional):
            Path to config file (used in error messages for better debugging).
            If None, generic error messages are used.

    Raises:
        ValueError:
            If any validation check fails:
            - Missing required top-level sections
            - Missing required fields within sections
            - Values outside acceptable ranges
            - Invalid data types

    Validation Checks:
    ------------------
    1. **Required Sections:**
       - 'reward_function': Must exist with C, epsilon, lambda_complexity
       - 'lgn_network': Must exist with num_inputs, max_gates

    2. **reward_function.C:**
       - Type: float or int
       - Range: > 0 (must be positive)
       - Typical: [0.1, 10.0]

    3. **reward_function.epsilon:**
       - Type: float
       - Range: > 0 (must be positive)
       - Typical: [1e-10, 1e-6]
       - Warning if > 1e-3 (too large, may affect reward)

    4. **reward_function.lambda_complexity:**
       - Type: float or int
       - Range: >= 0 (non-negative)
       - Typical: [0.0, 1.0]

    5. **lgn_network.num_inputs:**
       - Type: int
       - Range: >= 1 (at least one input)
       - Typical: [5, 100]

    6. **lgn_network.max_gates:**
       - Type: int
       - Range: >= 1 (at least one gate allowed)
       - Typical: [5, 50]

    Example:
        >>> config = {
        ...     'reward_function': {'C': 1.0, 'epsilon': 1e-6, 'lambda_complexity': 0.1},
        ...     'lgn_network': {'num_inputs': 10, 'max_gates': 15}
        ... }
        >>> validate_reward_config(config)  # Passes silently
        >>>
        >>> bad_config = {'reward_function': {'C': -1.0}}  # Missing fields, negative C
        >>> validate_reward_config(bad_config)  # Raises ValueError

    Notes:
        - Validation is strict to catch errors early
        - All checks are based on mathematical/logical constraints
        - Typical ranges are suggestions, not hard constraints
        - This function does NOT modify the config, only validates it
    """
    path_info = f" (from {config_path})" if config_path else ""

    # -------------------------------------------------------------------------
    # 1. Check required top-level sections exist
    # -------------------------------------------------------------------------
    required_sections = ['reward_function', 'lgn_network']
    for section in required_sections:
        if section not in config:
            raise ValueError(
                f"Missing required section '{section}' in configuration{path_info}.\n"
                f"Expected sections: {required_sections}\n"
                f"Found sections: {list(config.keys())}"
            )

    # -------------------------------------------------------------------------
    # 2. Validate reward_function parameters
    # -------------------------------------------------------------------------
    reward_fn_config = config['reward_function']

    # Required fields in reward_function
    required_reward_fields = ['C', 'epsilon', 'lambda_complexity']
    for field in required_reward_fields:
        if field not in reward_fn_config:
            raise ValueError(
                f"Missing required field '{field}' in reward_function section{path_info}.\n"
                f"Expected fields: {required_reward_fields}\n"
                f"Found fields: {list(reward_fn_config.keys())}"
            )

    # Validate C (Real/Fake balance parameter)
    C = reward_fn_config['C']
    if not isinstance(C, (int, float)):
        raise ValueError(
            f"reward_function.C must be a number (int or float){path_info}.\n"
            f"Got type: {type(C).__name__}, value: {C}"
        )
    if C <= 0:
        raise ValueError(
            f"reward_function.C must be positive (> 0){path_info}.\n"
            f"Got: {C}\n"
            f"Typical range: [0.1, 10.0]"
        )

    # Validate epsilon (numerical stability constant)
    epsilon = reward_fn_config['epsilon']
    if not isinstance(epsilon, float):
        raise ValueError(
            f"reward_function.epsilon must be a float{path_info}.\n"
            f"Got type: {type(epsilon).__name__}, value: {epsilon}\n"
            f"Use scientific notation like 1.0e-6 or 0.000001"
        )
    if epsilon <= 0:
        raise ValueError(
            f"reward_function.epsilon must be positive (> 0){path_info}.\n"
            f"Got: {epsilon}\n"
            f"Typical range: [1e-10, 1e-6]"
        )
    if epsilon > 1e-3:
        # Warning: epsilon is unusually large (but not invalid)
        import warnings
        warnings.warn(
            f"reward_function.epsilon is unusually large: {epsilon}{path_info}.\n"
            f"This may significantly affect the fake data term in the reward.\n"
            f"Typical range: [1e-10, 1e-6]\n"
            f"Consider using a smaller value unless this is intentional."
        )

    # Validate lambda_complexity (complexity penalty weight)
    lambda_complexity = reward_fn_config['lambda_complexity']
    if not isinstance(lambda_complexity, (int, float)):
        raise ValueError(
            f"reward_function.lambda_complexity must be a number (int or float){path_info}.\n"
            f"Got type: {type(lambda_complexity).__name__}, value: {lambda_complexity}"
        )
    if lambda_complexity < 0:
        raise ValueError(
            f"reward_function.lambda_complexity must be non-negative (>= 0){path_info}.\n"
            f"Got: {lambda_complexity}\n"
            f"Typical range: [0.0, 1.0]"
        )

    # -------------------------------------------------------------------------
    # 3. Validate lgn_network parameters
    # -------------------------------------------------------------------------
    lgn_config = config['lgn_network']

    # Required fields in lgn_network
    required_lgn_fields = ['num_inputs', 'max_gates']
    for field in required_lgn_fields:
        if field not in lgn_config:
            raise ValueError(
                f"Missing required field '{field}' in lgn_network section{path_info}.\n"
                f"Expected fields: {required_lgn_fields}\n"
                f"Found fields: {list(lgn_config.keys())}"
            )

    # Validate num_inputs
    num_inputs = lgn_config['num_inputs']
    if not isinstance(num_inputs, int):
        raise ValueError(
            f"lgn_network.num_inputs must be an integer{path_info}.\n"
            f"Got type: {type(num_inputs).__name__}, value: {num_inputs}"
        )
    if num_inputs < 1:
        raise ValueError(
            f"lgn_network.num_inputs must be >= 1{path_info}.\n"
            f"Got: {num_inputs}\n"
            f"Network must have at least one input feature."
        )

    # Validate max_gates
    max_gates = lgn_config['max_gates']
    if not isinstance(max_gates, int):
        raise ValueError(
            f"lgn_network.max_gates must be an integer{path_info}.\n"
            f"Got type: {type(max_gates).__name__}, value: {max_gates}"
        )
    if max_gates < 1:
        raise ValueError(
            f"lgn_network.max_gates must be >= 1{path_info}.\n"
            f"Got: {max_gates}\n"
            f"Network must allow at least one gate."
        )

    # All validation checks passed!
    # (No return value - function succeeds silently if config is valid)


def get_preset_config(preset_name: str, base_config_path: str = "configs/reward_config.yaml") -> Dict[str, Any]:
    """
    Load a preset configuration from the base config file.

    This is a convenience function to quickly load predefined configurations
    (like 'baseline', 'high_accuracy', 'high_simplicity', etc.) from the
    presets section of a config file.

    Args:
        preset_name (str):
            Name of the preset to load.
            Must match a key in the 'presets' section of the config file.
            Examples: 'baseline', 'high_accuracy', 'high_simplicity', 'fast_prototype'

        base_config_path (str, optional):
            Path to the config file containing presets.
            Default: "configs/reward_config.yaml"

    Returns:
        Dict[str, Any]: Preset configuration values.
            Typically contains: C, epsilon, lambda_complexity, num_inputs, max_gates

    Raises:
        ValueError:
            If preset_name doesn't exist in the config file.

        FileNotFoundError:
            If base_config_path doesn't exist.

    Example:
        >>> # Load 'high_accuracy' preset
        >>> preset = get_preset_config('high_accuracy')
        >>> print(preset)
        >>> # {'C': 2.0, 'epsilon': 1e-10, 'lambda_complexity': 0.01, ...}
        >>>
        >>> # Use preset values
        >>> reward_fn = RewardFunction(
        ...     C=preset['C'],
        ...     epsilon=preset['epsilon'],
        ...     lambda_complexity=preset['lambda_complexity']
        ... )

    Notes:
        - Presets are defined in the 'presets' section of the config file
        - Presets provide quick starting points for different use cases
        - You can modify preset values after loading if needed
    """
    # Load base config
    config = load_reward_config(base_config_path, validate=False)

    # Check if presets section exists
    if 'presets' not in config:
        raise ValueError(
            f"No 'presets' section found in config file: {base_config_path}\n"
            f"Available sections: {list(config.keys())}"
        )

    presets = config['presets']

    # Check if requested preset exists
    if preset_name not in presets:
        raise ValueError(
            f"Preset '{preset_name}' not found in config file: {base_config_path}\n"
            f"Available presets: {list(presets.keys())}"
        )

    return presets[preset_name]
