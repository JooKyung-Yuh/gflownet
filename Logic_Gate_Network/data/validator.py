from rules.base import BaseRule


class DataValidator:
  """
  DataValidator: Validates the quality of generated Real and Fake datasets.

  This class provides methods to check data integrity including:
  - Duplicate detection within datasets
  - Separation between Real and Fake samples
  - Rule compliance verification
  """
  def check_no_duplicates_within(
    self,
    samples:list[list[int]]
    ) -> bool:
    """
    Check if there are any duplicate samples within a single dataset.

    Args:
        samples: List of samples to check for duplicates.

    Returns:
        bool: True if no duplicates found, False otherwise.
    """
    set_samples = []
    for sample in samples:
      tuple_sample = tuple(sample)
      set_samples.append(tuple_sample)
    
    if len(set(set_samples)) == len(samples):
      return True
    return False  
    
  
  def check_real_fake_separation(
    self,
    real_samples:list[list[int]],
    fake_samples:list[list[int]]
  ) -> bool:
    """
    Check if Real and Fake datasets have no overlapping samples.

    Args:
        real_samples: List of Real samples.
        fake_samples: List of Fake samples.

    Returns:
        bool: True if no overlap exists, False otherwise.
    """
    set_real_samples = []
    set_fake_samples = []
    intersection = []
    for sample in real_samples:
      set_real_samples.append(tuple(sample))
    for sample in fake_samples:
      set_fake_samples.append(tuple(sample))
    
    set_real_samples = set(set_real_samples)
    set_fake_samples = set(set_fake_samples)
    
    intersection = set_real_samples.intersection(set_fake_samples)
    
    if len(intersection) == 0:
      return True
    return False
  
  def check_rule_compliance(
    self,
    rule:BaseRule,
    samples:list[list[int]],
    expected_validity:bool
    ) -> bool:
    """
    Check if all samples comply with the expected validity status.

    Args:
        rule: A BaseRule instance to validate samples against.
        samples: List of samples to check.
        expected_validity: Expected validation result (True for valid, False for invalid).

    Returns:
        bool: True if all samples match expected_validity, False otherwise.
    """
    
    for sample in samples:
      if rule.is_valid(sample) != expected_validity:
        return False
    
    return True
  
  def validate_all(
    self,
    rule:BaseRule,
    real_samples:list[list[int]],
    fake_samples:list[list[int]]
    ) -> bool:
    """
    Run all validation checks on Real and Fake datasets.

    This orchestrator method executes:
    1. Duplicate check within Real samples
    2. Duplicate check within Fake samples
    3. Separation check between Real and Fake
    4. Rule compliance for Real samples (should be valid)
    5. Rule compliance for Fake samples (should be invalid)

    Args:
        rule: A BaseRule instance for validation.
        real_samples: List of Real samples.
        fake_samples: List of Fake samples.

    Returns:
        bool: True if all checks pass, False otherwise.
    """
    result = self.check_no_duplicates_within(real_samples) and self.check_no_duplicates_within(fake_samples) and self.check_real_fake_separation(real_samples, fake_samples) and self.check_rule_compliance(rule, real_samples, True) and self.check_rule_compliance(rule, fake_samples, False)
    
    
    return result