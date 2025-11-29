# Data generation system for Real/Fake samples
import datetime
import csv
import json
import os

from rules.base import BaseRule

class RealDataGenerator:
  """
  RealDataGenerator: Generates samples that satisfy given rules.

  This class creates binary sequences that follow the constraints
  defined by a rule object (e.g., Rule1_NoConsecutive1s).
  """
  def generate(self, rule: BaseRule, count: int = 10000) -> list[list[int]]:
    """
    Generate samples that satisfy the given rule.

    Args:
        rule: A BaseRule instance to validate samples.
        count: Number of samples to generate (default: 10000).

    Returns:
        List of valid samples (each sample is a list of 0s and 1s).

    Raises:
        RuntimeError: If unable to generate enough samples.
    """
    samples = []
    max_attempts = count * 100  # Maximum number of attempts
    attempts = 0
    
    while len(samples) < count and attempts < max_attempts:
      attempts += 1

      sample = rule.generate_candidate()
      if rule.is_valid(sample):
        if tuple(sample) not in {tuple(s) for s in samples}:  # Avoid duplicates
          samples.append(sample)
      
    if len(samples) < count:
      raise RuntimeError(f"Failed to generate {count} samples. Only generated {len(samples)} samples after {max_attempts} attempts.")
        
    return samples

  
  
class FakeDataGenerator: 
  """
  FakeDataGenerator: Generates samples that violate given rules.

  This class creates binary sequences that intentionally break the constraints
  defined by a rule object. Ensures no overlap with Real data samples.
  """
  def generate(self, rule: BaseRule, real_samples: list[list[int]], count: int = 10000) -> list[list[int]]:
    """
    Generate samples that violate the given rule.

    Args:
        rule: A BaseRule instance to validate samples.
        real_samples: List of Real samples (to avoid overlap).
        count: Number of samples to generate (default: 10000).

    Returns:
        List of invalid samples (each sample is a list of 0s and 1s).

    Raises:
        RuntimeError: If unable to generate enough samples.
    """
    samples = []  # Storage for fake samples
    real_set = {tuple(s) for s in real_samples}  # Convert to set for O(1) lookup
    max_attempts = count * 100  # Maximum number of attempts
    attempts = 0
    
    while len(samples) < count and attempts < max_attempts:
      sample = rule.generate_violating_candidate()
      if not rule.is_valid(sample):
        sample_tuple = tuple(sample)
        if sample_tuple not in real_set and sample_tuple not in {tuple(s) for s in samples}:
          samples.append(sample)

      attempts += 1
    
    if len(samples) < count:
      raise RuntimeError(f"Failed to generate {count} fake samples. Only generated {len(samples)} samples after {max_attempts} attempts.")
    return samples



def save_to_csv(samples:list[list[int]], label:str, filename=None) -> str:
  """
  Save samples to CSV file.

  Args:
      samples: List of samples to save.
      filename: Optional filename. If None, auto-generates with timestamp.

  Returns:
      str: Path to the saved CSV file.
  """
  output_dir = "data/csv"
  os.makedirs(output_dir, exist_ok=True)

  if filename is None:
    filename = os.path.join(output_dir, datetime.datetime.now().strftime(f"{label}_samples_%Y-%m-%d_%H-%M-%S.csv"))

  
  with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    header = [f'bit_{i}' for i in range(len(samples[0]))]  # Create header row
    writer.writerow(header)  # Write header to CSV
    
    for sample in samples:
      writer.writerow(sample)
      
  return os.path.abspath(filename)

def save_to_json(samples:list[list[int]], rule:BaseRule, label:str, filename=None) -> str:
  """
  Save samples to JSON file with metadata.

  Args:
      samples: List of samples to save.
      rule: Rule object for metadata extraction.
      filename: Optional filename. If None, auto-generates with timestamp.

  Returns:
      str: Path to the saved JSON file.
  """
  output_dir = "data/json"
  os.makedirs(output_dir, exist_ok=True)
  
  if filename is None:
    filename = os.path.join(output_dir, datetime.datetime.now().strftime(f"{label}_samples_%Y-%m-%d_%H-%M-%S.json"))
  
  data = {
    "metadata": {
      "rule_name": rule.get_name(),
      "rule_description": rule.get_description(),
      "dimension": rule.get_dimension(),
      "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
      "sample_count": len(samples)
    },
    "samples": samples
  }
  
  with open(filename, 'w') as jsonfile:
    json.dump(data, jsonfile, indent=2)
  
  return os.path.abspath(filename)