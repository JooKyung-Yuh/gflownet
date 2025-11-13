# 규칙을 만족하는 Real 데이터를 생성하는 시스템
import datetime
import csv
import json

class RealDataGenerator:
  """
  RealDataGenerator: Generates samples that satisfy given rules.

  This class creates binary sequences that follow the constraints
  defined by a rule object (e.g., Rule1_NoConsecutive1s).
  """
  def generate(self, rule, count=500) -> list[list[int]]:
    """
    Generate samples that satisfy the given rule.

    Args:
        rule: A BaseRule instance to validate samples.
        count: Number of samples to generate (default: 500).

    Returns:
        List of valid samples (each sample is a list of 0s and 1s).

    Raises:
        RuntimeError: If unable to generate enough samples.
    """
    samples = []
    max_attempts = count * 100 # 시도 횟수
    attempts = 0
    
    while len(samples) < count and attempts < max_attempts: # 샘플이 충분히 모이면 (len(samples) >= count) 루프 종료, 또는 시도 횟수 초과하면 (attempts >= max_attempts) 루프 종료
      attempts += 1
      
      sample = rule.generate_candidate()
      if rule.is_valid(sample):                              # valid 한지 확인
        if tuple(sample) not in {tuple(s) for s in samples}: # 중복방지
          samples.append(sample)
      
    if len(samples) < count:
      raise RuntimeError(f"Failed to generate {count} samples. Only generated {len(samples)} samples after {max_attempts} attempts.")
        
    return samples
  
  def save_to_csv(self, samples, filename=None) -> str:
    """
    Save samples to CSV file.

    Args:
        samples: List of samples to save.
        filename: Optional filename. If None, auto-generates with timestamp.

    Returns:
        str: Path to the saved CSV file.
    """
    if filename is None:
      filename = datetime.datetime.now().strftime("samples_%Y-%m-%d_%H-%M-%S.csv")
    
    with open(filename, 'w', newline='') as csvfile:
      writer = csv.writer(csvfile)
      
      header = [f'bit_{i}' for i in range(len(samples[0]))] # len(samples[0]): 첫 번째 샘플의 길이 (10개), List comprehension으로 ['bit_0', 'bit_1', ..., 'bit_9'] 생성
      writer.writerow(header) #CSV 첫 줄에 헤더 작성
      
      for sample in samples:
        writer.writerow(sample)
        
    return filename
  
  def save_to_json(self, samples, rule, filename=None) -> str:
    """
    Save samples to JSON file with metadata.

    Args:
        samples: List of samples to save.
        rule: Rule object for metadata extraction.
        filename: Optional filename. If None, auto-generates with timestamp.

    Returns:
        str: Path to the saved JSON file.
    """
    if filename is None:
      filename = datetime.datetime.now().strftime("samples_%Y-%m-%d_%H-%M-%S.json")
    
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
    
    return filename