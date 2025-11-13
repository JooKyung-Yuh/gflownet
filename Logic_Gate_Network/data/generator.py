# 규칙을 만족하는 Real 데이터를 생성하는 시스템
import random

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