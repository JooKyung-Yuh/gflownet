import random

class LGNDataset:
  """
  LGNDataset: Manages Logic Gate Network datasets with train/test split functionality.

  This class handles Real and Fake samples, providing methods to split data
  into training and testing sets.
  """
  def __init__(
    self,
    real_samples:list[list[int]],
    fake_samples:list[list[int]]
    ) -> None:
    """
    Initialize LGNDataset with Real and Fake samples.

    Args:
        real_samples: List of Real (valid) samples.
        fake_samples: List of Fake (invalid) samples.
    """
    self.real_samples:list[list[int]] = real_samples
    self.fake_samples:list[list[int]] = fake_samples
    self.train_real:list[list[int]] | None = None
    self.train_fake:list[list[int]] | None = None
    self.test_real:list[list[int]] | None = None
    self.test_fake:list[list[int]] | None = None
  
  def split_train_test(
    self,
    test_ratio: float = 0.2,
    random_seed: int | None = None) -> None:
    """
    Split Real and Fake samples into training and testing sets.

    Args:
        test_ratio: Proportion of data to use for testing (default: 0.2).
        random_seed: Random seed for reproducibility (optional).

    Returns:
        None. Stores train/test splits internally.
    """
    if random_seed is not None:
      random.seed(random_seed)
      
    copy_real_samples = self.real_samples.copy()
    copy_fake_samples = self.fake_samples.copy()
    
    random.shuffle(copy_real_samples)
    random.shuffle(copy_fake_samples)
    
    real_split_point = int(len(copy_real_samples) * (1 - test_ratio))
    train_real = copy_real_samples[:real_split_point]
    test_real = copy_real_samples[real_split_point:]
    
    fake_split_point = int(len(copy_fake_samples) * (1 - test_ratio))
    train_fake = copy_fake_samples[:fake_split_point]
    test_fake = copy_fake_samples[fake_split_point:]
    
    self.train_real = train_real
    self.test_real = test_real
    
    self.train_fake = train_fake
    self.test_fake = test_fake
    
  
  def get_train(self) -> tuple[list[list[int]] | None, list[list[int]] | None]:
    """
    Get training dataset.

    Returns:
        tuple: (train_real_samples, train_fake_samples)
    """
    return (self.train_real, self.train_fake)
  
  def get_test(self) -> tuple[list[list[int]] | None, list[list[int]] | None]:
    """
    Get testing dataset.

    Returns:
        tuple: (test_real_samples, test_fake_samples)
    """
    return (self.test_real, self.test_fake)