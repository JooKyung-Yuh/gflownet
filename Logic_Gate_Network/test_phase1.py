from .rules import Rule1_NoConsecutive1s
from .data import RealDataGenerator, FakeDataGenerator
from .data import DataValidator
from .data import LGNDataset
from .data import save_to_csv, save_to_json

def main():
  print("=== Phase 1 Integration Test ===")
  print("\n1. Initializing Rule 1...")
  rule1 = Rule1_NoConsecutive1s()
  print(f"Rule 1 initialized: {rule1.get_name()}")
  
  print("\n2. Generating Real data...")
  real_gen = RealDataGenerator()
  real_samples = real_gen.generate(rule1, count=500)
  print(f"Generated {len(real_samples)} Real samples")
  
  
  print("\n3. Generating Fake data...")
  fake_gen = FakeDataGenerator()
  fake_samples = fake_gen.generate(rule1, real_samples, count=500)
  print(f"Generated {len(fake_samples)} Fake samples")
  
  
  print("\n4. Validating data...")
  validator = DataValidator()
  validation_result = validator.validate_all(rule1, real_samples, fake_samples)
  if validation_result:
    print("✓ All validations passed!")
  else:
    print("✗ Validation failed!")
    
  
  print("\n5. Splitting data into train/test...")
  dataset = LGNDataset(real_samples, fake_samples)
  dataset.split_train_test(test_ratio=0.2, random_seed=42)
  train_real, train_fake = dataset.get_train()
  test_real, test_fake = dataset.get_test()
  
  assert train_real is not None and train_fake is not None
  assert test_real is not None and test_fake is not None
  
  print(f"Train: {len(train_real)} Real + {len(train_fake)} Fake")
  print(f"Test: {len(test_real)} Real + {len(test_fake)} Fake")
  
  
  print("\n6. Saving data to files...")
  real_csv_path = save_to_csv(real_samples)
  fake_csv_path = save_to_csv(fake_samples)
  real_json_path = save_to_json(real_samples, rule1)
  fake_json_path = save_to_json(fake_samples, rule1)
  
  print(f"✓ Real samples saved:")
  print(f"  - CSV: {real_csv_path}")
  print(f"  - JSON: {real_json_path}")
  print(f"✓ Fake samples saved:")
  print(f"  - CSV: {fake_csv_path}")
  print(f"  - JSON: {fake_json_path}")
  
  
  


  
  
if __name__ == "__main__":
  main()