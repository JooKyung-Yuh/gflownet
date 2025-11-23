"""Quick test to verify data generation works"""
import sys
from rules.rule_1 import Rule1_NoConsecutive1s
from data.generator import RealDataGenerator, FakeDataGenerator

print("Testing data generation...")
rule = Rule1_NoConsecutive1s()

print(f"\n1. Generating 50 real samples...")
sys.stdout.flush()
real_gen = RealDataGenerator()
real_samples = real_gen.generate(rule, count=50)
print(f"   ✅ Generated {len(real_samples)} real samples")

print(f"\n2. Generating 50 fake samples...")
sys.stdout.flush()
fake_gen = FakeDataGenerator()
fake_samples = fake_gen.generate(rule, real_samples, count=50)
print(f"   ✅ Generated {len(fake_samples)} fake samples")

print("\n✅ Data generation works!")
