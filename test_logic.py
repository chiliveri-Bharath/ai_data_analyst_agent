import pandas as pd
from data_cleaner import create_calculated_feature, extract_date_parts, one_hot_encode

# Load test data
df = pd.read_csv('test_features.csv')
print("Original Columns:", df.columns.tolist())

# 1. Test Math
print("\nTesting Math (Salary * 1.1)...")
df = create_calculated_feature(df, 'Salary', 'Salary', '+', 'DoubledSalary') # Simple add just for test
df = create_calculated_feature(df, 'Salary', None, '*', 'Bonus') # Test multiplier logic if implemented (actually I implemented binary)
# Let's test the implemented binary op: col1 op col2
df['TenPercent'] = 1.1
df = create_calculated_feature(df, 'Salary', 'TenPercent', '*', 'NewSalary')
print("After Math Columns:", df.columns.tolist())
print(df[['Salary', 'NewSalary']].head())

# 2. Test Date Extract
print("\nTesting Date Extract...")
df = extract_date_parts(df, 'JoinDate')
print("After Date Extract Columns:", [c for c in df.columns if 'JoinDate' in c])

# 3. Test One-Hot
print("\nTesting One-Hot...")
df = one_hot_encode(df, 'Department')
print("After One-Hot Columns:", [c for c in df.columns if 'Department' in c])

print("\nLogic Verification Complete!")
