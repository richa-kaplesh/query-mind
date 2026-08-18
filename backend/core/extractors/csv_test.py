import pandas as pd
from core.extractors.csv_extractor import CSVExtractor
# 1. Create a dummy CSV file to test
dummy_data = {
    "Employee_ID": [101, 102, 103, 104, 105],
    "Department": ["Sales", "Engineering", "Sales", "HR", "Engineering"],
    "Salary": [65000, 90000, 72000, 55000, 95000],
    "Status": ["Active", "Active", "On Leave", "Active", "Terminated"]
}
df_test = pd.DataFrame(dummy_data)
df_test.to_csv("test_data.csv", index=False)

# 2. Instantiate your extractor and run it
extractor = CSVExtractor()
extracted_pages = extractor.extract("test_data.csv")

# 3. Print the result
print("--- RAW PAGE TEXT SENT TO LLM ---\n")
print(extracted_pages[0].text)