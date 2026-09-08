from run_csv_eval import extract_number  # adjust import path if needed

test_str = "The standard deviation of column **f1** is about **0.2367**."
print(extract_number(test_str))