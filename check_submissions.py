
import pandas as pd
import os

# Original test data
test = pd.read_csv("data/test.csv", dtype={"id": str})

# Required submission files
files = [
    "outputs/predictions/submission.csv",
    "outputs/predictions/submission_full.csv",
    "outputs/predictions/submission_E1126_2032.csv"
]

print("=" * 65)
print("       NEURAHACK SUBMISSION VALIDATION")
print("=" * 65)

for file in files:

    print(f"\nChecking: {os.path.basename(file)}")
    print("-" * 50)

    if not os.path.exists(file):
        print("FILE NOT FOUND")
        continue

    df = pd.read_csv(file, dtype={"id": str})

    # 1. Check columns
    correct_columns = list(df.columns) == ["id", "label"]

    # 2. Check row count
    correct_rows = len(df) == len(test)

    # 3. Check duplicate IDs
    no_duplicates = df["id"].is_unique

    # 4. Check missing values
    no_missing = df["id"].notna().all() and df["label"].notna().all()

    # 5. Check all test IDs are present exactly once
    correct_ids = (
        df["id"].isin(test["id"]).all()
        and test["id"].isin(df["id"]).all()
        and no_duplicates
    )

    # 6. Check labels are integers from 0 to 5
    numeric_labels = pd.to_numeric(df["label"], errors="coerce")

    valid_labels = (
        numeric_labels.notna().all()
        and numeric_labels.between(0, 5).all()
        and (numeric_labels % 1 == 0).all()
    )

    # Print results
    checks = {
        "Correct columns": correct_columns,
        "Correct row count": correct_rows,
        "No duplicate IDs": no_duplicates,
        "No missing values": no_missing,
        "All test IDs present exactly once": correct_ids,
        "Labels are integers (0-5)": valid_labels
    }

    for name, result in checks.items():
        print(f"{'PASS' if result else 'FAIL'} : {name}")

    print(f"\nTotal rows: {len(df)}")
    print(f"Unique labels: {sorted(numeric_labels.dropna().unique())}")

    if all(checks.values()):
        print("\nRESULT: VALID SUBMISSION")
    else:
        print("\nRESULT: INVALID SUBMISSION")

print("\n" + "=" * 65)
