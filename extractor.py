import pandas as pd
import re

excel_path = "HSCodeandDescription.xlsx"

df = pd.read_excel(excel_path, dtype=str)
category_map = {}

for _, row in df.iterrows():
    raw_code = str(row.get("Code", "")).strip()
    description = str(row.get("Description", "")).strip()
    code_digits = re.sub(r"[^\d]", "", raw_code)
    if len(code_digits) >= 2:
        prefix = code_digits[:2].zfill(2)
        if prefix not in category_map:
            category_map[prefix] = description

# Output for copy-paste into code
print("HS_CATEGORY_MAP = {")
for k, v in sorted(category_map.items()):
    print(f'    "{k}": "{v}",')
print("}")
