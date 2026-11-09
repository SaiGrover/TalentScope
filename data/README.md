# Data

`raw/ibm_hr_employee_attrition.csv` is the **IBM HR Analytics Employee Attrition & Performance** dataset.

| | |
|---|---|
| Kaggle page | https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset |
| Copy used | https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv (IBM's public repository; identical content, no Kaggle login needed) |
| Rows × columns | 1,470 × 35 |
| Target | `Attrition`: 237 Yes (16.1%), 1,233 No |
| SHA-256 | `d11789e1db393cd1d985ca41a0e73a1d405543fb2c0d540a3b4f7d723bca92f7` |
| Nature | **fictional**; created by IBM data scientists, describes no real people |
| Licence | Open Database License (ODbL) as published on Kaggle |

The only change from the source file is encoding: the source has a UTF-8 byte-order mark and CRLF line endings, while
this copy is plain UTF-8 with LF line endings. Values and column order are unchanged.

Rebuild the file:

```bash
curl -L -o data/raw/source.csv https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv
```

```bash
python -c "import pandas as pd; pd.read_csv('data/raw/source.csv', encoding='utf-8-sig').to_csv('data/raw/ibm_hr_employee_attrition.csv', index=False, lineterminator='\n')"
```

Then confirm the hash above with `sha256sum data/raw/ibm_hr_employee_attrition.csv`.

The column meanings, roles (identifier, constant, numeric, ordinal, nominal) and plausible ranges are documented in
`src/talentscope/data/schema.py`. The validation report on the app's *Data lab* page checks the file against them on
every load.
