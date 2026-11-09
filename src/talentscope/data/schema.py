"""Column-level knowledge about the IBM HR attrition dataset.

Everything the pipeline needs to know about the columns lives here: the
target, the identifier and constant columns to drop, which features are
numeric, ordinal or nominal, readable labels for the ordinal codes and the
plausible value ranges the validation step checks against.
"""
from __future__ import annotations

TARGET = "Attrition"
POSITIVE_LABEL = "Yes"
TARGET_FLAG = "AttritionFlag"          # 1 = left, 0 = stayed (added by prepare_dataset)

EXPECTED_ROWS = 1470
EXPECTED_COLUMNS = 35
EXPECTED_POSITIVES = 237

ID_COLUMNS = ["EmployeeNumber"]
CONSTANT_COLUMNS = ["EmployeeCount", "Over18", "StandardHours"]
DROP_COLUMNS = ID_COLUMNS + CONSTANT_COLUMNS

NUMERIC_FEATURES = [
    "Age", "DailyRate", "DistanceFromHome", "HourlyRate", "MonthlyIncome", "MonthlyRate",
    "NumCompaniesWorked", "PercentSalaryHike", "TotalWorkingYears", "TrainingTimesLastYear",
    "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
]

# Ordinal codes stay integers for modelling (their order carries meaning) and
# are decoded with these labels for charts and tables.
ORDINAL_LEVELS: dict[str, dict[int, str]] = {
    "Education": {1: "Below College", 2: "College", 3: "Bachelor", 4: "Master", 5: "Doctor"},
    "EnvironmentSatisfaction": {1: "Low", 2: "Medium", 3: "High", 4: "Very High"},
    "JobInvolvement": {1: "Low", 2: "Medium", 3: "High", 4: "Very High"},
    "JobLevel": {1: "Level 1", 2: "Level 2", 3: "Level 3", 4: "Level 4", 5: "Level 5"},
    "JobSatisfaction": {1: "Low", 2: "Medium", 3: "High", 4: "Very High"},
    "PerformanceRating": {1: "Low", 2: "Good", 3: "Excellent", 4: "Outstanding"},
    "RelationshipSatisfaction": {1: "Low", 2: "Medium", 3: "High", 4: "Very High"},
    "StockOptionLevel": {0: "None", 1: "Level 1", 2: "Level 2", 3: "Level 3"},
    "WorkLifeBalance": {1: "Bad", 2: "Good", 3: "Better", 4: "Best"},
}
ORDINAL_FEATURES = list(ORDINAL_LEVELS)

NOMINAL_FEATURES = [
    "BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "OverTime",
]

MODEL_FEATURES = NUMERIC_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES
ALL_COLUMNS = sorted(MODEL_FEATURES + DROP_COLUMNS + [TARGET])

EXPECTED_CATEGORIES: dict[str, list[str]] = {
    "Attrition": ["No", "Yes"],
    "BusinessTravel": ["Non-Travel", "Travel_Frequently", "Travel_Rarely"],
    "Department": ["Human Resources", "Research & Development", "Sales"],
    "EducationField": ["Human Resources", "Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"],
    "Gender": ["Female", "Male"],
    "JobRole": [
        "Healthcare Representative", "Human Resources", "Laboratory Technician", "Manager",
        "Manufacturing Director", "Research Director", "Research Scientist", "Sales Executive",
        "Sales Representative",
    ],
    "MaritalStatus": ["Divorced", "Married", "Single"],
    "OverTime": ["No", "Yes"],
    "Over18": ["Y"],
}

# Plausible inclusive ranges, used to flag unexpected values. They come from
# the published data dictionary and from what an HR record could contain.
VALUE_RANGES: dict[str, tuple[float, float]] = {
    "Age": (18, 65), "DailyRate": (100, 1500), "DistanceFromHome": (1, 30), "HourlyRate": (30, 100),
    "MonthlyIncome": (1000, 20000), "MonthlyRate": (2000, 27000), "NumCompaniesWorked": (0, 10),
    "PercentSalaryHike": (0, 30), "TotalWorkingYears": (0, 45), "TrainingTimesLastYear": (0, 6),
    "YearsAtCompany": (0, 45), "YearsInCurrentRole": (0, 45), "YearsSinceLastPromotion": (0, 45),
    "YearsWithCurrManager": (0, 45), "Education": (1, 5), "EnvironmentSatisfaction": (1, 4),
    "JobInvolvement": (1, 4), "JobLevel": (1, 5), "JobSatisfaction": (1, 4), "PerformanceRating": (1, 4),
    "RelationshipSatisfaction": (1, 4), "StockOptionLevel": (0, 3), "WorkLifeBalance": (1, 4),
    "EmployeeCount": (1, 1), "StandardHours": (80, 80),
}

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "Age": "Employee age in years",
    "Attrition": "Target: whether the employee left the company (Yes/No)",
    "BusinessTravel": "Frequency of business travel",
    "DailyRate": "Daily pay rate (synthetic currency units)",
    "Department": "Organisational department",
    "DistanceFromHome": "Commute distance from home to work",
    "Education": "Education level, 1 (Below College) to 5 (Doctor)",
    "EducationField": "Field of the highest qualification",
    "EmployeeCount": "Constant column, always 1 (dropped)",
    "EmployeeNumber": "Unique employee identifier (dropped)",
    "EnvironmentSatisfaction": "Satisfaction with the work environment, 1 (Low) to 4 (Very High)",
    "Gender": "Gender as recorded in the dataset",
    "HourlyRate": "Hourly pay rate (synthetic currency units)",
    "JobInvolvement": "Job involvement rating, 1 (Low) to 4 (Very High)",
    "JobLevel": "Seniority level, 1 to 5",
    "JobRole": "Job title",
    "JobSatisfaction": "Job satisfaction rating, 1 (Low) to 4 (Very High)",
    "MaritalStatus": "Marital status",
    "MonthlyIncome": "Monthly income (synthetic currency units)",
    "MonthlyRate": "Monthly rate (synthetic currency units)",
    "NumCompaniesWorked": "Number of companies the employee has worked for",
    "Over18": "Constant column, always Y (dropped)",
    "OverTime": "Whether the employee works overtime",
    "PercentSalaryHike": "Last salary increase, percent",
    "PerformanceRating": "Performance rating, 1 (Low) to 4 (Outstanding); only 3 and 4 occur",
    "RelationshipSatisfaction": "Relationship satisfaction rating, 1 (Low) to 4 (Very High)",
    "StandardHours": "Constant column, always 80 (dropped)",
    "StockOptionLevel": "Stock option level, 0 to 3",
    "TotalWorkingYears": "Total years of professional experience",
    "TrainingTimesLastYear": "Training sessions attended last year",
    "WorkLifeBalance": "Work-life balance rating, 1 (Bad) to 4 (Best)",
    "YearsAtCompany": "Tenure at the company in years",
    "YearsInCurrentRole": "Years in the current role",
    "YearsSinceLastPromotion": "Years since the last promotion",
    "YearsWithCurrManager": "Years with the current manager",
}

PRETTY_NAMES: dict[str, str] = {
    "Age": "Age", "DailyRate": "Daily rate", "DistanceFromHome": "Distance from home",
    "HourlyRate": "Hourly rate", "MonthlyIncome": "Monthly income", "MonthlyRate": "Monthly rate",
    "NumCompaniesWorked": "Companies worked", "PercentSalaryHike": "Salary hike %",
    "TotalWorkingYears": "Total working years", "TrainingTimesLastYear": "Trainings last year",
    "YearsAtCompany": "Years at company", "YearsInCurrentRole": "Years in current role",
    "YearsSinceLastPromotion": "Years since promotion", "YearsWithCurrManager": "Years with manager",
    "Education": "Education level", "EnvironmentSatisfaction": "Environment satisfaction",
    "JobInvolvement": "Job involvement", "JobLevel": "Job level", "JobSatisfaction": "Job satisfaction",
    "PerformanceRating": "Performance rating", "RelationshipSatisfaction": "Relationship satisfaction",
    "StockOptionLevel": "Stock option level", "WorkLifeBalance": "Work-life balance",
    "BusinessTravel": "Business travel", "Department": "Department", "EducationField": "Education field",
    "Gender": "Gender", "JobRole": "Job role", "MaritalStatus": "Marital status", "OverTime": "Overtime",
}


def pretty_feature(name: str) -> str:
    """Translate a transformed feature name into a readable label.

    ``num__Age`` -> ``Age``; ``cat__JobRole_Sales Executive`` -> ``Job role = Sales Executive``.
    """
    for prefix in ("num__", "ord__", "cat__", "remainder__"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    for col in sorted(NOMINAL_FEATURES, key=len, reverse=True):
        if name.startswith(col + "_"):
            return f"{PRETTY_NAMES.get(col, col)} = {name[len(col) + 1:].replace('_', ' ')}"
    return PRETTY_NAMES.get(name, name)


def source_feature(name: str) -> str:
    """Map a transformed feature name back to the raw column it came from."""
    for prefix in ("num__", "ord__", "cat__", "remainder__"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    for col in sorted(NOMINAL_FEATURES, key=len, reverse=True):
        if name.startswith(col + "_"):
            return col
    return name
