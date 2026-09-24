## Dataset size

The original UCI dataset contains:

- **4,424 student records**
- **36 original predictor variables**
- **1 original outcome column**
- Raw shape: **4,424 rows × 37 columns**

For XGBoost, the project uses **31 early-warning predictors** after excluding sensitive/governance variables and second-semester information, then adding five engineered features.

The split is:

| Dataset | Records | Percentage | Purpose |
|---|---:|---:|---|
| Training | 2,654 | 60% | Model training and hyperparameter tuning |
| Validation | 885 | 20% | Classification-threshold selection |
| Test | 885 | 20% | Final evaluation |
| **Total** | **4,424** | **100%** | |

See [iteration3.py](/Users/Shared/RD/uoa/sem2/INFOSYS722/iteration3/project-python/iteration3.py:368).

## Target variable

The model predicts:

```text
is_dropout
```

- `1` = student dropped out
- `0` = student enrolled or graduated

There are **1,421 dropout students**, giving an overall dropout rate of approximately **32.1%**.

## XGBoost predictors

The 31 predictors are grouped below.

**Application and enrolment information**

1. Marital status
2. Application mode
3. Application order
4. Course
5. Daytime/evening attendance
6. Previous qualification
7. Previous qualification grade
8. Admission grade
9. Displaced status
10. Age at enrolment

**Family background**

11. Mother’s qualification
12. Father’s qualification
13. Mother’s occupation
14. Father’s occupation

**Financial information**

15. Debtor status
16. Tuition fees up to date
17. Scholarship holder
18. Financial risk flag — engineered

**First-semester academic information**

19. Curricular units credited
20. Curricular units enrolled
21. Curricular units evaluated
22. Curricular units approved
23. First-semester grade
24. Units without evaluations

**Engineered first-semester predictors**

25. First-semester approval ratio
26. Number of unapproved units
27. Approved units per evaluation
28. Units-without-evaluation ratio

**Economic information**

29. Unemployment rate
30. Inflation rate
31. GDP

The exact construction is in [iteration3.py](/Users/Shared/RD/uoa/sem2/INFOSYS722/iteration3/project-python/iteration3.py:193).

Second-semester variables are deliberately excluded to prevent data leakage. Gender, nationality, international status, and special-needs status are excluded as a governance decision.

One detail worth checking: the current code treats the mother’s/father’s qualification and occupation codes as numeric because their names do not match the categorical list exactly. You may want to ask your professor whether these coded variables should instead be treated as categorical.