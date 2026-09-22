# Student Dropout Early-Warning ML

A reproducible machine-learning project for identifying higher-education students at risk of dropping out using enrolment and first-semester information.

The project prioritises early intervention, data-leakage prevention, transparent evaluation, and responsible reporting. It compares Logistic Regression, Random Forest, and XGBoost before tuning the selected model and its intervention threshold.

> This is a research and portfolio project, not a production-ready decision system.

## Project objectives

- Predict dropout risk early enough to support student interventions
- Convert the original three-class outcome into:
  - `Dropout`
  - `Not Dropout` (`Enrolled` or `Graduate`)
- Prevent second-semester information from entering the model
- Compare multiple classification algorithms using cross-validation
- Tune model parameters using training data only
- Select the classification threshold using validation data only
- Evaluate the final model once on an untouched test set
- Report unmet targets honestly

## Dataset

This project uses the UCI Machine Learning Repository dataset:

**Predict Students' Dropout and Academic Success**

- 4,424 student records
- 36 original features
- Enrolment, demographic, socioeconomic, macroeconomic, and academic data
- Original outcomes: Dropout, Enrolled, and Graduate
- No missing values reported in the source dataset

Dataset page:  
https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

### Citation

Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021).  
*Predict Students' Dropout and Academic Success* [Dataset].  
UCI Machine Learning Repository.  
https://doi.org/10.24432/C5MC89

The source dataset is distributed under the Creative Commons Attribution
4.0 International license (CC BY 4.0).

## Methodology

### Early-warning feature window

Only information available at enrolment or by the end of the first semester
is used for modelling. All second-semester variables are programmatically
excluded, with assertions that stop execution if a prohibited feature enters
the model.

The final model uses 31 early-warning features, including engineered measures
such as:

- First-semester approval ratio
- Number of unapproved first-semester units
- Approved units per evaluation
- Units without evaluation ratio
- Financial risk indicator

Gender, nationality, international status, and educational special-needs
status are excluded as a governance choice.

### Experimental design

The data is divided using stratified sampling:

| Dataset | Share | Purpose |
|---|---:|---|
| Training | 60% | Model comparison and hyperparameter tuning |
| Validation | 20% | Classification-threshold selection |
| Test | 20% | Final holdout evaluation only |

Model comparison uses five-fold stratified cross-validation on the training
set.

### Models compared

- Logistic Regression
- Random Forest
- XGBoost

XGBoost achieved the highest training cross-validation average precision and
was selected for tuning.

An optional expanded search for a future iteration is defined separately in
`iteration8_5_tuning.py`. Importing that module does not run its Random Forest
or XGBoost searches and does not alter the results reported below.

## Results

The classification threshold was set to `0.27` using validation data. This
threshold prioritises identifying students at risk while maintaining the
highest possible precision under the recall requirement.

### Final holdout performance

| Metric | Result |
|---|---:|
| Accuracy | 0.806 |
| Precision | 0.644 |
| Recall | 0.884 |
| F1 score | 0.745 |
| F2 score | 0.822 |
| Average precision | 0.885 |
| ROC AUC | 0.914 |
| Lift at top 20% | 3.011 |
| Train-test accuracy gap | 0.026 |

### Success criteria

| Criterion | Target | Result | Status |
|---|---:|---:|:---:|
| Lift at top 20% | > 2.50 | 3.011 | Pass |
| Dropout recall | ≥ 0.85 | 0.884 | Pass |
| Dropout precision | ≥ 0.70 | 0.644 | **Not met** |
| Train-test accuracy gap | < 0.10 | 0.026 | Pass |

The model identifies approximately 88% of dropout cases in the holdout set,
but its precision remains below the predefined 70% target. It should therefore
be treated as a research prototype requiring further validation rather than a
deployment-ready system.

## Key findings

- First-semester academic progress was the strongest source of predictive
  information.
- The first-semester approval ratio was the most important XGBoost feature.
- Tuition-fee status, financial-risk indicators, age at enrolment, debtor
  status, scholarship status, and course were also influential.
- Students whose tuition fees were not up to date had a substantially higher
  observed dropout rate.
- A threshold below the conventional `0.50` was required to achieve the
  intervention-focused recall target.
- The small train-test accuracy gap indicates limited evidence of severe
  overfitting, although external validation is still required.

## Repository structure

```text
.
├── iteration3.py
├── iteration3_report.ipynb
├── iteration8_5_tuning.py
└── iteration3_outputs/
    ├── data_dictionary.csv
    ├── data_integration_report.md
    ├── data_overview.csv
    ├── final_holdout_metrics.csv
    ├── final_success_criteria.csv
    ├── model_comparison_train_cv.csv
    ├── randomized_search_cv_results.csv
    ├── validation_threshold_search.csv
    └── integration_*.csv
```

- `iteration3.py` contains the reproducible modelling workflow.
- `iteration3_report.ipynb` contains the executed analysis and visualisations.
- `iteration3_outputs/` contains the integration audit, model comparisons,
  threshold results, and final evaluation tables.

## Running the project

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/student-dropout-early-warning-ml.git
cd student-dropout-early-warning-ml
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn xgboost ucimlrepo jupyter
```

### 4. Run the analysis

```bash
python iteration3.py
```

The dataset is downloaded through `ucimlrepo` unless a local CSV path is
configured in the script.

To explore the notebook:

```bash
jupyter notebook iteration3_report.ipynb
```

## Responsible-use considerations

Dropout-risk predictions can affect students in sensitive ways.

- Predictions should be used to offer support, not impose penalties.
- A high-risk score does not establish why a student may leave.
- Excluding protected attributes does not by itself guarantee fairness.
- Performance should be evaluated across relevant student groups before use.
- The model was trained on data from a specific institution and may not
  generalise to other institutions or countries.
- Real-world use would require privacy review, stakeholder consultation,
  fairness testing, monitoring, and external validation.

## Limitations and future work

- Validate the workflow using data from additional institutions.
- Conduct subgroup performance and fairness assessments.
- Improve precision without sacrificing intervention-focused recall.
- Add probability calibration and uncertainty analysis.
- Use SHAP or permutation importance for stronger explanations.
- Add automated tests for leakage, schema changes, and reproducibility.
- Evaluate temporal validation using later student cohorts.
- Develop a human-in-the-loop intervention workflow.

## License

The project code may be released under the MIT License.

The original UCI dataset is separately licensed under CC BY 4.0 and must be
credited to its original creators. See the dataset page for its licensing and
citation requirements.

## Disclaimer

This repository is an educational machine-learning project. It is not intended
to make autonomous admissions, academic, disciplinary, or funding decisions.
