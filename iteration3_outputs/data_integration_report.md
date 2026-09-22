# Data Integration Report

## Purpose

The student dataset was organised into logical column groups to make the data preparation process transparent and reproducible. The groups were assembled horizontally using the student row index before modelling.

## Integration method

The analytical dataframe contained 4,424 student records and 46 columns. It was split into five components: demographic, academic, financial/context, application/context, and outcome/label information. Each component retained the original row index, so the tables could be combined with `pd.concat(..., axis=1)` without changing the student-level observations.

| Component | Columns | Rows |
|---|---:|---:|
| demographic | 7 | 4424 |
| academic | 21 | 4424 |
| financial_context | 7 | 4424 |
| application_context | 6 | 4424 |
| outcome_and_labels | 5 | 4424 |

## Assembly and validation

The component tables were concatenated horizontally and reordered to the original column sequence. The reconstructed dataframe was then compared with the source dataframe. The validation confirmed that the assembled table had the same 4,424 rows, 46 columns, index, column order, and values as the source table.

**Integration result:** PASS

## Output files

The split component tables and reconstructed integrated table were exported to `iteration3_outputs/`. These files provide an auditable record of the integration process and can be used to reproduce the analysis.
