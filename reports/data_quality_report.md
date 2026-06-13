# Data Quality Report

| check              | column                       | passed   |   metric | detail                                 |
|:-------------------|:-----------------------------|:---------|---------:|:---------------------------------------|
| required_column    | product                      | True     | 1        | missing_like_rate=0.0000               |
| required_column    | issue                        | True     | 1        | missing_like_rate=0.0000               |
| required_column    | consumer_complaint_narrative | True     | 1        | missing_like_rate=0.0000               |
| required_column    | company_response_to_consumer | True     | 1        | missing_like_rate=0.0000               |
| required_column    | timely_response              | True     | 1        | missing_like_rate=0.0000               |
| required_column    | date_received                | True     | 1        | missing_like_rate=0.0000               |
| required_column    | state                        | True     | 0.996599 | missing_like_rate=0.0034               |
| duplicate_rate     | *                            | True     | 0        | duplicate_rate=0.0000                  |
| completeness       | *                            | True     | 0.999514 | missing-like-aware completeness=0.9995 |
| label_missing_rate | product                      | True     | 0        | label_missing_like_rate=0.0000         |
| label_missing_rate | timely_response              | True     | 0        | label_missing_like_rate=0.0000         |
| date_validity      | date_received                | True     | 1        | valid_date_rate=1.0000                 |
| evidence_coverage  | consumer_complaint_narrative | True     | 0.996699 | evidence_coverage=0.9967               |
