# GenAI Telemetry Report

```json
{
  "unsupported_claim_rate": 0.029,
  "task_completion_rate": 0.971,
  "latency_p50_ms": 1045.08305850815,
  "latency_p95_ms": 1660.8444753296328,
  "regression_flag_rate": 0.04933333333333333,
  "human_review_rate": 0.06433333333333334
}
```


## Human review queue

| workflow_id   |   case_id | prompt_type       | review_reason                                                         |   priority_score |   unsupported_claim_proxy |   evidence_coverage |   task_completion |   latency_ms |   regression_flag |
|:--------------|----------:|:------------------|:----------------------------------------------------------------------|-----------------:|--------------------------:|--------------------:|------------------:|-------------:|------------------:|
| wf_001603     |      1603 | risk_routing      | unsupported_claim;regression_flag;low_evidence_coverage;high_latency; |          6.18491 |                         1 |            0.458061 |                 0 |     1694.12  |                 1 |
| wf_000552     |       552 | risk_routing      | unsupported_claim;regression_flag;low_evidence_coverage;              |          6.15107 |                         1 |            0.430229 |                 0 |     1531.65  |                 1 |
| wf_002569     |      2569 | risk_routing      | unsupported_claim;regression_flag;low_evidence_coverage;              |          6.03563 |                         1 |            0.525026 |                 0 |     1477.24  |                 1 |
| wf_001750     |      1750 | risk_routing      | unsupported_claim;regression_flag;low_evidence_coverage;              |          6.03408 |                         1 |            0.471474 |                 0 |     1332.07  |                 1 |
| wf_000217     |       217 | response_check    | unsupported_claim;regression_flag;low_evidence_coverage;              |          6.03173 |                         1 |            0.52788  |                 0 |     1474.5   |                 1 |
| wf_000431     |       431 | complaint_summary | unsupported_claim;regression_flag;low_evidence_coverage;              |          6.02825 |                         1 |            0.368438 |                 0 |     1045.21  |                 1 |
| wf_002552     |      2552 | evidence_lookup   | unsupported_claim;regression_flag;low_evidence_coverage;              |          6.01178 |                         1 |            0.528866 |                 0 |     1424.52  |                 1 |
| wf_001987     |      1987 | risk_routing      | unsupported_claim;regression_flag;low_evidence_coverage;              |          5.97474 |                         1 |            0.479848 |                 0 |     1197.78  |                 1 |
| wf_002582     |      2582 | risk_routing      | unsupported_claim;regression_flag;low_evidence_coverage;              |          5.97134 |                         1 |            0.348439 |                 0 |      842.566 |                 1 |
| wf_000210     |       210 | response_check    | unsupported_claim;regression_flag;low_evidence_coverage;              |          5.96339 |                         1 |            0.527528 |                 0 |     1293.49  |                 1 |
| wf_001764     |      1764 | response_check    | unsupported_claim;regression_flag;low_evidence_coverage;              |          5.9547  |                         1 |            0.517252 |                 0 |     1243.52  |                 1 |
| wf_001806     |      1806 | complaint_summary | unsupported_claim;regression_flag;low_evidence_coverage;              |          5.9527  |                         1 |            0.522258 |                 0 |     1251.43  |                 1 |
| wf_000964     |       964 | response_check    | unsupported_claim;regression_flag;low_evidence_coverage;              |          5.93118 |                         1 |            0.373275 |                 0 |      802.198 |                 1 |
| wf_001744     |      1744 | risk_routing      | unsupported_claim;regression_flag;low_evidence_coverage;              |          5.92886 |                         1 |            0.486405 |                 0 |     1094.17  |                 1 |
| wf_001614     |      1614 | complaint_summary | unsupported_claim;regression_flag;low_evidence_coverage;              |          5.92709 |                         1 |            0.404869 |                 0 |      874.671 |                 1 |
| wf_002993     |      2993 | risk_routing      | unsupported_claim;regression_flag;low_evidence_coverage;              |          5.91648 |                         1 |            0.464002 |                 0 |     1002.52  |                 1 |
| wf_000983     |       983 | risk_routing      | unsupported_claim;regression_flag;low_evidence_coverage;              |          5.91557 |                         1 |            0.492968 |                 0 |     1076.43  |                 1 |
| wf_002427     |      2427 | evidence_lookup   | unsupported_claim;regression_flag;low_evidence_coverage;              |          5.91083 |                         1 |            0.496458 |                 0 |     1073.14  |                 1 |
| wf_001189     |      1189 | evidence_lookup   | unsupported_claim;regression_flag;low_evidence_coverage;              |          5.90694 |                         1 |            0.521489 |                 0 |     1128.85  |                 1 |
| wf_001874     |      1874 | response_check    | unsupported_claim;regression_flag;low_evidence_coverage;              |          5.87325 |                         1 |            0.544869 |                 0 |     1101.69  |                 1 |
| wf_001926     |      1926 | response_check    | unsupported_claim;regression_flag;low_evidence_coverage;              |          5.87108 |                         1 |            0.539808 |                 0 |     1082.62  |                 1 |
| wf_002323     |      2323 | risk_routing      | unsupported_claim;regression_flag;low_evidence_coverage;              |          5.86801 |                         1 |            0.521072 |                 0 |     1025.16  |                 1 |
| wf_000258     |       258 | evidence_lookup   | unsupported_claim;regression_flag;low_evidence_coverage;              |          5.86415 |                         1 |            0.405923 |                 0 |      711.601 |                 1 |
| wf_002441     |      2441 | risk_routing      | unsupported_claim;regression_flag;low_evidence_coverage;              |          5.86277 |                         1 |            0.482831 |                 0 |      910.601 |                 1 |
| wf_000846     |       846 | risk_routing      | unsupported_claim;regression_flag;low_evidence_coverage;              |          5.85773 |                         1 |            0.462208 |                 0 |      842.989 |                 1 |
