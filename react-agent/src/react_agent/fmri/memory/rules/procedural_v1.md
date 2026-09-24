# Procedural rules v1

These are versioned repository rules. A memory curator may propose lessons but cannot edit this file.

1. claim_scope is always numeric_consistency_only.
2. Do not treat descriptive metrics (gray RMS, ROI top-K, RSA, CortexMAE embedding) as rejection thresholds.
3. Do not pad or interpolate T=12 arrays to T=16.
4. Do not bind 12-point reference or cohort to a 16s profile.
5. CortexMAE without a compatible reference score cannot answer quality_screening.
6. Unverified memory cannot change verdicts or verification_status.
7. Missing resources are resource gaps, not sample damage.
8. Gray-control not_applicable is normal for the control sample itself.
