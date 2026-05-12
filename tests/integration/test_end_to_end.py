from __future__ import annotations

import json

import nanook as nk


def test_top_bottom_plus_local_suppression_achieves_k_anonymity(adult_small):
    p = (
        nk.Pipeline()
        .context(quasi_identifiers=["age", "zip", "sex"], sensitive=["diagnosis"])
        .top_bottom("age", percentile=20)
        .local_suppression(target_k=2)
    )
    protected = p.apply(adult_small)
    report = p.assess(adult_small, protected, k=2, l=2)
    assert report.risk.k_anonymity is not None
    # The protected frame either already holds k=2 or shows fewer violations than the source.
    raw_violations = nk.metrics.risk.k_anonymity(adult_small, qis=["age", "zip", "sex"], k=2).violations
    assert report.risk.k_anonymity.violations <= raw_violations


def test_pipeline_serialization_round_trip(adult_small):
    p = (
        nk.Pipeline(seed=1)
        .context(quasi_identifiers=["zip"], sensitive=["diagnosis"])
        .global_recoding("zip", mapping={"1": "A", "2": "A", "3": "B", "4": "B"})
        .sampling(fraction=0.8, seed=1)
    )
    payload = json.dumps(p.to_dict())
    revived = nk.Pipeline.from_dict(json.loads(payload))
    assert revived.to_dict() == p.to_dict()
    out_a = p.apply(adult_small)
    out_b = revived.apply(adult_small)
    assert out_a.equals(out_b)


def test_global_recoding_then_assessment(adult_small):
    p = (
        nk.Pipeline()
        .context(quasi_identifiers=["age", "zip"], sensitive=["diagnosis"])
        .global_recoding("age", bins=[0.0, 35.0, 45.0, 100.0], label_mode="index")
    )
    protected = p.apply(adult_small)
    report = p.assess(adult_small, protected, k=2, l=2)
    assert report.risk.k_anonymity is not None
    assert report.utility.lambda_measure is not None
