from __future__ import annotations

from workflows.audit import check_keshu_completion


def test_owner_completion_requires_real_flab_and_laco_closure():
    methods = {
        "Flab-Pruner": {"structural_primary": True, "benchmark_guided_experimental": {"implementation_closed": False, "target_smoke_closed": False}},
        "LLM-Pruner": {"primary_audit": True},
        "SliceGPT": {"primary_audit": True},
        "LaCo": {"registry": {"execution_status": "partial", "validity_status": "diagnostic_only"}},
    }
    out = check_keshu_completion.assess_owner_completion(methods)
    assert out["owner_execution_closed"] is False
    methods["Flab-Pruner"]["benchmark_guided_experimental"] = {"implementation_closed": True, "target_smoke_closed": True}
    methods["LaCo"]["registry"] = {"execution_status": "skipped", "validity_status": "not_applicable"}
    out = check_keshu_completion.assess_owner_completion(methods)
    assert out["owner_execution_closed"] is True


def test_owner_scope_does_not_close_global_stage():
    out = check_keshu_completion.global_stage_override_reason()
    assert out["stage1_execution_closed"] is False
    assert "Only the methods owned" in out["reason"]
