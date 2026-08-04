from __future__ import annotations

from workflows.audit import check_keshu_completion


def test_owner_completion_requires_real_flab_and_laco_closure():
    methods = {
        "Flab-Pruner": {"structural_primary": True, "activation_smoke": False, "activation_blocker": "plain_hf_no_prune"},
        "LLM-Pruner": {"primary_audit": True},
        "SliceGPT": {"primary_audit": True},
        "LaCo": {"laco_core_smoke": False, "laco_file_probe_only": True},
    }
    out = check_keshu_completion.assess_owner_completion(methods)
    assert out["owner_execution_closed"] is False
    methods["Flab-Pruner"]["activation_blocker"] = "vendored_config_only_no_external_mask_schema"
    methods["LaCo"]["laco_core_smoke"] = True
    methods["LaCo"]["laco_file_probe_only"] = False
    out = check_keshu_completion.assess_owner_completion(methods)
    assert out["owner_execution_closed"] is True


def test_owner_scope_does_not_close_global_stage():
    out = check_keshu_completion.global_stage_override_reason()
    assert out["stage1_execution_closed"] is False
    assert "Only the methods owned" in out["reason"]
