from __future__ import annotations

from workflows.audit import check_stage1_completion


def test_completion_changes_when_all_methods_successful():
    methods = [
        {"method": "A", "execution_status": "completed", "validity_status": "valid"},
        {"method": "B", "execution_status": "completed", "validity_status": "valid"},
    ]
    out = check_stage1_completion.assess_completion(methods, repository_integrity=True)
    assert out["stage1_execution_closed"] is True
    assert out["stage1_all_methods_successful"] is True
    methods[1]["execution_status"] = "blocked"
    out = check_stage1_completion.assess_completion(methods, repository_integrity=True)
    assert out["stage1_execution_closed"] is False
    assert out["stage1_all_methods_successful"] is False
    assert out["blocked_with_evidence"] == ["B"]
