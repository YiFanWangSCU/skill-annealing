from scripts.build_stagec_causal_ablation_20260629 import (
    balanced_shuffled_exposures,
    balanced_stage_exposures,
)


def test_balanced_shuffled_exposures_are_uniform_and_deterministic() -> None:
    first = balanced_shuffled_exposures(total_count=24, seed=7)
    second = balanced_shuffled_exposures(total_count=24, seed=7)

    assert first == second
    assert {exposure: first.count(exposure) for exposure in set(first)} == {
        "full": 6,
        "partial": 6,
        "minimal": 6,
        "no_skill": 6,
    }


def test_balanced_stage_exposures_have_progressive_order_and_uniform_total() -> None:
    by_stage = {
        stage: balanced_stage_exposures(stage=stage, stage_sample_count=8, seed=11)
        for stage in (
            "stage1_full_partial",
            "stage2_partial_minimal",
            "stage3_minimal_no_skill",
        )
    }
    all_exposures = [exposure for exposures in by_stage.values() for exposure in exposures]

    assert set(by_stage["stage1_full_partial"]) == {"full", "partial"}
    assert set(by_stage["stage2_partial_minimal"]) == {"partial", "minimal"}
    assert set(by_stage["stage3_minimal_no_skill"]) == {"minimal", "no_skill"}
    assert {exposure: all_exposures.count(exposure) for exposure in set(all_exposures)} == {
        "full": 6,
        "partial": 6,
        "minimal": 6,
        "no_skill": 6,
    }
