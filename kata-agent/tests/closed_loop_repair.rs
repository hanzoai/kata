use std::path::PathBuf;

use kata_agent::repair::{run_closed_loop_repair, RepairConfig};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .join("examples/fixtures/broken_math")
}

#[tokio::test]
async fn closed_loop_repair_detects_fixes_and_verifies_broken_math_fixture() {
    let outcome = run_closed_loop_repair(&RepairConfig::new(fixture_dir()))
        .await
        .expect("repair runner succeeds");

    assert!(
        outcome.passed,
        "expected tests to pass after repair: {outcome:?}"
    );
    assert!(!outcome.escalated);
    assert_eq!(outcome.attempts_used, 1);
    assert!(
        outcome.initial_stderr.contains("adds_positive_integers")
            || outcome.initial_stderr.contains("FAILED"),
        "expected Detect output to name the failing test, got: {}",
        outcome.initial_stderr
    );
    assert!(
        outcome.final_stdout.contains("2 passed") && outcome.final_stdout.contains("0 failed"),
        "expected Verify output to pass both fixture tests, got: {}",
        outcome.final_stdout
    );
    assert!(outcome.transcript_len >= 5);
    assert!(outcome.stages.iter().any(|s| s.starts_with("detect(exit=")));
    assert!(outcome.stages.iter().any(|s| s.starts_with("triage(")));
    assert!(outcome.stages.iter().any(|s| s.starts_with("synthesize(")));
    assert!(outcome.stages.iter().any(|s| s == "verify(exit=0)"));
}

#[tokio::test]
async fn closed_loop_repair_escalates_cleanly_when_max_attempts_exhausted() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("lib.rs"),
        "#[test] fn unfixable() { assert_eq!(1, 2); }\n",
    )
    .expect("write fixture");

    let cfg = RepairConfig::new(dir.path()).with_max_attempts(1);
    let outcome = run_closed_loop_repair(&cfg)
        .await
        .expect("runner returns outcome cleanly");

    assert!(!outcome.passed);
    assert!(outcome.escalated);
    assert_eq!(outcome.attempts_used, 1);
    assert!(outcome.stages.last().is_some_and(|s| s == "escalate"));
}
