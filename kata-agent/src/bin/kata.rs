//! Minimal `kata` CLI entry point for running closed-loop test repair fixtures.

use std::path::PathBuf;
use std::process::ExitCode;

use kata_agent::repair::{run_closed_loop_repair, RepairConfig};

fn print_usage() {
    println!(
        "kata - Hanzo Kata autonomous test-repair runner\n\n\
         USAGE:\n  kata run [--fixture <dir>] [--max-attempts <n>]\n\n\
         OPTIONS:\n  --fixture <dir>       Path to broken test fixture directory (default: examples/fixtures/broken_math)\n  --max-attempts <n>    Maximum repair attempts before escalating (default: 3)\n  -h, --help            Print help"
    );
}

#[tokio::main]
async fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        print_usage();
        return ExitCode::SUCCESS;
    }

    if args.first().map(String::as_str) != Some("run") {
        eprintln!("unsupported command `{}` (try `kata run`)", args[0]);
        return ExitCode::FAILURE;
    }

    let mut fixture_dir = PathBuf::from("examples/fixtures/broken_math");
    let mut max_attempts: u32 = 3;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--fixture" => {
                let Some(val) = args.get(i + 1) else {
                    eprintln!("missing path after --fixture");
                    return ExitCode::FAILURE;
                };
                fixture_dir = PathBuf::from(val);
                i += 2;
            }
            "--max-attempts" => {
                let Some(val) = args.get(i + 1) else {
                    eprintln!("missing integer after --max-attempts");
                    return ExitCode::FAILURE;
                };
                match val.parse::<u32>() {
                    Ok(n) => max_attempts = n,
                    Err(e) => {
                        eprintln!("invalid --max-attempts `{val}`: {e}");
                        return ExitCode::FAILURE;
                    }
                }
                i += 2;
            }
            other => {
                eprintln!("unknown argument `{other}`");
                return ExitCode::FAILURE;
            }
        }
    }

    let cfg = RepairConfig::new(fixture_dir).with_max_attempts(max_attempts);
    match run_closed_loop_repair(&cfg).await {
        Ok(outcome) => {
            println!("Stages: {}", outcome.stages.join(" -> "));
            if !outcome.initial_stderr.is_empty() {
                println!("Detect failure:\n{}", outcome.initial_stderr);
            }
            println!("Verify output:\n{}", outcome.final_stdout);
            println!(
                "Transcript entries recorded: {} (attempts used: {})",
                outcome.transcript_len, outcome.attempts_used
            );
            if outcome.passed {
                println!("Result: PASS (closed-loop repair verified)");
                ExitCode::SUCCESS
            } else {
                eprintln!(
                    "Result: ESCALATED after {} attempt(s)",
                    outcome.attempts_used
                );
                ExitCode::FAILURE
            }
        }
        Err(err) => {
            eprintln!("Error: {err}");
            ExitCode::FAILURE
        }
    }
}
