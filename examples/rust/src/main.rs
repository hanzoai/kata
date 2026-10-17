//! Hanzo Kata Rust SDK Quickstart.
//!
//! Submits a build/test failure triage prompt to generate minimal unified diff patches.

use serde_json::json;
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_host = env::var("HANZO_API_HOST").unwrap_or_else(|_| "https://api.hanzo.ai".into());
    let api_key = env::var("HANZO_API_KEY").unwrap_or_default();

    println!("Connecting to Kata at {api_host}...");

    let payload = json!({
        "model": "zen",
        "messages": [
            {"role": "system", "content": "You are Hanzo Kata, an autonomous coding agent. Generate minimal unified diffs."},
            {"role": "user", "content": "Repo: hanzoai/core\nError: line 5: expected semicolon\nProvide patch."}
        ],
        "temperature": 0.0,
        "max_tokens": 512
    });

    let client = reqwest::blocking::Client::new();
    let mut req = client.post(format!("{}/v1/chat/completions", api_host.trim_end_matches('/')))
        .header("Content-Type", "application/json")
        .json(&payload);

    if !api_key.is_empty() {
        req = req.header("Authorization", format!("Bearer {api_key}"));
    }

    let resp = req.send()?;
    let status = resp.status();
    let body: serde_json::Value = resp.json()?;

    println!("HTTP Status: {status}");
    let msg = &body["choices"][0]["message"];
    let patch = msg["content"].as_str().or_else(|| msg["reasoning_content"].as_str()).unwrap_or("");
    println!("Kata Patch:\n{}", patch.trim());

    Ok(())
}
