/**
 * Hanzo Kata TypeScript SDK Quickstart.
 *
 * Demonstrates autonomous code triage and patch generation.
 */

const API_HOST = process.env.HANZO_API_HOST || "https://api.hanzo.ai";
const API_KEY = process.env.HANZO_API_KEY || "";

async function generatePatch(repo: string, errorLog: string): Promise<string> {
  const url = `${API_HOST.replace(/\/$/, "")}/v1/chat/completions`;
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(API_KEY ? { Authorization: `Bearer ${API_KEY}` } : {}),
    },
    body: JSON.stringify({
      model: "zen",
      messages: [
        { role: "system", content: "You are Hanzo Kata, an autonomous coding agent." },
        {
          role: "user",
          content: `Repo: ${repo}\nError: ${errorLog}\nGenerate minimal unified diff patch.`,
        },
      ],
      max_tokens: 1024,
      temperature: 0.0,
    }),
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${await response.text()}`);
  }

  const data: any = await response.json();
  const msg = data.choices[0]?.message;
  return msg?.content || msg?.reasoning_content || "";
}

async function main() {
  console.log(`Connecting to Kata engine at ${API_HOST}...`);
  const patch = await generatePatch(
    "hanzoai/gateway",
    "error[E0308]: mismatched types: expected usize, found u64"
  );
  console.log("\n--- Kata Patch Generated ---");
  console.log(patch.trim());
}

main().catch(console.error);
