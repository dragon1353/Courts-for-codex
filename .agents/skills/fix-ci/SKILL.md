---
name: fix-ci
description: Diagnose and repair failing CI or GitHub Actions checks for the Courts Python legal-RAG repository. Use when given a failed check, workflow run, log excerpt, or request to fix CI; reproduce the failure, identify the root cause, implement the smallest in-scope fix, and verify it without triggering downloads, training, or publishing.
---

# Fix Courts CI

Diagnose from logs and repository evidence before editing. Preserve user changes and avoid expensive or external application workflows.

## Gather evidence

1. Read `AGENTS.md`.
2. Capture the failing workflow, job, step, command, runner, Python version, and first causal error rather than the final cascade.
3. Use a supplied GitHub URL or an available GitHub connector only to read checks and logs. If logs are unavailable, ask for the failing output instead of guessing.
4. Inspect `.github/workflows/`, dependency files, and test configuration. If they do not exist, state that the repository has no in-repo CI definition and distinguish a remote configuration failure from a source failure.
5. Inspect the working tree before editing. Never overwrite unrelated changes, especially `config.py`, tracked bytecode, datasets, model checkpoints, or generated reports.

## Reproduce narrowly

1. Run the failing command exactly when its dependency and fixture requirements are available.
2. Start with no-side-effect checks such as `git diff --check` and Python parsing without imports or bytecode writes.
3. Keep TurboQuant collection/import failures separate from numerical test failures.
4. Do not start PyWebView, Selenium, ChromeDriverManager, Ollama, model training, vector-database rebuilding, or Hugging Face validation unless the failing CI step explicitly requires it and the user authorizes any external or large-resource operation.
5. Record whether the failure reproduces locally and any environment difference.

## Fix the root cause

- Make the smallest change that fixes the causal failure.
- Prefer correcting source, imports, workflow configuration, or declared dependencies over weakening assertions or ignoring errors.
- Do not add broad dependency upgrades or regenerate large data artifacts without explicit scope.
- Preserve Windows behavior, UTF-8 Traditional Chinese text, Flask API contracts, and CPU-only fallback paths.
- Add or adjust a focused regression test when the repository has a usable test harness and the failure represents a code defect.

## Verify and hand off

1. Rerun the failing command, then the closest cheap surrounding checks.
2. Run `git diff --check` and inspect the final diff for unrelated files or generated artifacts.
3. Report the root cause, changed files, commands and results, skipped checks, and remaining environment risk.
4. Do not push, rerun or cancel GitHub Actions, edit repository settings, or merge changes unless explicitly authorized.
