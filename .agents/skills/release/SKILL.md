---
name: release
description: Audit and prepare releases for the Courts Python legal-RAG repository. Use when asked to review a tag or commit range, assess release readiness, draft release notes, choose files for a release, create a version or tag, or publish a GitHub Release; default to a read-only readiness report and require explicit authorization for tags, pushes, uploads, or publishing.
---

# Prepare a Courts release

Separate readiness analysis from release execution. The repository currently has no version manifest, release workflow, or established artifact definition, so never invent those decisions silently.

## Define the release scope

1. Read `AGENTS.md`.
2. Resolve the requested target version, previous tag or base revision, release candidate revision, and intended audience.
3. If the user did not specify a range, inspect tags and history. When no usable tag exists, report the ambiguity and use an explicitly disclosed comparison base only for a draft audit.
4. Determine whether the requested output is a readiness report, release-note draft, local packaging change, Git tag, or published GitHub Release.
5. Keep all GitHub and publication actions read-only until the user explicitly authorizes the exact mutation.

## Audit the range

Inspect commits, changed files, diff statistics, user-visible behavior, and verification evidence. Check:

- working-tree cleanliness and unrelated local changes;
- dependency reproducibility, Python compatibility, and missing manifest or lock files;
- Flask and streamed UI compatibility;
- model and vocabulary checkpoint compatibility;
- dataset, ChromaDB, report, PDF, bytecode, and large-binary changes;
- potential personal data in judgment-derived artifacts;
- Ollama model/configuration changes and documentation drift;
- TurboQuant import, numerical, device, and optional Hugging Face validation risk;
- migration, rollback, and operator steps.

Do not include `dataset/`, `temp/`, PDFs, model checkpoints, vector databases, or bytecode in a release artifact unless the user explicitly selects them and confirms the data and size implications.

## Verify readiness

1. Run `git diff --check` and parse Python without imports or bytecode writes.
2. Run only locally available, relevant tests. Do not call Selenium, Ollama, training, ChromaDB rebuilding, or remote model downloads during a routine readiness audit.
3. Verify that documented startup and dependency instructions match the code. Flag the current README Playwright/Selenium mismatch until corrected.
4. Classify blockers, warnings, and accepted risks. Do not describe the release as ready when required checks were unavailable.

## Produce release materials

Create a concise draft containing:

- comparison range and release candidate;
- user-visible changes grouped by feature, fix, and operational change;
- breaking changes, migrations, model or data compatibility notes;
- verification results and unrun checks;
- blockers and rollback guidance;
- proposed tag and release title only when the versioning scheme is known.

## Execute only with authorization

Before creating a tag, pushing, uploading assets, or publishing a GitHub Release, show the exact tag, commit, title, selected artifacts, and destination. Proceed only after explicit approval. Verify the published state afterward and report immutable identifiers and links.
