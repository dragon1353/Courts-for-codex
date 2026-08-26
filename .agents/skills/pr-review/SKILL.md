---
name: pr-review
description: Review pull requests and local diffs for the Courts Python legal-RAG repository. Use when given a GitHub PR URL or number, asked to review a branch or diff, or asked whether a change is ready to merge; prioritize correctness, data safety, API and streaming compatibility, CI evidence, and release risk. Do not use for implementing fixes unless the user also asks to change code.
---

# Review a Courts pull request

Perform an evidence-first review. Keep the review read-only unless the user explicitly asks for fixes or GitHub review actions.

## Establish the review scope

1. Read `AGENTS.md` and preserve unrelated working-tree changes.
2. Resolve the PR target from the supplied URL or number. Use an available GitHub connector or CLI only for read-only metadata, diffs, review threads, and checks.
3. If no PR is supplied, review the current local diff against the merge base or the user-named base branch.
4. If Git reports dubious ownership, use a per-command `-c safe.directory=<repo-root>` option. Do not modify global Git configuration.
5. Record the base and head revisions, changed files, diff statistics, and available check results before judging the change.

## Inspect the change

Inspect the complete diff and trace changed behavior into its callers and consumers. Prioritize:

- data loss, unexpected dataset or model regeneration, and disclosure of judgment text;
- Flask request validation, response status, task-state transitions, and thread safety;
- streamed HTML fragment ordering and the `<!-- PROGRESS: ... -->` contract used by `static/app.js`;
- retrieval relevance, empty-data behavior, model/checkpoint compatibility, and Ollama failure handling;
- Selenium download behavior, paths, timeouts, retries, and unintended browser or network side effects;
- TurboQuant dimensions, device placement, imports, numerical correctness, and CPU/GPU assumptions;
- dependency, configuration, documentation, and release compatibility.

Do not report style-only preferences as defects. Verify that every finding is introduced by the reviewed change and is reproducible or directly supported by code.

## Verify safely

1. Run `git diff --check`.
2. Parse changed Python files, or all project Python files when practical, without importing modules or writing bytecode.
3. Run the exact relevant CI or test command only when it is available locally.
4. Do not launch the GUI, download judgments, train models, rebuild datasets or ChromaDB, call Ollama, or download Hugging Face models as part of a routine review.
5. State every skipped check and the reason.

## Report findings

List actionable findings first, ordered by severity:

- `P0`: immediate data loss, security, privacy, or unusable release;
- `P1`: likely production failure or major incorrect legal-analysis behavior;
- `P2`: bounded defect, regression, or important missing validation;
- `P3`: low-risk maintainability issue with a concrete consequence.

For each finding, provide a short title, exact file and tight line range, triggering scenario, consequence, and smallest reasonable correction. Follow with open questions, verification performed, and a concise merge-readiness assessment. If no actionable findings exist, say so explicitly and still list residual risks and unrun checks.

Never submit a GitHub review, comment, approval, or change request unless the user explicitly authorizes that write action.
