# Courts repository guidance

## Mandatory skill usage

- Use `$pr-review` for pull-request URLs, PR numbers, review requests, or review-ready diffs.
- Use `$fix-ci` for failed GitHub Actions checks, CI logs, or requests to repair a failing check.
- Use `$release` for release-readiness audits, tag or commit-range reviews, changelog preparation, versioning, or publishing requests.

## Project overview

- `app.py` is the Flask and PyWebView entry point. It exposes download, training, analysis, status, and report routes.
- `config.py` is the central runtime configuration. Keep paths, model names, thresholds, and training limits there.
- `googledata.py` automates the Taiwan Judicial Yuan site with Selenium and downloads PDFs.
- `pdf_loader.py`, `local_auto_label.py`, and `legal_data_processor.py` turn PDFs into combined text and CSV datasets.
- `models.py` and `train_model.py` define and train the character-level PyTorch autoencoder.
- `build_vectordb.py` builds the local ChromaDB index from PDFs and the trained model.
- `rag_agent.py` performs statistics, TF-IDF retrieval, local-model reranking, TurboQuant scoring, and streamed Ollama generation.
- `templates/` and `static/` contain the browser UI. Preserve the streamed HTML and progress-marker contract between `rag_agent.py` and `static/app.js`.
- `turboquant_pkg/` is an embedded quantization package with standalone validation code.
- `dataset/` and `temp/` contain large generated or derived artifacts. Treat them as data, not ordinary source files.

## Safety and data handling

- Preserve unrelated working-tree changes. Never reset, clean, or rewrite user changes to `config.py`, tracked bytecode, datasets, models, or reports.
- Treat judgment text and generated datasets as potentially sensitive personal data. Inspect only the minimum rows or excerpts needed, and do not upload or paste full records into external services.
- Do not run `app.py`, `googledata.py`, `local_auto_label.py`, `legal_data_processor.py`, `train_model.py`, `build_vectordb.py`, or `turboquant_pkg/validate.py` unless the task explicitly requires their side effects and the user approves any browser, network, download, model, or dataset operation.
- Do not delete or regenerate `dataset/best_model.pth`, `dataset/chroma_db/`, either CSV dataset, or `temp/` outputs unless explicitly requested.
- Do not change GitHub settings, push commits, rerun workflows, create tags, or publish releases without explicit authorization.

## Verification

- The repository has `requirements.txt` and `requirements-dev.txt`, but no lock file, GitHub Actions workflow, or documented release mechanism. Do not invent a green CI result.
- Always run `git diff --check` for source or workflow changes.
- Parse Python without producing tracked bytecode:
  `python -B -c "from pathlib import Path; files=[p for p in Path('.').rglob('*.py') if not any(part in {'.git','__pycache__','.agents'} for part in p.parts)]; [compile(p.read_text(encoding='utf-8-sig'), str(p), 'exec') for p in files]; print(f'parsed {len(files)} Python files')"`
- Reproduce a provided CI command exactly when its dependencies and fixtures are available. Report missing tools or services instead of silently skipping them.
- Treat the TurboQuant tests as a separate surface. Their current import style may fail during collection; report that failure precisely before changing imports or test commands.

## Change rules

- Keep source files UTF-8 and preserve Traditional Chinese user-facing text.
- Keep environment-specific paths and tunable values in `config.py`; avoid new hard-coded paths in application modules.
- Preserve Flask endpoint payloads and status codes unless the requested change includes an API migration.
- Preserve background-task state transitions and streamed response ordering when changing `app.py`, `rag_agent.py`, or `static/app.js`.
- Avoid committing new generated datasets, model checkpoints, vector databases, PDF content, browser downloads, or bytecode.
- Call out documentation drift: the README mentions Playwright, while the current downloader uses Selenium.

## Code Review Rules

### Sensitive and generated data

- Flag changes that add, regenerate, delete, or expose judgment-derived datasets, PDFs, reports, `dataset/best_model.pth`, `dataset/chroma_db/`, or `temp/` outputs without explicit authorization and a recovery plan.
- Safe path: keep routine changes source-only, inspect the minimum sensitive content needed, and require explicit approval before data or model mutations.

### API and streaming compatibility

- Flag changes that alter Flask endpoint payloads or status codes, background-task state transitions, streamed HTML ordering, or `<!-- PROGRESS: ... -->` markers without a documented migration and focused regression tests.
- Safe path: preserve the existing contract between `app.py`, `rag_agent.py`, and `static/app.js`, or update all producers, consumers, and tests together.

### Model and index compatibility

- Flag changes to vocabulary construction, latent dimensions, checkpoint fields, embedding behavior, chunking parameters, or ChromaDB metadata that lack an explicit compatibility or rebuild plan.
- Safe path: retain backward-compatible readers where practical and identify exactly which model or index artifacts must be rebuilt.

### Verification evidence

- Do not treat local tests as GitHub CI evidence. Report the exact commands run, distinguish unavailable external services, and identify skipped Selenium, Ollama, training, or vector-database checks.

## Handoff

- Summarize changed files, checks run, checks not run, data or external operations avoided, and any remaining release or CI risk.
