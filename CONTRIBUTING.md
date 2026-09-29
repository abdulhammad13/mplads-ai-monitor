# Contributing

Thank you for helping improve MPLADS AI Monitor.

## Before making changes

- Read the README.
- Use a virtual environment.
- Do not commit credentials or private data.
- Keep changes focused and explain why they are needed.

## Development workflow

```powershell
git checkout -b feature/short-description
```

Make the change, then run at minimum:

```powershell
python -m py_compile dashboard\app.py
```

Run any project-specific API/data tests available in the repository.

## Commit style

Prefer clear commits such as:

```text
feat: add state risk heatmap
fix: correct team portrait asset resolution
refactor: isolate copilot styling
docs: improve local setup guide
```

## Pull requests

A good PR should include:

- What changed
- Why it changed
- How it was tested
- Screenshots for meaningful UI changes
- Any known limitations or follow-up work

Do not claim functionality that was not actually tested.
