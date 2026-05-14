# Retail Insights Assistant

This repository contains the Streamlit retail analytics assistant demo along with deployment support files.

## Purpose

- Keep the project source, deployment config, and documentation together.
- Use GitHub as the source of truth for changes.
- Maintain a clean branch and commit workflow.

## GitHub workflow

1. Make changes locally.
2. Check the repository state with `git status`.
3. Stage files with `git add <file>`.
4. Commit with a concise message:
   - `git commit -m "Update README with GitHub workflow"`
5. Push to the remote repository:
   - `git push origin main`

## Branching guidance

- Use `main` for stable updates and cleanup.
- For larger changes, create a feature branch:
  - `git checkout -b feature/<name>`
- Push the branch and open a pull request if review is needed.

## Local cleanup

- The local `.venv` folder is ignored and should not be committed.
- Keep temporary files such as `__pycache__`, `.DS_Store`, and environment files out of Git.
- Use `.gitignore` to prevent local artifacts from entering the repository.

## VS Code Source Control

- The Source Control panel shows modified files and branch status.
- Commit first, then push to update GitHub.
- The branch indicator in the lower-left corner shows the active branch.

## Notes

- This README focuses on GitHub workflow rather than implementation details.
- The repository is ready for continuous collaboration once changes are committed and pushed.
