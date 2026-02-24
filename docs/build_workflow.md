# GitHub Actions Build Workflow

The repository includes a GitHub Actions workflow to automate the build process for Windows and Linux.

## Release Checklist (Internal)

Before triggering a build:

1.  **Version:** Update `version` in `pyproject.toml` (e.g. `1.5.2` → `1.5.3`).
2.  **Commit:** Commit and push your changes to `main`.
3.  **Release notes:** Write a few sentences for the GitHub Release (what changed, fixes, highlights). You can add them when the workflow finishes, or prepare them in advance.
4.  **Trigger:** Use one of the two options below.

## Triggers

The build workflow is **not** triggered on every push. It runs only in the following scenarios:

1.  **Manual (recommended):** GitHub → Actions → "Build BLITZ" → "Run workflow". No CLI needed. The workflow creates a tag (e.g. `build-1.5.3-42`) and publishes the Release; you add your release notes on the Release page.
2.  **Tag push (CLI):** Create and push a tag starting with `build`:

    ```bash
    git tag build-v1.5.3
    git push origin build-v1.5.3
    ```

    The tag name is arbitrary (e.g. `build-v1.5.3`, `build_test`); the Release version comes from `pyproject.toml`.

## Artifacts

Upon successful completion, the workflow produces the following artifacts:

*   **Windows:** `BLITZ-Windows` (contains `BLITZ.exe`)
*   **Linux:** `BLITZ-Linux` (contains `BLITZ` executable)

These artifacts can be downloaded directly from the workflow run summary page.

## Local Build Artefacts

`build/` and `dist/` are in `.gitignore`. They are created by PyInstaller during builds (locally or in CI). The GitHub Action produces them in a fresh runner and does not commit them.

If you have local `build/` or `dist/` from manual PyInstaller runs, you can safely delete them:
```bash
rm -rf build dist
```
