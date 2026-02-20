# GitHub Actions Build Workflow

The repository includes a GitHub Actions workflow to automate the build process for Windows and Linux.

## Triggers

The build workflow is **not** triggered on every push. It runs only in the following scenarios:

1.  **Manual Trigger:** You can manually start the workflow from the "Actions" tab in the GitHub repository. Select the "Build BLITZ" workflow and click "Run workflow".
2.  **Tag Push:** Pushing a tag that starts with `build` (e.g., `build-v1.4.2`, `build_test`) will trigger the workflow.

    ```bash
    git tag build-v1.4.3
    git push origin build-v1.4.3
    ```

## Artifacts

Upon successful completion, the workflow produces the following artifacts:

*   **Windows:** `BLITZ-Windows` (contains `BLITZ.exe`)
*   **Linux:** `BLITZ-Linux` (contains `BLITZ` executable)

These artifacts can be downloaded directly from the workflow run summary page.
