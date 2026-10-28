# Releasing

Releases are published by `.github/workflows/publish.yml` through PyPI
trusted publishing. The repository stores no API tokens.

## One-time setup

1. On https://pypi.org/manage/account/publishing/, add a pending publisher:
   - PyPI project name: `turkmenOCR-lib`
   - Owner: `alibeg-begow`
   - Repository name: `turkmenOCR-lib`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`
2. On https://test.pypi.org/manage/account/publishing/, add the same entry,
   with the environment name `testpypi`.
3. In the GitHub repository settings, under Environments, the environments
   `pypi` and `testpypi` must exist. `pypi` only accepts deployments from
   tags that match `v*`. Optionally, add yourself as a required reviewer on
   `pypi`, so that every PyPI upload waits for your approval.

## Dry run on TestPyPI

Open the Actions tab, choose Publish and click Run workflow. This builds the
package and uploads it to TestPyPI only. Then verify it in a fresh
environment:

```
python -m venv /tmp/try
/tmp/try/bin/pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ turkmenOCR-lib
/tmp/try/bin/turkmenocr --version
```

TestPyPI, like PyPI, never accepts the same version twice. A repeated dry
run of an unchanged version is skipped without failing.

## Publishing a release

1. Run the GPU check below on a machine with the real weights.
2. Update `__version__` in `turkmenocr/__init__.py` and add a section for the
   new version to `CHANGELOG.md`.
3. Commit, then create and push a tag that matches the version:

   ```
   git tag v0.1.0
   git push origin v0.1.0
   ```

   The tag must contain the full version: `v0.1.0` matches version `0.1.0`,
   while `v0.1` does not and stops the workflow before anything is uploaded.

4. The workflow checks that the tag matches the version, builds the
   package, publishes it to TestPyPI and then PyPI, and creates a GitHub
   release with the built files attached.

## GPU check before a release

```
pip install dist/turkmenOCR_lib-<version>-py3-none-any.whl
turkmenocr -path="<weights folder>" -img="<sample page>.png"
turkmenocr -path="<weights folder>" -img="<sample page>.png" --task markdown
```

Confirm that the text is correct, that standard output contains only the
text, and that a second run starts without installing anything.
