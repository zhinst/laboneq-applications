# Deploy

This document contains information about the deployment procedure.

## Procedure

- Update the changelog in `docs/sources/release_notes.md`.

- Bump the version to the new version using `tbump <new-version`. This
  will automatically update the version in `pyproject.toml` and other files.
  See `tbump.toml` for the `tbump` configuration.

- Create a tag in the following format: vX.X.X* (e.g `v0.0.1`, `v0.0.1rc1`)

- Push the tag

- Push the `main` branch to GitHub. Make sure the commit pushed works with
  the current release of LabOne Q. The `release-*` branches are mirrored to
  GitHub automatically.

- Bump the version again to a development version, e.g. `v0.2.0dev0`.

Pushing the tag with the specified format will trigger a pipeline which is described in the next section.

## Publish pipeline

- Build and deploy the tagged commit to the GitLab package registry
