# Deploy

This document contains information about the deployment procedure.

## Procedure

- Create a tag in the following format: vX.X.X* (e.g `v0.0.1`, `v0.0.1rc1`)

- Ensure that the tag version matches the one specified in `pyproject.toml` (TODO: version pickup automatically)

- Push the tag

Pushing the tag with the specified format will trigger a pipeline which is described in the next section.

## Publish pipeline

- Build and deploy the tagged commit to the GitLab package registry
