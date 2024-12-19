# Deploy

This document contains information about the release process.

## Version bumping

This project uses `tbump` to manage its version information.

Running `tbump` will automatically update the version in `pyproject.toml`
and other files, commit the changes, *tag the commit* and
*push the commit and tag* to the remote. You can disable
automatically pushing changes by running `tbump --no-push ...`.

See `tbump.toml` for the `tbump` configuration and `tbump --help` for
command line help. Documentation can be found at
https://github.com/your-tools/tbump.

## Creating a release branch

- A release branch should be created for each new minor release. Patch
  releases should happen on the appropriate existing release branch.
  
- Release branch names should look like `release-<major>.<minor>.X`
  where the `X` is a literal `X`. For example, `release-1.1.X`.
  
- After creating the release branch, bump the version of `main` to the
  development version for the next minor version. For example,
  `tbump 1.3.0dev0 --no-tag`.

## Performing a release

- Ensure the release notes have been updated. These are maintained
  outside of this repository.

- Everything from here onwards happens on the release branch.

- Cherry pick changes from main and add fixes to the release branch
  as needed. Repeat as necessary.

- Wait for LabOne Q to be released.

- Change `ci/gitlab/test.yml` to build against the latest LabOne Q by
  switching `USE_LABONEQ_DEVELOP: "true"` to
  `USE_LABONEQ_DEVELOP: "false"`.

- Wait for the release branch builds to pass.

- Bump the version to the new version using `tbump <new-version>`. This
  will also create an appropriate tag and push the tag. Pushing the
  tag triggers the automated release pipeline (see next section).

- Once the release has been published, bump the version on the release
  branch to the next patch version. For example, `tbump 1.2.1dev0 --no-tag`.

- If this is a release on the latest release branch, manually push the `main`
  branch to GitHub. Make sure the commit pushed works with
  the current release of LabOne Q. The `release-*` branches are mirrored to
  GitHub automatically.

## Release pipeline

The release pipeline is defined in `ci/gitlab/release.yml`. It does
the following:

- Builds `sdist` and `wheels` for the package.
- Publishes these to the internal GitLab PyPI repository.
- Publishes these to PyPI.

It is triggered only on tagged builds.
