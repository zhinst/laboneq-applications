# Installing the package

This document describes the internal installation procedure of `laboneq_applications`.

The releases are published to the project's public GitLab [package registry](https://gitlab.zhinst.com/qccs/laboneq-applications/-/packages).

## Install

`pip install laboneq-applications --index-url https://L1QAL_REGISTRY:glpat-E2_41PzMiJcaxD2z6Xk3@gitlab.zhinst.com/api/v4/projects/637/packages/pypi/simple`

or a specific version:

`pip install laboneq-applications==0.1.0 --index-url https://L1QAL_REGISTRY:glpat-E2_41PzMiJcaxD2z6Xk3@gitlab.zhinst.com/api/v4/projects/637/packages/pypi/simple`


## Install from Git

For the users who wish to install a specific brach, you can install the `laboneq-applications` directly from `Git`. By default the `main` branch will be installed.

```
pip install --upgrade git+https://gitlab.zhinst.com/qccs/laboneq-applications.git
or
pip install --upgrade git+https://gitlab.zhinst.com/qccs/laboneq-applications.git@<feature_branch_to_be_used>
```
