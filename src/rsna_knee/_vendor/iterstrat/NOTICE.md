# Vendored dependency: iterative-stratification

`ml_stratifiers.py` in this directory is copied unmodified from the
[`iterative-stratification`](https://github.com/trent-b/iterative-stratification)
package (PyPI: `iterative-stratification`, import name `iterstrat`), version 0.1.9,
by Trent J. Bradberry. Licensed under the BSD 3-Clause license (see the original
file's own header for the full attribution, including scikit-learn's copied
`StratifiedKFold` internals it builds on).

## Why vendored instead of pip-installed

The competition's scored submission kernels run with internet access disabled, so
any dependency not already present in Kaggle's base Docker image can't be
`pip install`-ed at runtime. `iterative-stratification` isn't part of that base
image. Rather than reimplementing multilabel-stratified K-fold splitting, this
small (~400 line), self-contained, permissively-licensed module is copied directly
into our own package so it ships with our code with no separate install step --
fully compliant with the competition's open-source-license requirement for any
code used to generate a submission.
