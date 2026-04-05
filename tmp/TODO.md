# TODO: Open-Source Professionalization

Items that require GitHub repository access or external setup and cannot be automated locally.

## GitHub Actions CI

Set up a minimal CI workflow that runs on every push/PR:

**File:** `.github/workflows/ci.yml`

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[dev]"
      - run: python -m pytest tests/ -v
```

**Why it can't be done locally:** Requires pushing to GitHub and enabling Actions in the repo settings.

---

## README Badges

Add badges to the top of `README.md` and `README_TR.md` once CI is live:

```markdown
[![CI](https://github.com/<owner>/odyssnet/actions/workflows/ci.yml/badge.svg)](https://github.com/<owner>/odyssnet/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
```

**Why it can't be done locally:** Need the actual GitHub repo URL and CI must be running for the badge to resolve.

---

## PyPI Publishing (Optional)

If you want `pip install odyssnet` to work publicly:

1. Create a PyPI account at https://pypi.org
2. Add a GitHub Actions workflow for publishing on tag push:

```yaml
name: Publish
on:
  push:
    tags: ['v*']
jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install build twine
      - run: python -m build
      - run: twine upload dist/*
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
```

3. Add `PYPI_TOKEN` to GitHub repo secrets.

---

## GitHub Repository Settings

Once the repo is public:

- [ ] Enable "Require pull request reviews" on main branch
- [ ] Enable "Require status checks to pass" (link to CI workflow)
- [ ] Set up branch protection rules
- [ ] Add repository topics: `deep-learning`, `pytorch`, `temporal-depth`, `recurrent-networks`, `chaos-dynamics`
- [ ] Add a repository description and website URL
- [ ] Enable Discussions for community Q&A
