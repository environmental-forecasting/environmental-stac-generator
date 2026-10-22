# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

Here are some ways you can contribute:

## Types of Contributions

### Report Bugs

Report bugs at https://github.com/environmental-forecasting/environmental-stac-generator/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

### Write Documentation

`environmental-stac-generator` could always use more documentation, whether as part of the
official `environmental-stac-generator` docs, in docstrings, or even on the web in blog posts,
articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at https://github.com/environmental-forecasting/environmental-stac-generator/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

### Get Started!

Ready to contribute? Here's how to set up `environmental-stac-generator` for local
development. It requires Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).

1. Fork the repository on GitHub and clone your fork locally:

   ```bash
   git clone git@github.com:<your-username>/environmental-stac-generator.git
   cd environmental-stac-generator
   ```

2. Install the project and its dependencies into a virtual environment:

   ```bash
   uv sync
   ```

3. Create a branch for your changes:

   ```bash
   git checkout -b name-of-your-bugfix-or-feature
   ```

4. Make your changes, adding or updating tests in `tests/` as appropriate.

5. Before committing, run the checks that CI runs:

   ```bash
   make format  # auto-fix lint issues and format the code with ruff
   make lint    # check code style and static rules with ruff
   make test    # run the test suite with pytest and coverage
   make audit   # check locked dependencies for known vulnerabilities
   ```

6. If you changed the CLI or netCDF requirements, update the docs in `docs/`
   and preview them locally:

   ```bash
   make docs-install
   make docs  # serves at http://127.0.0.1:8000
   ```

7. Commit your changes and push your branch to GitHub:

   ```bash
   git add .
   git commit -m "Description of your changes"
   git push origin name-of-your-bugfix-or-feature
   ```

8. Open a pull request against `main`. Tests (Python 3.12 and 3.13), linting,
   CodeQL and the OpenSSF Scorecard run automatically on pull requests; please
   make sure they pass.

Dependency updates are handled automatically by Dependabot.
