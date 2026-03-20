<!--suppress HtmlDeprecatedAttribute -->
<div align="center">
   <p align="center">
      <br>
      <a href="docs/assets/images/AutoSklong_banner.avif">
         <img src="docs/assets/images/AutoSklong_banner.avif" alt="Auto-Sklong banner">
      </a>
   </p>
   <h4 align="center">
      An Automated Machine Learning library for longitudinal classification built on GAMA and Scikit-Longitudinal —
      <a href="https://doi.org/10.1109/BIBM62325.2024.10821737">Paper</a> ·
      <a href="https://auto-sklong.readthedocs.io/en/latest/">Documentation</a> ·
      <a href="https://pypi.org/project/Auto-Sklong/">PyPi Index</a>
   </h4>
</div>

Thank you for contributing to Auto-Sklong.

Please open an issue before large changes so we can align on scope, interfaces,
and documentation impact.

Before opening a pull request, please:

1. Install the project dependencies with `uv sync --all-groups`.
2. Run the test suite with `uv run pytest -sv tests/unit -p no:warnings` and `uv run pytest -sv tests/system -p no:warnings`.
3. Run the repository checks with `uv run pre-commit run --all-files`.
4. Build the documentation with `uv run mkdocs build`.
5. Update relevant docs or tutorials when your change affects user-facing behavior.

Please do not include private, sensitive, or real-world datasets in pull
requests, issues, or example artifacts.
