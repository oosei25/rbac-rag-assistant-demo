.PHONY: test test-unit test-integration test-security test-e2e security-report

test:
	pytest

test-unit:
	pytest -m unit

test-integration:
	pytest -m integration

test-security:
	pytest -m security

test-e2e:
	pytest -o "addopts=-ra --strict-markers" -m e2e tests/e2e

security-report:
	python -m evals.runner --output-dir reports/security-evals
