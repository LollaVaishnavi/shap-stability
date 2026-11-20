# The one command this repository promises a reviewer is `make verify`: it lints, runs
# the suite as its own step, then re-measures every figure and checks the prose against it,
# offline, from a fresh clone. Evidence regeneration is separate because it needs browser
# and node extras that CI does not install for the pure-Python jobs.

.PHONY: lint test receipts verify evidence clean

lint:
	ruff check .

test:
	mkdir -p reports
	python -m pytest -q \
		--junitxml=reports/junit.xml \
		--cov=shapstab \
		--cov-report=json:reports/coverage.json \
		--cov-report=term-missing \
		--cov-fail-under=100

receipts:
	python tools/collect_metrics.py --skip-tests
	python tools/check_numbers.py

verify: lint test receipts

evidence:
	python tools/render_diagrams.py
	python tools/capture_screenshots.py
	python tools/record_demo.py

clean:
	rm -rf reports .pytest_cache .ruff_cache **/__pycache__ *.egg-info
