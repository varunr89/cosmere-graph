.PHONY: serve test test-python test-all

serve:
	python3 -m http.server 8000

test:
	npx playwright test tests/ --reporter=list

test-python:
	python -m pytest tests/test_export_scores.py -v

test-all: test test-python
