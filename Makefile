.PHONY: install test lint format clean build publish

install:
	pip install -e ".[dev]" # -e for editable mode

install-prod:
	pip install .

test:
	pytest -v --cov= rag_journal tests/

test-watch:
	pytest -f tests/ # -f for watch mode (if pytest supports)

lint:
	flake8 rag_journal tests/
	mypy rag_journal

format:
	black rag_journal tests/
	isort rag_journal tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .mypy_cache .pytest_cache build dist *.egg-info .coverage

build:
	python -m build

publish-test:
	python -m twine upload --repository testpypi dist/*

publish:
	python -m twine upload dist/*

run:
	python -m rag_journal

dev:
	uvicorn rag_journal.app:app --reload  # For web projects

# Short aliases
t: test
f: format
c: clean
b: build
