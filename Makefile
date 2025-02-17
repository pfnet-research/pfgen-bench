SHELL=/bin/bash
SRC = *.py

.PNONY: report
report: trials FORCE
	python3 pfgen_eval.py result/**/trials.jsonl.xz
	python3 pfgen_report.py

.PHONY: trials
trials: $(patsubst %/trial_001.jsonl,%/trials.jsonl.xz, \
	$(wildcard result/*/*/*/trial_001.jsonl))

%/trials.jsonl.xz: %/trial_001.jsonl
	cat $*/trial_???.jsonl | LC_ALL=C sort > $*/trials.jsonl
	xz -9 -k -f $*/trials.jsonl

# Lint (ruff, mypy)
.PHONY: lint
lint:
	ruff check $(SRC)
	mypy $(SRC)

# Format code (black, isort)
.PHONY: format
format:
	black $(SRC)
	isort $(SRC)

#########################################################################################
# Generic rules
#########################################################################################
.PHONY: FORCE
FORCE:
