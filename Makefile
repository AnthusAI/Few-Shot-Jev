.PHONY: install test preflight run report emotion-large-preflight emotion-large-run emotion-large-report

install:
	python3 -m venv .venv
	.venv/bin/pip install -e '.[dev]'

test:
	.venv/bin/pytest -q

preflight:
	.venv/bin/jev-agnews preflight

# Live API use is intentionally gated. Run `make preflight`, inspect its manifest,
# then supply the exact request ceiling you approve, e.g. MAX_REQUESTS=32000 make run.
run:
	@test -n "$(MAX_REQUESTS)" || (echo "Set MAX_REQUESTS after inspecting make preflight."; exit 2)
	.venv/bin/jev-agnews run --approve --max-requests $(MAX_REQUESTS)

report:
	.venv/bin/jev-agnews report

emotion-large-preflight:
	.venv/bin/python scripts/emotion_large_context_study.py preflight

emotion-large-run:
	@test -n "$(MAX_REQUESTS)" || (echo "Set MAX_REQUESTS after inspecting emotion-large-preflight."; exit 2)
	.venv/bin/python scripts/emotion_large_context_study.py run --approve --max-requests $(MAX_REQUESTS)

emotion-large-report:
	.venv/bin/python scripts/emotion_large_context_study.py report
