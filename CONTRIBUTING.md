# Contributing

Thanks for helping. This is a learning lab, so the bar for a change is: does it help someone understand how these
algorithms learn, and is it easy to read?

## Set up

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
pip install -r requirements-dev.txt          # the lab, the tests, and the screenshot tool
python -m playwright install chromium         # only for tools/screenshots.py
python -m pytest -q tests                     # about half a minute
python -m dashboard.server                    # http://127.0.0.1:8000
```

The AI analysis needs a TokenRouter or OpenRouter key in `.env` (see `.env.example`); everything else works without it,
and the tests never call a paid API.

## Making a change

- **New visions, reward features and algorithms** go in their registries; the command line, the New run form, the AI
  analysis and the glossary pick them up. See *Extending it* in the README. Add a glossary entry (`EXTRA` in
  `glossary.py`) that says what raising or lowering each setting does; a test checks this.
- **Keep it readable.** Match the code around your change. Explain *why* in comments, not what. Standard library or the
  existing dependencies only, unless there is a strong reason.
- **Dashboard:** plain HTML, canvas and Chart.js, no build step. Check new views at phone width (no sideways scrolling)
  and in dark mode. Buttons that change something get `"data-write": true`, so the read-only demo hides them.
- **Tests:** add one for behaviour you change. Tests must not need an API key or the network (see `tests/test_jev.py`
  for a fake decision model).
- **Screenshots:** if a view changes visibly, re-shoot the README images with `python tools/screenshots.py`.

## Sharing runs and lessons

Runs you train are saved in `lessons/` and can be committed: a run records only what is needed to judge its results.
Run `python lessons.py scrub` first if you trained with an older version, and never commit `lessons/*/runs/*/logs/`
(git-ignored) or `.env`. A new lesson should ask one question, change one setting, write its prediction down before
the results, and train each recipe with several seeds.

## Pull requests

CI runs the tests on Linux, Windows and macOS with Python 3.10 and 3.13, and in the Docker image. Describe what the
change teaches or fixes, and include before / after numbers or screenshots when results or views change.
