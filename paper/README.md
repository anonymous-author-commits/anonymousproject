# Manuscript sources

`main.tex` is the full manuscript. `tables/` holds the auto-generated
LaTeX tables (re-run the relevant experiment driver to refresh).

## Compile

```bash
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex      # second pass for cross-refs
```

Required figures live under `../figures/` (gitignored) and are produced
by the experiment drivers in `../experiments/`.
