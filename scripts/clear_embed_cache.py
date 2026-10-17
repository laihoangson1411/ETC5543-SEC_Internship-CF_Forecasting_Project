"""Pre-render step (_quarto.yml): delete Quarto's cache of embedded notebooks (.quarto/embed).

presentation.qmd embeds figures and tables from the analysis notebooks ({{< embed >}}). Quarto reuses the cached
outputs while a notebook's .qmd is unchanged, so a change in functions/*.py alone would not reach the slides.
Clearing the cache makes every render rebuild them from the current code. Plain Python, so it works on any OS.
"""
import os
import shutil

shutil.rmtree(os.path.join('.quarto', 'embed'), ignore_errors=True)
