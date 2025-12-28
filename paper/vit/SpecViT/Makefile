# Makefile for SpecViT LaTeX project
# Usage: make [target]

MAIN = main
LATEX = pdflatex
BIBTEX = bibtex
LATEXMK = latexmk

# Default target: build with latexmk (recommended)
.PHONY: all
all: latexmk

# Build with latexmk (handles dependencies automatically)
.PHONY: latexmk
latexmk:
	$(LATEXMK) -pdf $(MAIN).tex

# Manual build (if latexmk not available)
.PHONY: pdf
pdf:
	$(LATEX) $(MAIN).tex
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN).tex
	$(LATEX) $(MAIN).tex

# Clean temporary files
.PHONY: clean
clean:
	rm -f *.aux *.log *.out *.bbl *.blg *.synctex.gz *.fls *.fdb_latexmk
	rm -f *.toc *.lof *.lot *.nav *.snm *.vrb
	rm -f sections/*.aux

# Clean all including PDF
.PHONY: cleanall
cleanall: clean
	rm -f $(MAIN).pdf

# Watch mode (continuous compilation)
.PHONY: watch
watch:
	$(LATEXMK) -pdf -pvc $(MAIN).tex

# Help
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all      - Build with latexmk (default)"
	@echo "  latexmk  - Build with latexmk"
	@echo "  pdf      - Manual build (pdflatex + bibtex)"
	@echo "  clean    - Remove temporary files"
	@echo "  cleanall - Remove all generated files including PDF"
	@echo "  watch    - Continuous compilation with latexmk"
	@echo "  help     - Show this help"
