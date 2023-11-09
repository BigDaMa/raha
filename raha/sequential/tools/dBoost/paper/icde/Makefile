FIGDIR = ../graphics
TEXFILES=$(wildcard *.tex)

PYPLOTS_PDFS=$(wildcard $(FIGDIR)/*.pdf.py)
PYPLOTS_PNGS=$(wildcard $(FIGDIR)/*.png.py)
PYPLOTS_PDFS_RENDERED=$(PYPLOTS_PDFS:.pdf.py=.pdf)
PYPLOTS_PNGS_RENDERED=$(PYPLOTS_PNGS:.png.py=.png)

print-%:
	@echo $* = $($*)

all: paper.pdf

$(PYPLOTS_PDFS_RENDERED): %.pdf: %.pdf.py
	$< $@
	pdfcrop $@ $@

$(PYPLOTS_PNGS_RENDERED): %.png: %.png.py
	$< $@
	convert $@ -trim "$@"

paper.pdf: $(TEXFILES) $(PYPLOTS_PDFS_RENDERED) $(PYPLOTS_PNGS_RENDERED)
	pdflatex -draftmode -shell-escape paper.tex
	bibtex paper.aux
	pdflatex -draftmode -shell-escape paper.tex
	pdflatex -shell-escape paper.tex

paper-grayscale.pdf: paper.pdf
	gs -sDEVICE=pdfwrite -sProcessColorModel=DeviceGray \
	   -sColorConversionStrategy=Gray -dOverrideICC \
	   -o "paper-grayscale.pdf" -f "paper.pdf"

grayscale: paper-grayscale.pdf

view: paper.pdf
	open paper.pdf

clean:
	rm -fv *.pdf *.aux *.bbl *.blg *.log *.out $(PYPLOTS_PDFS_RENDERED)

clean-all: clean
	rm -fv $(PYPLOTS_PNGS_RENDERED)
