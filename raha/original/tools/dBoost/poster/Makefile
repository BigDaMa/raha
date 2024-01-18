TEXFILES=$(wildcard *.tex)

all: master.pdf

../graphics/models-plots.pdf:
	cd ../graphics/ && python3 models-plots.py

../graphics/models-plots-crop.pdf: ../graphics/models-plots.pdf
	cd ../graphics/ && pdfcrop models-plots.pdf

../graphics/sensors-crop.pdf: ../graphics/sensors.pdf
	cd ../graphics/ && pdfcrop sensors.pdf

master.pdf: $(TEXFILES) ../graphics/models-plots-crop.pdf
	pdflatex master.tex

clean:
	rm -fv *.pdf *.aux *.log *.out *.nav *.snm *.toc *.vrb
