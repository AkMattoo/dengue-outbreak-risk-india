.PHONY: all pop coords prep climate features train figures test clean

all: prep features train figures

pop:        ## Census 2011 -> data/raw/state_population.csv
	python -m src.prepare_population

coords:     ## derive state centroids from EpiClim -> data/raw/state_coords.csv
	python -m src.derive_state_coords

prep:       ## EpiClim outbreak records -> complete state x month panel
	python -m src.epiclim_prep

climate:    ## optional: Open-Meteo monthly climate (needs internet, no API key)
	python -m src.fetch_climate

features:
	python -m src.features

train:
	python -m src.train

figures:
	python -m src.visualize

test:
	python -m pytest tests/ -q

clean:
	rm -f data/interim/* data/processed/*.csv models/*.pkl reports/figures/*.png
