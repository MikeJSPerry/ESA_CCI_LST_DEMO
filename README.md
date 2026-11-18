# ESA CCI Land Surface Temperature (LST) demo

Short Jupyter notebooks showing how to read, plot and explore data from the ESA Climate Change Initiative (CCI) Land Surface Temperature product using Python and [xarray](https://xarray.dev). No installation needed: launch the notebooks in your browser with Binder.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MikeJSPerry/ESA_CCI_LST_DEMO/HEAD)

## The notebooks

| Notebook | What it covers |
|---|---|
| [example_reader.ipynb](example_reader.ipynb) | Opening an LST file, plotting a global map and histogram, selecting a region and a latitude transect with `.sel`, and exploring land cover classes (maps, distribution, LST by class). |
| [example_uncertainty.ipynb](example_uncertainty.ipynb) | Working with the uncertainty information: total uncertainty and its components, regional maps and histograms, combining components in quadrature, uncertainty along a transect, screening data by uncertainty, area-weighted and zonal means, and the number of clear-sky observations. |

Start with `example_reader.ipynb`.

## The data

The `Data/` folder holds two small monthly Level-3C (L3C) daytime files from MODIS Aqua (product version 3.00), chosen to keep the repository small:

- **0.25° global**, July 2006.
- **0.01° regional subset** (lat 40–45°N, lon 117–122°E), September 2022. This file also includes land cover class, the number of clear-sky observations and their variance.

`ESACCI_LC_Legend_LST_UPDATE.csv` gives the names and colours of the land cover classes.

These files are only samples. For more data, including other sensors, dates and resolutions, see the [ESA Climate Change Initiative](https://climate.esa.int) website.

## Running locally

The notebooks need Python with xarray, matplotlib and netCDF4 (see [requirements.txt](requirements.txt)). Matplotlib 3.10 or newer is required.

```bash
git clone https://github.com/MikeJSPerry/ESA_CCI_LST_DEMO.git
cd ESA_CCI_LST_DEMO
pip install -r requirements.txt jupyterlab
jupyter lab
```

Run Jupyter from the repository folder, because the notebooks use relative paths to `Data/` and `SubRoutines/`.

## Repository layout

```
example_reader.ipynb       basic reading, plotting and land cover
example_uncertainty.ipynb  uncertainty examples
SubRoutines/plotting.py    plotting helpers used by the notebooks
Data/                      sample LST files and land cover legend
requirements.txt           Python dependencies
```
