# Hopfion FMR Analysis

This repository contains a MuMax3 simulation input file and Python scripts for post-processing the dynamic response of a magnetic hopfion, including FFT spectrum analysis and quality factor estimation.

## Contents

- `RFM_Hopfion.mx3.mx3`  
  MuMax3 input file used to simulate the resonant dynamics / ferromagnetic resonance response of a magnetic hopfion.

- `FFT_Analysis.py`  
  Python script for reading `.odt` files exported from MuMax3, extracting a selected magnetization component, computing the FFT cell by cell, and exporting magnitude and phase maps in CSV format.

- `Quality_Factor.py`  
  Python script for calculating the resonance peak frequency, full width at half maximum (FWHM), and quality factor `Q` from a CSV spectrum.

## Description

The workflow of this repository is:

1. Run the MuMax3 simulation using the provided `.mx3` file.
2. Export the time-dependent magnetization data.
3. Use `FFT_Analysis.py` to:
   - read the `.odt` files,
   - extract the selected magnetization component,
   - compute the FFT for each cell,
   - remove the DC component,
   - export FFT magnitude and phase data into CSV files.
4. Use `Quality_Factor.py` to:
   - load a frequency-amplitude CSV file,
   - interpolate the spectrum,
   - identify the resonance peak,
   - calculate the FWHM,
   - compute the quality factor `Q = f0 / Δf`.

## Requirements

The Python scripts require:

- Python 3
- NumPy
- Pandas
- Matplotlib

You can install the required packages with:

```bash
pip install numpy pandas matplotlib
