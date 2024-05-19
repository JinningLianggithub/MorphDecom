# MorphDecom

## Overview
"MorphDecom" is a morphological decomposition method introduced by Liang, Jiang et al. (2024), building on earlier work by Zana et al. (2022). This innovative method combines threshold-based decomposition for elliptical components with a machine learning approach for disky components. It dynamically adjusts binding energy and circularity thresholds based on the specific galaxy system under analysis. If you find this method helpful to you and want to use it in your research, please cite Liang, Jiang et al. (2024). Please feel free to contact the authors Jinning Liang (jinning.liang@durham.ac.uk) and Fangzhou Jiang (fangzhou.jiang@pku.edu.cn), if you have any question.

## Installation
Clone the repository using the following command:

`git clone https://github.com/JinningLianggithub/MorphDecom.git`


## Modules
### `decomposition_method.py`
- Functions to load data from TNG50
- Calculate dynamical quantities
- Re-estimate gravitational potential
- Find binding energy and circularity thresholds
- Decompose galaxies into their constituent components

### `config.py`
- Contains global variables and user controls including galaxy ID ```ID``` 

### `snap_z_a.txt`
- Provides a lookup table for matching snapshots to redshift and scale factor in TNG50

## Dependencies
Install the following libraries and packages to ensure the software functions correctly:
- numpy
- scida
- scipy
- pytreegrav
- intersect
- scikit-learn
- matplotlib

## Usage Guide
### Configuration
Before starting the decomposition process, configure the following parameters:
1. Path to your simulation data
2. Cosmological parameters
3. ID and snapshot of the galaxy for decomposition

Example configuration script:
```python
import numpy as np
import config as cfg
import decomposition_method as de

# Load and set configuration from file
timetable = np.loadtxt("./snap_z_a.txt")
snaplist = timetable.T[0]
scalefactorlist = timetable.T[1]
zlist = timetable.T[2]

# Set the configuration parameters
cfg.path = "/media/31TB1/TNG50-1/output"
cfg.snap = 99
cfg.h = il.groupcat.loadHeader(cfg.path, cfg.snap)["HubbleParam"]
cfg.z = zlist[cfg.snap]
cfg.c = scalefactorlist[cfg.snap]
cfg.G = 4.4985e-06
cfg.Lbox = np.array([35000*cfg.c/cfg.h, 35000*cfg.c/cfg.h, 35000*cfg.c/cfg.h])
ID = 516101
```

### Decompose the galaxy
After setting the configurations, run the decomposition process with:
```python
jzojc_s, jpojc_s, eb_s, pos_s, vel_s, mass_s, VirialRatio_s, profile_s = de.get_kinematics_archeology(ID)
Ecut = de.get_Ecut(eb_s, mass_s)
eta_cut = de.get_etacut(jzojc_s, jpojc_s, eb_s, mass_s, Ecut, smoothing=True, sigma=1)
decomposition = de.assign_label_Zana_fixed(eb_s, jzojc_s, mass_s, Ecut, eta_cut)
```
Here, "decomposition" saves the labels for stellar particles (bulge=1, halo=2, thin disk=3, thick disk=4)


### output
The following is a sample output for a decomposed galaxy:
```
data loaded
galaxy rotated
phi, T calculated
jc calculated
kinematics calculated
stellar particles selected
density profile calculated --- all done
Ecut = -0.6770199113338333
nbins = 401
Etacut = 0.796667590832109
```

## More usage
Here, we only calculate dynamics for different galactic components, one can also calculate more quantities, e.g., chemical abundance and age for components by editing `decomposition_method.get_kinematics_archeology`
