# MorphDecom

## Overview
"MorphDecom" is a morphological decomposition method introduced by Liang, Jiang et al. (2024), building on earlier work by Zana et al. (2022). This innovative method combines threshold-based decomposition for elliptical components with a machine learning approach for disky components. It dynamically adjusts binding energy and circularity thresholds based on the specific galaxy system under analysis. If you find this method helpful to you and want to use it in your research, please cite Liang, Jiang et al. (2024). Please feel free to contact the authors Jinning Liang (jinning.liang@durham.ac.uk) and Fangzhou Jiang (fangzhou.jiang@pku.edu.cn), if you have any question.

## Installation
Clone the repository using the following command:

`git clone https://github.com/JinningLianggithub/MorphDecom.git`


## Modules
### `decomposition.py`
- Functions to use data from loaded simulation to calculate dynamical quantities ```decomposition.get_kinematics```
- Find binding energy ```decomposition.get_Ecut```
- Find circularity thresholds ```decomposition.get_etacut```
- Decompose galaxies into their constituent components ```decomposition.assign_label```
- Plot the face-on and edge-on image for different components ```decomposition.image_plot```

### `config.py`
- Contains global variables and user controls including galaxy (Subfind) ID ```ID```, snapshot of the simulation ```snap```, hubble constant "little h" ```h```, redshift of the simulation ```z```, scale factor of the simulation ```c```, total matter density of the simulation at redshift 0 ```Om0```, baryonic matter density of the simulation at redshift 0 ```Omb```, dark energy desity of the simulation at redshift 0 ```OmL```, dark matter mass of the simulation as unit of 1e10Msun/h ```DMmass```, and gravitational solfening length of the simulation at this redshift ```epsilon```

### `snap_z_a.txt`
- Provides a lookup table for matching snapshots to redshift and scale factor in TNG50. This is only for converting redshift and scale factor from snapshot. One can use the table from other simulation or just input the redshift and scale factor manually

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
First, one needs to download the particle data and obtain the information for the host subhalo. Here we assume that we already get the data and information. Full example including download TNG50 data is provided in ```example.ipynb```.

To load the particle data, one can simply use ```h5py``` to read hdf5 file. Here we use powerful tool ```scida``` to load the data. And we construct two dictionaries saving subhalo information and particle data. For subhalo dictionary, it contains its position, velocity, half-stellar-mass radius, the number of gas, dark matter and stellar particles. For particle disctionary, it contains the coordinates, velocities, masses for gas, dark matter and stellar particles. One can also include the formation time for stellar particles to exlucde wind particles. All units must be physical rather than comoving. Here, we take kpc for length unit, kpc/Gyr for velocity unit and Msun for mass unit. This could be done by ```scida```. 

After constructing the subhalo and particle dictionary, one can calculate dynamical quantities using ```decomposition.get_kinematics```. These quantities can be further used in finding energy and circularity threshold in ```decomposition.get_Ecut``` and  ```decomposition.get_etacut```.

Finally, one can use dynamical quantities and two thresholds to do decomposition and obtain a decomposition mask by ```decomposition.assign_label```


## Decomposition image
Using the plotting function ```decomposition.image_plot```, one can generate the face-on and edge-on images of different components in this example galaxy
<p align="center">
  <img src="https://github.com/JinningLianggithub/MorphDecom/blob/main/snap99_516101.png" width=100% height=100%>
</p>

## More usage
After doing decomposition for all stellar particles, one gets a decomposition mask for them. Next, one can select bulge, halo, thin disc and thick particles can calculate other quantities for them, e.g. chemical abundance, age, etc.
