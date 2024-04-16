# MorphDecom

A new morphological decomposition method introduced in Liang, Jiang et al. (2024), extended from Zana et al. (2022). This method adapts binding energy threshold-based method for decomposing elliptical components from Zana et al. (2022) combining with machine learning-based method for decomposing disky components. In this method, we can obtain running threshold for binding energy and circularity depending on the galaxy system.

- Installation
`git clone https://github.com/JinningLianggithub/MorphDecom.git` to clone the repository

- Modules
  
decomposition_method.py: a series of functions for loading data from TNG50, calculating dynamical quantities, re-estimating gravitational potential, finding binding energy threshold and circularity threshold and decomposing galaxies

config.py: global variables and user controls including the path of your simulation and cosmology parameters that you use

snap_z_a.txt: a table matching snapshot with redshift and scale factor in TNG50

- Dependent libraries and packages
  
numpy, illustris_python, scipy, pytreegrav, intersect, scikit-learn


- Guidence
  
Before decompose galaxies, one must set the path of your simulation, cosmology parameters, the ID and snapshot of the galaxy that you want to decompose

`
import numpy as np
import config as cfg
import decomposition_method as de


timetabel=np.loadtxt("./snap_z_a.txt")
snaplist=timetabel.T[0]
scalefactorlist=timetabel.T[1]
zlist=timetabel.T[2]


cfg.path="/media/31TB1/TNG50-1/output"
cfg.snap=99
cfg.h=il.groupcat.loadHeader(cfg.path,cfg.snap)["HubbleParam"]
cfg.z=zlist[cfg.snap]
cfg.c=scalefactorlist[cfg.snap]
cfg.G=4.4985e-06
cfg.Lbox=np.array([35000*cfg.c/cfg.h,35000*cfg.c/cfg.h,35000*cfg.c/cfg.h])
ID=516101`

This define the path, snapshot, hubble constant "h", redshift "z", scale factor "c", gravitational constant "G", the box length "Lbox", the ID of galaxy at this redshift of your interest. With these defined, one can decompose the galaxy
`
jzojc_s, jpojc_s, eb_s, pos_s, vel_s, mass_s, VirialRatio_s,profile_s=de.get_kinematics_archeology(ID)
Ecut=de.get_Ecut(eb_s,mass_s)
eta_cut=de.get_etacut(jzojc_s,jpojc_s,eb_s,mass_s,Ecut,smoothing=True,sigma=1)
decomposition=de.assign_label_Zana_fixed(eb_s,jzojc_s,mass_s,Ecut,eta_cut)
`
This will return the circularity (jzojc_s), polarity (jpojc_s), binding energy (eb_s), phase-space coordinates and masses (pos_s, vel_s, mass_s) of the stellar particles within 5 times half stellar mass radius in the galaxy and virial ratio ( VirialRatio_s), 3D density profile (profile_s), binding energy threshold (Ecut), circularity threshold (eta_cut) of the galaxy. Finally, all stellar particles will be assigned as bulge, halo, thin disc and thick disc (1=bulge, 2=halo, 3=thin disc, 4=thick disc).

For example, the output for this galaxy will be
`
data loaded
galaxy rotated
phi,T calculated
jc calculated
kinematics calculated
stellar particles selected 
density profile calculated --- all done
Ecut =  -0.6770199113338333
nbins =  401
Etacut =  0.796667590832109
`




