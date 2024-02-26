#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 11:42:46 2024

@author: Jinning Liang @ICC, Durham; Fangzhou Jiang @KIAA, Peking University

"""


#------Package import------
#One can change the path for your simulation
#Some package might be useless


import numpy as np

#TNG loading
import sys
sys.path.insert(0, "C:/Users/97018/Desktop/Galaxy/Code/TNG50-1/illustris_python-master/")
import illustris_python as il
import h5py


#Personal Ploting setting
#One can change
import matplotlib
import matplotlib.pyplot as plt
plt.style.use(["default"])
matplotlib.rc( 'lines', linewidth=3 )
matplotlib.rc( 'font', family='monospace', weight='normal', size=16 )
c_frame = (0,0,0,.8)
for tick in 'xtick', 'ytick':
    matplotlib.rc( tick+'.major', width=2, size=8)
    matplotlib.rc( tick+'.minor', width=1.5, size=4, visible=True )
    matplotlib.rc( tick, color=c_frame, labelsize=15, direction='in' )
matplotlib.rc( 'xtick', top=True )
matplotlib.rc( 'ytick', right=True )
matplotlib.rc( 'axes', linewidth=2.5, edgecolor=c_frame, labelweight='normal' )
matplotlib.rc( 'grid', color=c_frame )
matplotlib.rc( 'patch', edgecolor=c_frame )
from matplotlib.ticker import MultipleLocator, LinearLocator, AutoMinorLocator,StrMethodFormatter



#Integration and interpolation
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
from scipy import ndimage



#Cosmology
from astropy.cosmology import FlatLambdaCDM


#Potential estimation
from pytreegrav.frontend import ConstructTree, Potential,PotentialTarget,Accel,AccelTarget


#Algorithms for decomposing discs
from sklearn.mixture import GaussianMixture as GMM
from intersect import intersection



#------Change the path for subhalo and particle data for your need------
basePath100 = "G:/TNG100/output"
basePath50 = 'G:/TNG50/output'
TNG50=h5py.File('G:/TNG50/postprocessing/simulation.hdf5file/simulation.hdf5', 'r')




#------Define the snapshot for the galaxy population------
snap=99

#------Define cosmology parameters in TNG50------
#Hubble constant 'little h' in units of [100 km/s/Mpc]
h=il.groupcat.loadHeader(basePath50,snap)["HubbleParam"]
#Scale factor c=1/(1+z)
c=il.groupcat.loadHeader(basePath50,snap)["Time"]
#Box size in TNG50 in unit of kpc
Lbox=35000*c/h
#Gravitational constant in unif of [kpc^3 Gyr^-2 Msun^-1]
G=4.4985e-06
#Hubble constant in units of [km/s/Mpc]
H0=100*h
#Matter density at this snapshot
Om0=il.groupcat.loadHeader(basePath50,snap)['Omega0']
#Dark energy density at this snapshot
OmL=il.groupcat.loadHeader(basePath50,snap)['OmegaLambda']
#Baryonic matter density at this snapshot
Omb=0.0486
#Define cosmology model using astropy
cosmo=FlatLambdaCDM(H0=H0, Om0=Om0,Ob0=Omb)
#Get the age of the universe in this snapshot
universe_age=cosmo.age(0).to_value()




#Transformation between redshift and snapshot
redshift=[]
for i in range(100):
    redshift.append(il.groupcat.loadHeader(basePath100,i)["Redshift"])
redshift=np.array(redshift)



#------Auxcilary functions------
def gsoft(z):
    """
    Compute the softening length epsilon [kpc] (this is for TNG50)
    
    Parameters
    ----------
    z : array_like or float
        The redshift for calculating softening length
        
    Returns
    ----------
    epsilon : array_like or float
        softening length epsilon at this redshift [kpc]
    """

    if z<=1:
        epsilon=0.288
    elif z>1:
        epsilon=(0.288*2)/(1+z)
    return epsilon
    
    
    

def kpcoGyr_to_kmos(x):
    """
    Transform the unit of velocity from [kpc/Gyr] to [km/s]
    
    Parameters
    ----------
    x : array_like or float
        Velocity [kpc/Gyr]
        
    Returns
    ----------
    xp : array_like or float
        Velocity [km/s]
    """
    xp=0.9778139490713819*x
    return xp
def kmos_to_kpcoGyr(x):
    """
    Transform the unit of velocity from [km/s] to [kpc/Gyr]
    
    Parameters
    ----------
    x : array_like or float
        Velocity [km/s]
        
    Returns
    ----------
    xp : array_like or float
        Velocity [kpc/Gyr]
    """
    return 1.02268944*x

def smooth_curve(x,y,sigma=1):
    """
    Function for smoothing curves
    
    Parameters
    ----------
    x : array_like
        array of shape (N,) storing x-axis of the curve
    y : array_like
        array of shape (N,) storing y-axis of the curve
    sigma: float
        smoothing level, higher sigma means more smoothing
        
    Returns
    ----------
    x_g1d : array_like
        array of shape (N,) storing smoothed x-axis of the curve
    y_g1d : array_like
        array of shape (N,) storing smoothed y-axis of the curve
    """

    # convert both to arrays
    x_sm = np.array(x)
    y_sm = np.array(y)

    
    x_g1d = ndimage.gaussian_filter1d(x_sm, sigma)
    
    if len(y_sm.shape)>1:
        y_g1d=[]
        for i in y_sm:
            y_g1d.append(ndimage.gaussian_filter1d(i, sigma))
        y_g1d=np.array(y_g1d)
    
    else:
        y_g1d = ndimage.gaussian_filter1d(y_sm, sigma)
    
    return x_g1d,y_g1d

def format_particles(coords,center, Lbox=[Lbox,Lbox,Lbox]):
    """
    center the particle coordinates on (0,0,0) and account for PBCs
    
    Parameters
    ----------
    center : array_like
        array of shape (3,) storing the coordinates of the
        center of the particle distribution [kpc]
    coords :  array_like
        array of shape (nptcls, 3) storing the coordinates of the
        particles [kpc]
    Lbox : array_like
        length 3-array giving the box size in each dimension [kpc]
    Returns
    ----------
    coords : numpy.array
        array of shape (nptcls, 3) storing the centered particle 
        coordinates [kpc]
    """

    dx = coords[:,0] - center[0]
    dy = coords[:,1] - center[1]
    dz = coords[:,2] - center[2]

    # x-coordinate
    mask = (dx > Lbox[0]/2.0)
    dx[mask] = dx[mask] - Lbox[0]
    mask = (dx < -Lbox[0]/2.0)
    dx[mask] = dx[mask] + Lbox[0]

    # y-coordinate
    mask = (dy > Lbox[1]/2.0)
    dy[mask] = dy[mask] - Lbox[1]
    mask = (dy < -Lbox[1]/2.0)
    dy[mask] = dy[mask] + Lbox[1]

    # z-coordinate
    mask = (dz > Lbox[2]/2.0)
    dz[mask] = dz[mask] - Lbox[2]
    mask = (dz < -Lbox[2]/2.0)
    dz[mask] = dz[mask] + Lbox[2]

    # format coordinates
    coords = np.vstack((dx,dy,dz)).T

    return coords

def M(coord,mass,R):
    """
    Calculate the enclosed mass (assuming spherical symmetry)
    
    Parameters
    ----------
    coord : array_like
        array of shape (N,3) storing the coordinates of the particle [kpc]
    mass : array_like
        array of shape (N) storing the mass of the particle [Msun]
    R : float
        the radius for calculating enclosed mass [kpc]
        
        
    Returns
    ----------
    enclosed mass : float
        the enclosed mass within radius R [Msun]
    """
    r=np.sqrt(np.sum(coord**2,axis=1))
    msk = r < R
    m = mass[msk]
    
    return m.sum()



def align_coordinates_with_angular_momentum(coordinates, j_direc):
    """
    Align coordinates of stellar particles, which makes their z-axis with the
    direction of total angularmomentum
    
    Parameters
    ----------
    coordinates : array_like
        array of shape (N,3) storing the coordinates of the particle [kpc]
    j_direc :  unit vector of total angular momentum [kpc^2/Gyr]
        
        
    Returns
    ----------
    transformed_coordinates : : array_like
        array of shape (N,3) storing the aligned coordinates of the particle [kpc]
    """
    
    # Normalize the total angular momentum to get the new z-axis
    z_axis = j_direc
    
    # Calculate the other two axes of the new coordinate system using cross products
    x_axis = np.cross([0, 0, 1], z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    
    # Construct the transformation matrix from the original coordinate system to the new coordinate system
    transformation_matrix = np.vstack((x_axis, y_axis, z_axis)).T
    
    # Transform the coordinates to the new coordinate system
    transformed_coordinates = np.dot(coordinates, transformation_matrix)
    
    return transformed_coordinates




#------Decomposition functions------
def get_kinematics_archeology(ID,snap=snap,RmaxoRhalf=5):
    """
    Calculate the phase-space coordinates and other dynamical quantites
    for galaxy with specific ID and snapshot
    One can add more quantities into consideration
    
    Parameters
    ----------
    ID : ID of the galaxy in simulation
    snap : Snap of the galaxy in simulation
    RmaxoRhalf : The maximal distance for stellar particles in the calculation,
                 which scaled by half-stellar-mass radius (Default value: 5)
    
    Returns
    ----------
    jzojc_s : array_like
        array of shape (N,) storing the circularity (j_z/j_c) of the
        selected stellar particles
    jpojc_s : array_like
        array of shape (N,) storing the polarity (j_p/j_c) of the
        selected stellar particles
    eb_s : array_like
        array of shape (N,) storing the binding energy (T+U) scaled by 
        absolute minimum of the selected stellar particles
    pos_s : array_like
        array of shape (N,3) storing the re-centered position of the 
        selected stellar particles [kpc]
    vel_s : array_like
        array of shape (N,3) storing the velocity of the 
        selected stellar particles [kpc/Gyr]
    mass_s : array_like
        array of shape (N,) storing the stellar mass of 
        the selected stellar particles [Msun]
    age_s : array_like
        array of shape (N,) storing the age of the selected 
        stellar particles [Gyr]
    Z_s : array_like
        array of shape (N,) storing the metallicity (Z/0.0127) of 
        the selected stellar particles [Zsun]
    TUratio_s : array_like
        array of shape (N,) storing the virial ratio (T/U) of 
        the selected stellar particles
    profile : array_like
        array of shape (25,25) storing the stellar density profile of 
        this subhalo
    """
    
    
    
    c=il.groupcat.loadHeader(basePath50,snap)["Time"]
    
    #-----------load data-----------
    subhalo=il.groupcat.loadSingle(basePath50,snap,subhaloID=ID)
    #stellar particles
    stellar_particles=il.snapshot.loadSubhalo(basePath50,snap,ID,"stars")
    #dm particles
    dm_particles=il.snapshot.loadSubhalo(basePath50,snap,ID,"dm")
    #gas particles
    gas_particles=il.snapshot.loadSubhalo(basePath50,snap,ID,"gas")
    
    #number of dm particles
    dm_count=dm_particles["count"]
    #number of stellar particles
    star_count=stellar_particles["count"]
    #number of gas particles
    gas_count=gas_particles["count"]
    
    #obtain galaxy position [kpc]
    galaxy_center_coord = subhalo["SubhaloPos"]*c/h
    #obtain galaxy velocity [km/s]
    galaxy_center_vel = subhalo["SubhaloVel"]
    #obtain half-stellar-mass radius [kpc]
    hmr = subhalo["SubhaloHalfmassRadType"].T[4]*c/h
    
    #obtain re-centered position [kpc]
    dm_pos = format_particles(dm_particles["Coordinates"]*c/h,galaxy_center_coord)
    #obtain dm velocity [km/s]
    dm_vel = dm_particles["Velocities"]*np.sqrt(c)-galaxy_center_vel
    #transform the unit of dm velocity to [kpc/Gyr]
    dm_vel = kmos_to_kpcoGyr(dm_vel)
    #obtain dm mass
    dm_mass = np.repeat(3.07367708626464e-05,dm_particles['count'])*1e10/h
    
    #if there is gas in this subhalo, obtain position, velocity and mass for
    #gas using similar way
    if gas_count!=0:
        gas_pos = format_particles(gas_particles["Coordinates"]*c/h,galaxy_center_coord)
        gas_vel = gas_particles["Velocities"]*np.sqrt(c)-galaxy_center_vel
        gas_vel = kmos_to_kpcoGyr(gas_vel)
        gas_mass = gas_particles["Masses"]*1e10/h

    #obtain position, velocity and mass for stars using similar way
    stars_pos = format_particles(stellar_particles["Coordinates"]*c/h,galaxy_center_coord)
    stars_vel = stellar_particles["Velocities"]*np.sqrt(c)-galaxy_center_vel
    stars_vel = kmos_to_kpcoGyr(stars_vel)
    stars_mass = stellar_particles["Masses"]*1e10/h
    
    #obtain position, velocity and mass for all particles
    if gas_count!=0:
        pos=np.array(list(dm_pos)+list(gas_pos)+list(stars_pos))
        vel=np.array(list(dm_vel)+list(gas_vel)+list(stars_vel))
        mass=np.array(list(dm_mass)+list(gas_mass)+list(stars_mass))
    else:
        pos=np.array(list(dm_pos)+list(stars_pos))
        vel=np.array(list(dm_vel)+list(stars_vel))
        mass=np.array(list(dm_mass)+list(stars_mass))
    
    #get the index of first stellar particles
    star_index=dm_count+gas_count
    print("data loaded")

    
    #-----------define softening length [kpc]-----------
    eps = gsoft(redshift[snap])*np.ones(len(mass))
    
    #-----------Region where to compute the-----------
    #-----------angular momentum to align the galaxy-----------
    size_ang = RmaxoRhalf*hmr
    
    #-----------Rotate the galaxy in order to align its angular momentum with the z-axis-----------
    disk_total_j_region = np.sum(np.cross(stars_pos, stars_vel * stars_mass[:, np.newaxis])[np.where(np.sqrt(np.sum(stars_pos**2,axis=1))<=size_ang)], axis=0)/np.sum(stars_mass[np.where(np.sqrt(np.sum(stars_pos**2,axis=1))<=size_ang)])
    z_axis = disk_total_j_region / np.linalg.norm(disk_total_j_region)
    pos=align_coordinates_with_angular_momentum(pos, z_axis)
    print("galaxy rotated")
    
    #-----------calculate potential and kinematic energy-----------
    Phi_all, Phi0, kdtree = get_phi(pos,mass,eps)
    ke = np.sum(vel**2,axis=1)/2
    te = ke + Phi_all
    #obtain virial ratio
    TUratio_s=2*np.sum(ke)/np.sum(Phi_all)
    print("phi,T calculated")

    
    #-----------calculate potential energy, circular velocity and angular momentum in midplane-----------
    r=np.sqrt(np.sum(stars_pos**2,axis=1))
    #binning start from 0.99*minimal radius to 1.01*maximal radius
    M_r = 1.01*np.max(r)
    m_r = 0.99*np.min(r[r>0])
    pbins = np.logspace(np.log10(m_r), np.log10(M_r), 100, endpoint=True)
    rxy_target=0.5 * (pbins[:-1] + pbins[1:])
    #calculate circular velocity and potential in the mid-plane
    mid_vels,mid_pots= get_midplane_pot_vc(rxy_target,pos,mass,eps,kdtree)
    mid_pots=mid_pots-Phi0
    #calculate maximal angular momentum in these position
    j_circ=rxy_target*mid_vels
    #calculate binding energy in these position
    E_circ=0.5 * (mid_vels ** 2) + mid_pots
    
    #obtain interpolation function jc(E)
    j_from_E = interp1d(np.log10(-E_circ)[::-1], np.log10(j_circ)[::-1], fill_value='extrapolate', bounds_error=False)
    j_circ_star = 10**j_from_E(np.log10(-te[star_index:]))
    #Force nearly unbound particles into the spheroid either by setting their circular angular momentum to infinity
    j_circ_star[np.where(te[star_index:] > E_circ.max())] = np.inf
    #Handle those few particles with E<min(E_circ) for numerical fluctuations
    j_circ_star[np.where(te[star_index:] < E_circ.min())] = j_circ[0]
    print("jc calculated")

    #-----------calculate jzojc, jpojc, eb for stars-----------
    j_star = np.cross(stars_pos, stars_vel)
    jz_star = np.dot(j_star,z_axis)
    jz_by_jcirc_star = jz_star / j_circ_star
    jp_by_jcirc_star = (np.sqrt(np.sum(j_star**2,axis=1)-jz_star**2)) / j_circ_star
    eb_star=(te/(np.abs(te).max()))[star_index:]
    print("kinematics calculated")
    
    #-----------calculate age and metallicity for stars-----------
    birth_a=stellar_particles["GFM_StellarFormationTime"]
    birth_z=1/birth_a-1
    age=cosmo.lookback_time(birth_z).to_value()
    #FeoH_sun=np.log10(55.845/1.0079) + (7.5 - 12)
    #FeoH=np.log10(stellar_particles["GFM_Metals"].T[8]/stellar_particles["GFM_Metals"].T[0])-FeoH_sun
    Z=stellar_particles["GFM_Metallicity"]/0.0127
    print("archeology calculated")
    
    #-----------select stars-----------
    skill=np.where((r<=RmaxoRhalf*hmr)&(r>eps[0])&(stellar_particles["GFM_StellarFormationTime"]>0)&
            (np.abs(jz_by_jcirc_star)<=1.5)&(eb_star<=0)&(jp_by_jcirc_star<=1.5))
    jzojc_s=jz_by_jcirc_star[skill]
    jpojc_s=jp_by_jcirc_star[skill]
    eb_s=eb_star[skill]
    pos_s=stars_pos[skill]
    vel_s=stars_vel[skill]
    mass_s=stars_mass[skill]
    age_s=age[skill]
    #FeoH_s=FeoH[skill]
    Z_s=Z[skill]
    print("stellar particles selected ")
    
    #use overlapping binning to obtain stellar density profile
    bincenter=np.linspace(1.01*np.sqrt(np.sum(pos_s**2,axis=1)).min(),0.99*np.sqrt(np.sum(pos_s**2,axis=1)).max(),25)
    bin_halfwidth=2.5*(bincenter[1]-bincenter[0])
    bin_left=bincenter-bin_halfwidth
    bin_right=bincenter+bin_halfwidth

    density=[]
    for rr in range(len(bincenter)):
        delta_mass=M(pos, mass, bin_right[rr])-M(pos, mass, bin_left[rr])
        density.append(delta_mass/ 4 / np.pi / (bincenter[rr]) ** 2 / (bin_right[rr] - bin_left[rr]))
    profile=np.array([density,bincenter])
    print("density profile calculated --- all done")
    
    

    return jzojc_s, jpojc_s, eb_s, pos_s, vel_s, mass_s, age_s, Z_s,TUratio_s,profile

def get_phi(pos,mass,eps,theta=0.5):
    """
    Calculate gravitational potential given all considered particles
    
    Parameters
    ----------
    pos : array_like
        array of shape (N,3) storing the position of 
        the selected particles [kpc]
    mass : array_like
        array of shape (N,) storing the mass of 
        the selected particles [Msun]
    eps : array_like
        array of shape (N,) storing the softening length of 
        the selected particles [kpc]
    theta : float
        opening angle in the tree (default value: 0.5)
    Returns
    ----------
    kdtree : object
        tree for calculating gravity
    Phi_all : array_like
        array of shape (N,) storing the gravitational potential of 
        the selected particles [kpc^2 Gyr^-2]
    Phi0 : float
        the gravitational potential of the farthest particle, i.e. the minimum
    """
    

    kdtree=ConstructTree(pos, mass, softening=eps)
    Phi_all=PotentialTarget(pos, pos, mass,softening_target=eps,softening_source=eps, G=G, theta=theta, tree=kdtree,parallel=True, method='tree')
    Phi0=Phi_all[np.where(np.sum(pos**2,axis=1)==np.sum(pos**2,axis=1).max())][0]
    Phi_all=Phi_all-Phi0
    
    return Phi_all, Phi0, kdtree
    




def get_midplane_pot_vc(rxy_target,pos,mass,eps,kdtree):
    """
    Calculate gravitational potential and circular velocity for mid-plane
    
    Parameters
    ----------
    rxy_target : array_like
        array of shape (N,) storing the radial position for sampling [kpc]
    pos : array_like
        array of shape (N,3) storing the position of 
        the selected particles [kpc]
    mass : array_like
        array of shape (N,) storing the mass of 
        the selected particles [Msun]
    eps : array_like
        array of shape (N,) storing the softening length of 
        the selected particles [kpc]
    kdtree : object
        tree for calculating gravity
        
    Returns
    ----------
    vels : array_like
        array of shape (N,) storing the circular velocity of 
        mid-plane [kpc/Gyr]
    pots : array_like
        array of shape (N,) storing the gravitational potential of 
        mid-plane [kpc^2 Gyr^-2]
    """
    
    #Assuming axisymmtry and calulate potential and velocity by averaging
    #four points with equal sampling distance in the mid-plane 
    rs = np.array([position for r in rxy_target for position in [(r, 0, 0), (0, r, 0), (-r, 0, 0), (0, -r, 0)]], dtype=float)
    potential=PotentialTarget(rs, pos, mass,softening_target=eps[0]*np.ones(len(rs)),softening_source=eps, G=G, theta=0.5, tree=kdtree,parallel=True, method='tree')
    accel = AccelTarget(rs, pos, mass,softening_target=eps[0]*np.ones(len(rs)),softening_source=eps, G=G, theta=0.5, tree=kdtree,parallel=True, method='tree')

    pots = []

    i = 0

    for r in rxy_target:
        # Do four samples
        pot = []
        for pos in [(r, 0, 0), (0, r, 0), (-r, 0, 0), (0, -r, 0)]:
            pot.append(potential[i])
            i = i + 1
        #averaging
        pots.append(np.mean(pot))



    vels = []

    i = 0
    for r in rxy_target:
        r_acc_r = []
        for pos in [(r, 0, 0), (0, r, 0), (-r, 0, 0), (0, -r, 0)]:
            r_acc_r.append(np.dot(-accel[i, :], pos))
            i = i + 1

        vel2 = np.mean(r_acc_r)
        if vel2 > 0:
            vel = vel2**0.5
        else:
            vel = 0
        #averaging
        vels.append(vel)
    
    return np.array(vels), np.array(pots)




def get_Ecut(eb,masses,nbins = 25,M_bin=400,m_bin=80,toll=1.5,shrink=2,Mmin = 0.05,Emin = -0.9):
    """
    Calculate energy threshold for subhalo using method from Zana et al. 2022
    
    Parameters
    ----------
    eb : array_like
        array of shape (N,) storing the scaled binding energy of 
        the selected particles
    masses : array_like
        array of shape (N,) storing the mass of 
        the selected particles [Msun]

        
    Returns
    ----------
    Ecut : float
        energy threshold found by algorithm
    """
    
    
    
    
    if len(eb)<100:
        print('Ecut = ', 0)
        return 0,np.array([0])

    #Fix the number of bin as a function of Npart
    NbinMax = max(min(int(0.5*np.sqrt(len(eb))), M_bin), m_bin)


    #This is to exclude the outer tail of bound particles
    M_E = np.quantile(eb, 0.9)
    m_E = np.min(eb)
    Ecut, E_val = FindMin(eb, m_E, M_E, nbins)
    

    #If no minimum is found or the only minimum is too close to -1 (Maybe a GC?)
    if len(Ecut)==0 or (len(Ecut)==1 and Ecut<Emin):


        M_E = np.max(eb)
        Ecut, E_val = FindMin(eb, m_E, M_E, nbins)
        Ecut = Ecut
    #If one or none minima are found
    if len(Ecut)<=1:
        D = (M_E-m_E)/float(nbins)
        #Avoid the following loop
        nbins = NbinMax+1
    else:
        D = (M_E-m_E)/float(nbins)
        lb = Ecut-(toll*D)
        rb = Ecut+(toll*D)

    #-------
    

    while nbins < NbinMax:
        nbins = shrink*nbins
        D = D/shrink
        pos_E_refined, val_refined = FindMin(eb, m_E, M_E, nbins)
        EcutTEMP = []
        E_valTEMP = []
        for i, v in enumerate(E_val):
            pTEMP = pos_E_refined[(pos_E_refined<=rb[i])*(pos_E_refined>=lb[i])]
            vTEMP = val_refined[(pos_E_refined<=rb[i])*(pos_E_refined>=lb[i])]
            if len(pTEMP)>0:
                #A rifened position and value for each original minimum is stored. 
                #The value of the minima is summed to the original ones to avoid strange local minima 
                EcutTEMP.append(pTEMP[np.argmin(vTEMP)])
                E_valTEMP.append(v+np.min(vTEMP))

        Ecut = np.array(EcutTEMP)
        E_val = np.array(E_valTEMP)



        if len(Ecut)<=1:
            break

        lb = Ecut-(toll*D)
        rb = Ecut+(toll*D)
    

    #-------
    
    #If no energy cut is found		
    if len(Ecut)==0:
        Ecut = 0
    else:
        #Try to avoid strange nuclear minima with low mass if there are better alternatives
        rel_filt = [bool((np.sum(masses[eb<E])/np.sum(masses)>=Mmin)+(E>=Emin)) for E in Ecut]
        if len(Ecut[rel_filt])==0:
            Ecut = Ecut[np.argmin(E_val)]
        else:
            Ecut = Ecut[rel_filt][np.argmin(E_val[rel_filt])]
        Ecut = RefineMin(eb, Ecut, D, (M_E-m_E)/NbinMax, shrink)
        

            
    print('Ecut = ', Ecut)
    print('nbins = ', nbins)
    

    
    return Ecut

def FindMin(q, m_E, M_E, nbins):

    """
    It looks for the minima in the distribution of energies q in the interval [m_E; M_E] with nbin bins
    Arguments:
        
    Parameters
    ----------
    q : array_like
        array of shape (N,) storing the scaled binding energy of 
        the selected particles
    m_E : float
        lower bound of the interval where to look for the minima
    M_E : float
        upper bound of the interval where to look for the minima
    nbins : float
        number of bins in the interval used to bin the distribution q
    
    Returns
    ----------
    array with the position of the minima of the distribution q, along with their values (NB: The values do depend on nbins)
    """

    #Minimum number of particles to perform a reliable Jcirc decomposition
    if len(q)>=1e4:
        Npart_min=1000
    elif 1e3<=len(q)<1e4:
        Npart_min=100
    else:
        Npart_min=10
    
    MinPart = max(Npart_min, 0.01*len(q))
    arr = q[(q>=m_E)*(q<=M_E)]
    #Build the histogram
    hist = np.histogram(arr, bins=np.linspace(m_E, M_E, nbins))

    #Evaluate the increment on both sides A
    diff = hist[0][1:]-hist[0][:-1]
    left = diff[:-1]
    right = diff[1:]
    #Find the minima
    id_E = np.where(((left<0)*(right>=0))+((left<=0)*(right>0)))

    #C
    R_part = np.array([np.sum(hist[0][i+1:]) for i in id_E[0]])
    id_E = id_E[0][R_part>MinPart]
    id_E_flag = [True]*len(id_E)
    
    
    #B
    for i, ids in enumerate(id_E):
        if len(hist[0])>ids+3:
            id_E_flag[i] *= hist[0][ids+3]>hist[0][ids+1]
        if ids > 0:
            id_E_flag[i] *= hist[0][ids-1]>hist[0][ids+1]

    id_E = id_E[id_E_flag]

    #if debug:
    #	print('Survived Energies:', hist[1][id_E+1])
    #	from matplotlib import pyplot as plt
    #	plt.bar(hist[1][:-1], hist[0], hist[1][1:]-hist[1][:-1], align='edge')
    #	plt.ylabel('Count')
    #	plt.xlabel('E/|Emax|')
    #	plt.show()		

    #Return the central position of the bins
    return 0.5*(hist[1][id_E+2]+hist[1][id_E+1]), hist[0][id_E+1]


def RefineMin(q, Vmin, D, Dmin, shrink):

    """
    It recursively refines a minimum Vmin of the energy distribution q, within the interval of size D, centred on Vmin. Each time the 
    refinement reduce the interval size of shrink as long as D>Dmin
    
    Parameters
    ----------
    q : array_like
        array of shape (N,) storing the scaled binding energy of 
        the selected particles
    Vmin : float
        value of the minimum (Energy) to refine
    D : float
        initial size of the energy interval around Vmin
    Dmin : float
        shortest interval size allowed
    shrink : float
        factor to reduce the interval size, at each cycle
        
    Returns
    ----------
    Vmin : float
        refined position of Vmin
    """

    if D<=Dmin:
        if len(q)>=1e4:
            coe=0.5
        elif 1e3<=len(q)<1e4:
            coe=2
        else:
            coe=2.5
        m_E = Vmin-coe*D 
        M_E = Vmin+coe*D
        arr = q[(q>=m_E)*(q<=M_E)]
        Vmin = np.median(arr)

    while D > Dmin:
        if len(q)>=1e3:
            coe=1.5
        else:
            coe=4
        m_E = max(Vmin-coe*D, q.min()+D) 
        M_E = Vmin+coe*D
        D = D/shrink
        arr = q[(q>=m_E)*(q<=M_E)]
        hist = np.histogram(arr, bins=np.arange(m_E, M_E, D))
        hist_min=(hist[0][np.where(hist[0]!=0)]).min()
        pid = np.where(hist[0]==hist_min)[0][0]
        #Get the energy as the median within the selected bin
        Vmin = np.median(arr[(arr>=hist[1][pid])*(arr<=(hist[1][pid+1]))])


        #print ('Refining:', Vmin)
        #plt.bar(hist[1][:-1], hist[0], hist[1][1:]-hist[1][:-1], align='edge')
        #plt.show()

    return Vmin





def get_etacut(jzojc,jpojc,eb,mass,Ecut,smoothing=True,sigma=1):
    
    """
    Obtain circularity threshold using method from Liang et al. 2024
    Input circularity, polarity and binding energy of disky particles to GMM
    Only set 2 components to be returned
    The one with smaller circularity distribution is thick disc
    While the other one is thin disc
    
    Parameters
    ----------
    jzojc : array_like
        array of shape (N,) storing the circularity (jz/jc) of 
        the selected particles
    jpojc : array_like
        array of shape (N,) storing the polarity (jp/jc) of 
        the selected particles
    eb : array_like
        array of shape (N,) storing the scaled binding energy of 
        the selected particles
    mass : array_like
        array of shape (N,) storing the mass of 
        the selected particles [Msun]
    Ecut : float
        energy threshold
    smoothing : bool
        if smooth curves to obtain intersection (threshold) (default value: True)
    sigma : float
        smoothing lever (default value: 1)
        
    Returns
    ----------
    eta_cut : float
        circulartiy threshold
    """
    #Separate disky stars based on Zana's method
    temp_label=assign_label_Zana_fixed(eb,jzojc,mass,Ecut,etacut=None)
    #Disky stars' label is assigned as -1
    non_sph=np.where(temp_label==-1)
    
    #GMM setting
    #n_init=10 adapted from Du et al. 2019
    #Separating discs by n_components=2
    aclus = GMM(n_components=2, covariance_type='full', n_init=10)
    data=(np.array([jzojc,jpojc,eb]).T)[non_sph]
    aclus.fit(data)
    GMMlabel = aclus.predict(data)
    
    #GMM label, which is random
    #Here 0 doesn't mean smaller one since it's random
    #Bin circularity for different gaussian components
    hist1 = np.histogram(data.T[0][np.where(GMMlabel==0)], bins='auto')
    hist2 = np.histogram(data.T[0][np.where(GMMlabel==1)], bins='auto')
    
    #Get bin center and density at this bin
    x1=(hist1[1][1:]+hist1[1][:-1])/2
    y1=hist1[0]
    x2=(hist2[1][1:]+hist2[1][:-1])/2
    y2=hist2[0]
    
    
    
    #Get the maximum density and the corresponding bin for two gaussian components
    tempy1=y1[np.where((y1!=y1[0])&(y1!=y1[-1]))]
    tempy2=y2[np.where((y2!=y2[0])&(y2!=y2[-1]))]
    hmax1=x1[np.where(y1==tempy1.max())]
    hmax2=x2[np.where(y2==tempy2.max())]
    
    #Safety check: avoid multiple maximum values, which is rare
    if len(hmax1)>1:
        hmax1=np.mean(hmax1)
    if len(hmax2)>1:
        hmax2=np.mean(hmax2)
    
    
    #Check which corresponding circularity is larger
    #if gaussian components with label 0's corresponding circularity is larger
    #swith their label and binning again
    circ_m1=min(hmax1,hmax2)
    circ_m2=max(hmax1,hmax2)
    
    if not isinstance(circ_m1,float):
        circ_m1=circ_m1[0]
    if not isinstance(circ_m2,float):
        circ_m2=circ_m2[0]
    
    if hmax2==circ_m2:
        pass
    else:
            
        hist1 = np.histogram(data.T[0][np.where(GMMlabel==1)], bins='auto')
        hist2 = np.histogram(data.T[0][np.where(GMMlabel==0)], bins='auto')
        
        x1=(hist1[1][1:]+hist1[1][:-1])/2
        y1=hist1[0]
        x2=(hist2[1][1:]+hist2[1][:-1])/2
        y2=hist2[0]
        
        
        
        
        
    #Smoothing curves and find threshold
    if smoothing==True:
        x1,y1=smooth_curve(x1,y1,sigma=sigma)
        x2,y2=smooth_curve(x2,y2,sigma=sigma)
    
    x, y = intersection(x1, y1, x2, y2)

    
    
    #if there is multiple threshold or no threshold
    #use some criteria to reduce the number of threshold
    #for some cases there might be no threshold, which will return 0
    if len(x)==0:
        min1=np.min(x1)
        min2=np.min(x2)
        max1=np.max(x1)
        max2=np.max(x2)
        
        if (max1 <= circ_m2 and min2 >= circ_m1):
            eta_cut=((min2+max1)/2)
        else:
            eta_cut=0.
            print("strange intersect 0")
    elif len(x)>1:
        x_mask=np.where((x<=circ_m2)&(x>=circ_m1))
        if len(x[x_mask])==1:
            eta_cut=(x[x_mask][0])
        elif len(x[x_mask])>1:
            eta_cut=(np.mean(x[x_mask]))
            print("more than 1 intersect")
        elif len(x[x_mask])==0:
            eta_cut=0.
            print("strange intersect 1")
    else:
        x_mask=np.where((x<=circ_m2)&(x>=circ_m1))
        if len(x[x_mask])==0:
            eta_cut=0.
            print("strange intersect 1")
        else:
            eta_cut=x[0]
        
    
    print('Etacut = ', eta_cut)
        
    return eta_cut



def assign_label_Zana_fixed(eb,jzojc,masses,Ecut,etacut=None):
        
    """
    Using method from Liang et al.2024 to decompose galaxies into bulge,
    halo, thin disc and thick disc
    
    Parameters
    ----------
    jzojc : array_like
        array of shape (N,) storing the circularity (jz/jc) of 
        the selected particles
    jpojc : array_like
        array of shape (N,) storing the polarity (jp/jc) of 
        the selected particles
    eb : array_like
        array of shape (N,) storing the scaled binding energy of 
        the selected particles
    eb : array_like
        array of shape (N,) storing the mass of 
        the selected particles [Msun]
    Ecut : float
        energy threshold
    etacut : float or bool
        circulartiy threshold, if etacut=None, then the method will find one.
        If etacut is a float, the function will use this threshold to do 
        decomposition
        
    Returns
    ----------
    labels_4comp : the label index for selected particles
                    Bulge=1
                    Halo=2
                    Thin Disc=3
                    Thick Disc=3
                If one need to find etacut, or only separate spheroid and disc,
                those particles with label=-1 are disky stars
    """
    #create empty label
    labels_4comp=np.zeros(len(eb))-1
    #separate energy into low energy component and high energy component
    #details for the reason can be seen in Zana et al. 2022 or Liang et al. 2024
    E_low = eb<=Ecut
    E_low_where=np.where(eb<=Ecut)[0]




    #Bulge:
    dist_low = np.histogram(jzojc[E_low], bins=np.arange(np.nanmin(jzojc[E_low]), np.nanmax(jzojc[E_low]), 0.01), weights=masses[E_low])

    if len(dist_low[0])>=4:
        PositiveCirc = jzojc[E_low]>0
        c = 0.5*(dist_low[1][1:]+dist_low[1][:-1])
        #calculate distribution function of circularity
        Bspl = UnivariateSpline(c, dist_low[0], s=0)
        #calculate mirrored circularity distribution function
        yBspl = Bspl(-jzojc[E_low][PositiveCirc])
        #The seed is fixed for reproducibility
        np.random.seed(42)
        #Calculate the probability for particles with circularity larger than 0
        #that will assign to be bulge by the distribution function of negative circularity
        p = yBspl/Bspl(jzojc[E_low][PositiveCirc])
        #[0;1)
        ra = np.random.random(len(yBspl))
        id_pos = np.where(E_low*(jzojc>0))[0]
        #MCMC sampling
        id_b = id_pos[ra<=p]
        #All particles with circularity smaller than 0 are bulge star
        bulge = np.where((eb<=Ecut)*(jzojc<=0))
        
        bulge = np.concatenate((bulge[0], id_b))
        labels_4comp[bulge]=1

    #Similar to bulge but only do that when there is a halo in galaxy, i.e. Ecut<0
    if Ecut<0:
        #Halo
        dist_high = np.histogram(jzojc[~E_low], bins=np.arange(np.nanmin(jzojc[~E_low]), np.nanmax(jzojc[~E_low]), 0.01), weights=masses[~E_low])

        if len(dist_high[0])>=4:
            PositiveCirc = jzojc[~E_low]>0
            c = 0.5*(dist_high[1][1:]+dist_high[1][:-1])
            Hspl = UnivariateSpline(c, dist_high[0], s=0)
            yHspl = Hspl(-jzojc[~E_low][PositiveCirc])

            #Ratio between negative tail and positive part
            p = yHspl/Hspl(jzojc[~E_low][PositiveCirc])
            ra = np.random.random(len(yHspl))
            id_pos = np.where((~E_low) * (jzojc>0))[0]
            id_h = id_pos[ra<=p]

            halo = np.where((eb > Ecut) * (jzojc <= 0))
            halo = np.concatenate((halo[0], id_h))
            labels_4comp[halo]=2
    
    #One need to find threshold to separate disky stars into thin disc and thick disc
    #Otherwise, these disky stars will be labeled as -1
    #One can use standard choice: 0.7
    if etacut==None:
        return labels_4comp
    
    #Disky stars with circularity larger than threshold will be assigned as thin disc
    #Disky stars with circularity smaller than threshold will be assigned as thick disc
    #Requiring there at least 100 particles for thin and thick discs
    #If one of them can't satisfy this, they will be assign as the other disky components
    #If both of them can't satify this, they will be assigned as bulge or halo by their energy
    #One can change or delete this requirement
    else:
        ThinDisk=np.where((labels_4comp!=1)&(labels_4comp!=2)&(jzojc>=etacut))
        ThickDisk=np.where((labels_4comp!=1)&(labels_4comp!=2)&(jzojc<etacut))
        
        if len(ThinDisk[0])<100 and len(ThickDisk[0])<100:
            labels_4comp[np.where((labels_4comp!=1)&(labels_4comp!=2)&(jzojc>=etacut)&(eb > Ecut))]=2
            labels_4comp[np.where((labels_4comp!=1)&(labels_4comp!=2)&(jzojc>=etacut)&(eb <= Ecut))]=1
            
        elif len(ThinDisk[0])<100 and len(ThickDisk[0])>=100:
            labels_4comp[np.where((labels_4comp!=1)&(labels_4comp!=2)&(jzojc>=etacut))]=4
            labels_4comp[np.where((labels_4comp!=1)&(labels_4comp!=2)&(jzojc>=etacut))]=4
        else:
            labels_4comp[ThinDisk]=3
            
            
        if len(ThickDisk[0])<100 and len(ThinDisk[0])<100:
            labels_4comp[np.where((labels_4comp!=1)&(labels_4comp!=2)&(jzojc<etacut)&(eb > Ecut))]=2
            labels_4comp[np.where((labels_4comp!=1)&(labels_4comp!=2)&(jzojc<etacut)&(eb <= Ecut))]=1
        elif len(ThickDisk[0])<100 and len(ThinDisk[0])>=100:
            labels_4comp[np.where((labels_4comp!=1)&(labels_4comp!=2)&(jzojc<etacut))]=3
            labels_4comp[np.where((labels_4comp!=1)&(labels_4comp!=2)&(jzojc<etacut))]=3
        else:
            labels_4comp[ThickDisk]=4
            
            
        #Require there at least 30 particles for bulges and halos
        #If one of them can't satisfy this, they will be assign as the other spheroidal components=
        #One can change or delete this requirement
        Bulge=np.where(labels_4comp==1)
        Halo=np.where(labels_4comp==2)
        if len(Bulge[0])<30 and len(Halo[0])>=30:
            labels_4comp[Bulge]=2
        elif len(Bulge[0])>=30 and len(Halo[0])<30:
            labels_4comp[Halo]=1

        return labels_4comp






















