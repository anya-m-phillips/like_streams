# %% 
import agama
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import Galactocentric, ICRS, CartesianRepresentation, CartesianDifferential, SkyCoord
import gala.coordinates as gc
import gala.dynamics as gd
import gala.potential as gp
from gala.dynamics import mockstream as ms
from gala.units import galactic
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import curve_fit



##### FUNCTIONS
def define_time_cmap():
    colors=["#212738","#4D3041","#79394A","#994250","#B24B54","#C55756","#D26A54","#DC8051","#E5974D","#EDAE49"]
    colors.reverse()
    time_cmap = LinearSegmentedColormap.from_list('time_cmap', colors)
    return time_cmap

def define_simcolors():
    lm_colors = ["#2D3047","#246569","#1B998B","#8DCB87"] # increasing diffuseness
    hm_colors = ["#BB333C", "#E84855","#FF9B71","#FFCC7A"]#,"#FFE57E"] # same; densest to most diffuse. 

    simcolors = [lm_colors[0],hm_colors[0],lm_colors[1],hm_colors[1],lm_colors[2],hm_colors[2],lm_colors[3],hm_colors[3]] * 3
    return lm_colors, hm_colors, simcolors


def reflex_correct(coords, galactocentric_frame=None):
    """
    solar reflex correction. same algorithm as gala.coordinates.reflex_correct,
    but astropy-8-safe: astropy 8 changed Galactocentric.galcen_v_sun from a
    DifferentialAttribute (CartesianDifferential) to a
    CartesianRepresentationAttribute (CartesianRepresentation), so gala 1.11's
    `differentials['s'] + v_sun` returns a representation instead of a
    differential and with_differentials() raises IndexError. normalize v_sun
    back to a CartesianDifferential first.
    """
    c = SkyCoord(coords)
    if galactocentric_frame is None:
        galactocentric_frame = Galactocentric()

    v_sun = galactocentric_frame.galcen_v_sun
    if not isinstance(v_sun, CartesianDifferential):  # astropy >= 8
        v_sun = CartesianDifferential(v_sun.xyz)

    observed = c.transform_to(galactocentric_frame)
    rep = observed.cartesian.without_differentials()
    rep = rep.with_differentials(observed.cartesian.differentials['s'] + v_sun)
    return SkyCoord(galactocentric_frame.realize_frame(rep).transform_to(c.frame))


import sys
import paths  # noqa: F401  puts scripts/ on sys.path so this repo's streamframe.py is used
from streamframe import StreamFrame
import astropy.units as u
from tqdm import tqdm
agama.setUnits(length=1, velocity=1, mass=1)   # kpc, km/s, Msun

_tu = agama.getUnits()['time']
# timeUnitGyr = agama.getUnits()['time'] / 1e3  # time unit is 1 kpc / (1 km/s)
timeUnitMyr = getattr(_tu, 'value', _tu)     # 977.79
timeUnitGyr = timeUnitMyr / 1e3            # 0.9778
KMS_TO_KPCMYR = (u.km/u.s).to(u.kpc/u.Myr)   # agama velocities -> jake units
# print('time unit: %.3f Gyr' % timeUnitGyr)
# %%
### additional utility functions will go here. 

# %%
#### getting info about real streams
class extended_grid_info():
    """
    from a simulation grid in a different project; for calling 
    info about different streams assuming no LMC milky way potentials. 
    """
    def __init__(self, scratch=True):
        ### this is the present day progenitor locations, should match FINAL_ics_nolmc.csv. 
        # will be able to use these for indexing and outlier rejection. 
        self.aau_age = 2620 # Myr
        self.aau_apo = 39.75165840315315 # kpc
        self.aau_mprog = 17715.90 
        self.aau_init_displacement = [-9.907759532235083,-1.0857662629027969,-18.552299159266028,78.34958384955387,182.3175620146923,113.42022498966486]#[-10146.00826766252976085525,8919.85851260721756261773,-24103.22041046262529562227,103.04154947248773055435,79.06293437892136921619,127.13180750768465543388]

        self.c19_age = 3670 # Myr
        self.c19_apo = 22.37095916931149 # kpc
        self.c19_mprog = 2286.58
        self.c19_init_displacement = [-12.182298569979494,15.707205926304553,-9.692524337878561,56.692529246818786,-55.86054764465028,-122.08537966238212]#[12417.38173580253715044819,-18474.56735762703829095699,-1361.38862424165904485562,-5.31601757529761442811,-9.00030612586674827469,147.27681047797699420698]
        
        self.gd1_age = 2700 # Myr
        self.gd1_apo = 25.05610863789965 # kpc
        self.gd1_mprog = 23379.01
        self.gd1_init_displacement = [-12.484144078594696,0.00720607992464517,6.261514875525582,-71.05813416538996,-214.57029003578054,-114.74724521065625]#[101.24414852604452619289,16450.78926659073476912454,-2019.90591424317676683131,-162.30565892719911857967,90.69437329262960645337,134.06063942732188820628]


        self.jet_age = 3950 # Myr
        self.jet_apo = 34.960302575800554 # kpc  
        self.jet_mprog = 41625.82
        self.jet_init_displacement = [-17.806122300878584,-28.168067684031378,8.773655302758453,7.876750223201231,-65.84786729918034,-106.16378612431582]#[4260.60002970255027321400,-4299.63916052530657907482,-14178.84863869313085160684,74.33606358822831339239,252.25314609744174276784,-52.12315522407579493347]

        self.m3_apo = 23.224619245545487
        self.m3_age = 2360
        self.m3_mprog = 2085.23
        self.m3_init_displacement = [-4.4771491375984755,3.3809473715491123,6.234858242661406,115.88439360603759,90.61277762964873,-248.4402626937964]#[-879.12379288676629585098,7660.03420407489829813130,13384.68144932837276428472,109.58335588916739311571,-47.69100718430096463862,-143.56484318030865665605]


        self.pa5_apo = 16.525918308666228
        self.pa5_age = 3570
        self.pa5_mprog = 15646.15
        self.pa5_init_displacement = [6.560307554591288,0.5463314085703045,14.874685053085969,-39.195075218824215,-124.70324581743922,-9.507357539942179]#[3984.70438680068355097319,4551.73882141487047192641,-13226.39622530415363144130,113.09106691651193443704,-70.74982682771090480855,88.07032224724790125947]


        self.circ_init_displacement = np.array([20.0,0.0,0.0,0.0,197.61111175113643,0.0])  # <-- comes from computing vcirc in mwp2014 at 20 kpc
        self.circ_age = 10000
        self.circ_mprog = 1e4
        self.circ_apo = 20.0

    def retrieve_sim_info(self, orbit):
        if orbit=='circ':
            age = self.circ_age
            init_displacement = self.circ_init_displacement
            apo = self.circ_apo
            mprog = self.circ_mprog

        if orbit=='aau':
            age = self.aau_age
            init_displacement = self.aau_init_displacement
            apo = self.aau_apo
            mprog = self.aau_mprog
            
        if orbit=='c19':
            age = self.c19_age
            init_displacement = self.c19_init_displacement
            apo = self.c19_apo
            mprog = self.c19_mprog

        if orbit=='gd1':
            age = self.gd1_age
            init_displacement = self.gd1_init_displacement
            apo = self.gd1_apo
            mprog = self.gd1_mprog

        if orbit=='jet':
            age = self.jet_age
            init_displacement = self.jet_init_displacement
            apo = self.jet_apo
            mprog = self.jet_mprog

        if orbit=='m3':
            age = self.m3_age
            init_displacement = self.m3_init_displacement
            apo = self.m3_apo
            mprog = self.m3_mprog

        if orbit=='pa5':
            age = self.pa5_age
            init_displacement = self.pa5_init_displacement
            apo = self.pa5_apo
            mprog = self.pa5_mprog
        

        return apo, age, init_displacement, mprog
# %%
def galcen_to_ICRS(pos, vel):
    """
    ideally pos and vel have astropy units, and shape nx3
    """
    # create a cartesian representation/differential
    rep = CartesianRepresentation(pos.T)
    rep_vel = CartesianDifferential(vel.T)
    rep = rep.with_differentials(rep_vel)

    # create coords in galactocentric frame
    galcen_frame = Galactocentric()
    coords_galcen = SkyCoord(rep, frame=galcen_frame)

    # transform to ICRS
    coords_ICRS = coords_galcen.transform_to(ICRS())
    return coords_ICRS

def icrs_to_gd1Koposov10(ra, dec):
    """
    Convert (ra, dec) in radians to (phi1, phi2) in radians
    using the Koposov et al. 2010 transformation matrix.
    """

    # Rotation matrix from equatorial (RA, Dec) to stream (phi1, phi2)
    M = np.array([
        [-0.4776303088, -0.1738432154,  0.8611897727],
        [ 0.5108445890, -0.8524449229,  0.1112450420],
        [ 0.7147776536,  0.4930681392,  0.4959603976]
    ])

    # Input unit vector in equatorial coordinates
    x_eq = np.cos(ra) * np.cos(dec)
    y_eq = np.sin(ra) * np.cos(dec)
    z_eq = np.sin(dec)

    vec_eq = np.array([x_eq, y_eq, z_eq])

    # Apply rotation matrix
    x_stream, y_stream, z_stream = np.dot(M, vec_eq)

    # Convert rotated Cartesian vector to spherical coordinates
    phi1 = np.arctan2(y_stream, x_stream)
    phi2 = np.arcsin(z_stream)  # safer than arctan2(z, sqrt(x^2 + y^2))

    return phi1, phi2

##### transform to observed frames: 
def streamframe_coords_observed(orbit, xv_stream, prog_tab):
    """
    orbit options are:
    gd1, pa5, jet, aau, m3, c19, circ
    
    (input as strings), will put into a great-circle coordinate frame 
    based on ICRS ra/dec. this will allow accounting for projection 
    effects. also will correct radial velocity + proper motion for solar
    reflex motion, using a distance that for now is assumed to 
    be known exactly. For real streams this would be based on 
    a dynamically-fit distance track and hopefully would also be 
    pretty well-constrained....

    data_dict should be one of the three choicdes of subdictionary (CoM, luminous, companions)
    so that it has position and velocity info. those should have astropy units. 

    returns a coordinate object, since all frames can be constructed in gala with great circles. 
    """
    pos, vel = xv_stream[:,:3]*u.kpc, xv_stream[:,3:]*u.km/u.s
    coords_ICRS = galcen_to_ICRS(pos, vel) 
    coords_ICRS = reflex_correct(coords_ICRS) #<-- correct for solar reflex motion
    row = prog_tab[prog_tab['name']==orbit]
    

    if orbit=='gd1': #<-- pre-defined frame in gala. 
        # sc=coords_ICRS.transform_to(gc.GD1Koposov10)
        selected_streamframe = gc.GD1Koposov10
    if orbit=='pa5': #<-- pre-defined frame in gala. 
        # sc = coords_ICRS.transform_to(gc.Pal5PriceWhelan18)
        selected_streamframe = gc.Pal5PriceWhelan18

    if orbit=='c19': # defined by pole/origin in ibata+24, mohammed+26
        alpha_0 = 354.356*u.degree #<-- sets the phi1 zero point

        pole = SkyCoord( #<-- defines the pole
            ra= 81.45*u.degree,
            dec = -6.346*u.degree
        )
        prog_pos = np.array([row['x'][0], row['y'][0], row['z'][0]]) * u.kpc
        prog_vel = np.array([row['vx'][0], row['vy'][0], row['vz'][0]]) * u.km/u.s
        prog_coord = galcen_to_ICRS(prog_pos, prog_vel) #<-- debug and print htis out to see if its between the endpoints...
        origin = SkyCoord(
            ra=prog_coord.ra, dec=prog_coord.dec #<-- choose an origin... in ra
        )

        C19Ibata24 = gc.GreatCircleICRSFrame.from_pole_ra0(
            pole=pole, ra0=alpha_0, origin_disambiguate=origin #[coordinate that the origin should be closest to?]
        )
        # sc = coords_ICRS.transform_to(C19Ibata24)
        selected_streamframe = C19Ibata24

    if orbit=='jet': #<-- this also has a pole/origin vibe: from Do+26
        pole = SkyCoord(
            ra=64.983*u.degree, 
            dec=34.747*u.degree
        )
        origin = SkyCoord(
            ra = 138.62*u.degree,
            dec = 22.10*u.degree
        )
        JetDo26 = gc.GreatCircleICRSFrame.from_pole_ra0(
            pole=pole, ra0=origin.ra, origin_disambiguate=origin
        )
        # sc = coords_ICRS.transform_to(JetDo26)
        selected_streamframe = JetDo26

    if orbit=='aau': #<-- endpoints defined for ATLAS in table 1 of Shipp+2018
        endpoints = SkyCoord(
            ra=[9.3, 30.7]*u.degree,
            dec=[-20.9, -33.2]*u.degree
        )
        # pole = SkyCoord( #<-- don't actually need in my definition. 
        #     ra=74.3*u.degree, dec=47.9*u.degree
        # )
        prog_pos = np.array([row['x'][0], row['y'][0], row['z'][0]]) * u.kpc
        prog_vel = np.array([row['vx'][0], row['vy'][0], row['vz'][0]]) * u.km/u.s
        prog_coord = galcen_to_ICRS(prog_pos, prog_vel) #<-- debug and print htis out to see if its between the endpoints...
        origin = SkyCoord(
            ra=prog_coord.ra, dec=prog_coord.dec #<-- choose an origin... in ra
        )
        AAUShipp18 = gc.GreatCircleICRSFrame.from_endpoints( #<-- see table 1; for the ATLAS stream. 
            endpoints[0], endpoints[1], ra0=origin.ra#origin=origin, priority='pole'
        )
        # sc = coords_ICRS.transform_to(AAUShipp18)
        selected_streamframe = AAUShipp18


    if orbit=='m3': #<-- endpoints defined in Yang+23, sec 4.4
        # TODO: ics file gives something that is off from this frame; might want to redefine, and settle for "stream on an m3-like orbit." since this is mostly just a high e, low pericenter test. 
        endpoints = SkyCoord(
            ra=[186.45,197.20]*u.degree,
            dec=[19.06,27.76]*u.degree
        )
        # choose the origin based on the ICRS coordinate of the progenitor 
        # at present day. might be smart to read these in from data/FINAL_ics_nolmc.csv rather than hard coding here.
        prog_pos = np.array([row['x'][0], row['y'][0], row['z'][0]]) * u.kpc
        prog_vel = np.array([row['vx'][0], row['vy'][0], row['vz'][0]]) * u.km/u.s
        prog_coord = galcen_to_ICRS(prog_pos, prog_vel) #<-- debug and print htis out to see if its between the endpoints...
        origin = SkyCoord(
            ra=prog_coord.ra, dec=prog_coord.dec #<-- choose an origin... in ra
        )
        M3Yang23 = gc.GreatCircleICRSFrame.from_endpoints(
            endpoints[0], endpoints[1], origin=origin, priority='origin' #<-- warns...
        )
        # sc = coords_ICRS.transform_to(M3Yang23) 
        selected_streamframe = M3Yang23
    
    # else: #<-- why do i get this behavior when orbit='gd1' ??
    #     raise ValueError('acceptable orbits are gd1, pa5, jet, aau, m3, c19. for circular orbits use nominal streamframe.')
    
    sc = coords_ICRS.transform_to(selected_streamframe)

    # coords_stream = {} #<-- a dictionary, like Jake', but will have v_gsr instead of vr, and distance instead of r (different keys. )
    # coords_stream['phi1'] = sc.phi1.to(u.degree).value
    # coords_stream['phi2'] = sc.phi2.to(u.degree).value
    # coords_stream['pm_phi1'] = sc.pm_phi1_cosphi2.to(u.mas/u.yr).value
    # coords_stream['pm_phi2'] = sc.pm_phi2.to(u.mas/u.yr).value
    # coords_stream['distance'] = sc.distance.to(u.kpc).value
    # coords_stream['v_gsr'] = sc.radial_velocity.to(u.km/u.s).value

    return sc, selected_streamframe #<-- returned as a coordinate object. 


def prog_orbit_track(w0, Dt):
    """
    assumes we're in mwp2014. w0 should be a gala phase space position. 
    """
    mwp = gp.BovyMWPotential2014(units=galactic)
    H=gp.Hamiltonian(mwp)
    orbit_forward = H.integrate_orbit(w0, dt=1, n_steps=Dt)
    orbit_backward = H.integrate_orbit(w0, dt=-1, n_steps=Dt)
    # some slick lambda usage from claude:
    P = lambda o: np.array([o.pos.x.to(u.kpc).value,
                            o.pos.y.to(u.kpc).value,
                            o.pos.z.to(u.kpc).value]).T
    V = lambda o: np.array([o.vel.d_x.to(u.km/u.s).value,
                            o.vel.d_y.to(u.km/u.s).value,
                            o.vel.d_z.to(u.km/u.s).value]).T
    orbit_pos = np.vstack([P(orbit_backward)[::-1][:-1], P(orbit_forward)])
    orbit_vel = np.vstack([V(orbit_backward)[::-1][:-1], V(orbit_forward)])
    # ^ note the first half of the orbit integration has the last element 
    # chopped off to avoid duplicating the progenitor position. 
    return orbit_pos*u.kpc, orbit_vel*u.km/u.s

def observed_orbit_track(orbit_pos, orbit_vel, obs_streamframe):
    """
    give orbit_pos and orbit_vel with astropy units. 
    """
    # orbit_pos, orbit_vel = orbit_w[:,:3] * u.kpc, orbit_w[:,3:] * u.kpc/u.Myr
    orbit_icrs = galcen_to_ICRS(orbit_pos, orbit_vel)
    orbit_icrs = reflex_correct(orbit_icrs) #<-- reflex correct the orbit too, or it doesn't match the data.
    sc = orbit_icrs.transform_to(obs_streamframe)
    return sc

def chop_orbit_track(phi1, jump_threshold=45):
    """
    safe-guard against non-strictly-increasing phi1
    in the progenitor orbit interpolant. 
    """
    phi1_track = np.asarray(phi1, dtype=float)
    n = len(phi1_track)
    i_prog = n//2 #<-- the center, where the prog is
    dphi1 = np.diff(phi1_track)
    bad = np.abs(dphi1)>jump_threshold

    ## guard against when things turn around in phi1 without discontinuity. 
    s = np.sign(dphi1[i_prog]) if i_prog < n-1 else np.sign(dphi1[i_prog-1])
    bad = bad | (np.sign(dphi1) != s)


    lo=i_prog
    while lo>0 and not bad[lo-1]:
        lo-=1
    hi=i_prog
    while hi<n-1 and not bad[hi]:
        hi+=1
    idx = np.arange(lo, hi+1)
    if phi1_track[idx[0]] > phi1_track[idx[-1]]: #<-- make sure phi1 is increasing. 
        idx = idx[::-1]   # so it can go straight into np.interp
    return idx

def straightened_obscoords_orbit_interp(orbit, xv_stream, prog_tab, Dt=500):
    """
    should _this_ return a dictionary ??? 
    """
    coords_obs, obs_streamframe = streamframe_coords_observed(orbit, xv_stream, prog_tab)
    row = prog_tab[prog_tab['name']==orbit]
    w0 = gd.PhaseSpacePosition(
        np.array([row['x'][0], row['y'][0], row['z'][0]])*u.kpc,
        np.array([row['vx'][0], row['vy'][0], row['vz'][0]])*u.km/u.s
        )
    orbit_pos, orbit_vel = prog_orbit_track(w0, Dt)
    orbit_sf = observed_orbit_track(orbit_pos, orbit_vel, obs_streamframe)
    idx = chop_orbit_track(orbit_sf.phi1.to(u.degree).value)

    scd = {} #<--"straight coord dict"
    scd['phi1'] = coords_obs.phi1.to(u.degree).value
    data_x = scd['phi1']
    data_y = [coords_obs.phi2.to(u.degree).value,
              coords_obs.pm_phi1_cosphi2.to(u.mas/u.yr).value,
              coords_obs.pm_phi2.to(u.mas/u.yr).value,
              coords_obs.radial_velocity.to(u.km/u.s).value,
              coords_obs.distance.to(u.kpc).value
              ]

    keys = ['phi2','pm_phi1','pm_phi2','v_gsr','distance']
    orbit_x = orbit_sf.phi1[idx].to(u.degree).value
    orbit_y = [orbit_sf.phi2[idx].to(u.degree).value,
               orbit_sf.pm_phi1_cosphi2[idx].to(u.mas/u.yr).value,
               orbit_sf.pm_phi2[idx].to(u.mas/u.yr).value,
               orbit_sf.radial_velocity[idx].to(u.km/u.s).value,
               orbit_sf.distance[idx].to(u.kpc).value
               ]

    # populate the coordinate dictionary:       
    for ii, key in enumerate(keys):
        scd[key] = data_y[ii] - np.interp(data_x, orbit_x, orbit_y[ii])
    return scd

##############################################
#       statistics

def find_minima(arr):
    min_indeces = []
    min_values = []
    for i,a in enumerate(arr):
        # print(i)
        if i==0 or i==int(len(arr)-1):
            # print("first/last value, skipping")
            continue
        if (arr[i-1]>a) & (arr[i+1]>a):
            # print("appending a local minimum")
            min_indeces.append(i)
            min_values.append(a)

    return np.array(min_indeces), np.array(min_values)

def find_maxima(arr):
    max_indeces = []
    max_values = []
    for i, a in enumerate(arr):
        if i==0 or i==int(len(arr)-1):
            continue
        if (arr[i-1]<a) & (arr[i+1]<a):
            max_indeces.append(i)
            max_values.append(a)

    return np.array(max_indeces), np.array(max_values)


def straighten_stream_polynomial(coords, keys=('phi2',), x_key='phi1',
                                 degree=5, show_plot=False,
                                 return_poly_fn=False):
    """
    straighten stream coordinates by fitting a polynomial in x_key (phi1 by default)
    to each of the y-values in keys, and subtracting it.

    coords: dict-like (dict, astropy Table, ...) of stream coordinates, indexable by
            x_key and by every entry of keys. Quantities are stripped to plain arrays.
    keys:   list of keys to straighten, e.g. ['phi2', 'v_gsr', 'pm_phi1', 'pm_phi2'].
            a single string is also accepted.

    returns a dict {key: straightened array} with exactly the keys in `keys`.
    with return_poly_fn=True, returns (straightened, poly, fits), where fits is a dict
    {key: polynomial coefficients}, so poly(x, *fits[key]) is the fitted track.
    non-finite (x, y) pairs are left out of the fit and come back as NaN.
    """
    if isinstance(keys, str):
        keys = [keys]
    x = np.asarray(coords[x_key], dtype=float)
    def make_polynomial(order):
        # Create the argument list: a0, a1, ..., an
        arg_list = ", ".join([f"a{i}" for i in range(order + 1)])

        # Build the function definition string
        func_str = f"def poly(x, {arg_list}):\n"
        func_str += "    return " + " + ".join([f"a{i}*x**{i}" for i in range(order + 1)]) + "\n"

        # Local namespace for exec
        local_ns = {}
        exec(func_str, {}, local_ns)
        return local_ns['poly']
    
    poly=make_polynomial(order=degree)

    # fit a polynomial in x to each key and subtract to straighten out
    straightened = {}
    fits = {}
    for key in keys:
        y = np.asarray(coords[key], dtype=float)
        good = np.isfinite(x) & np.isfinite(y)
        fit, cov = curve_fit(f=poly, xdata=x[good], ydata=y[good])
        straightened[key] = y - poly(x, *fit)
        fits[key] = fit

    if show_plot==True:
        fig, axs = plt.subplots(len(keys), 2, figsize=[16, 4*len(keys)], squeeze=False)
        x_plot = np.linspace(np.nanmin(x), np.nanmax(x))
        for (ax1, ax2), key in zip(axs, keys):
            ax1.scatter(x, coords[key], s=.1)
            ax1.plot(x_plot, poly(x_plot, *fits[key]), c='tomato')
            ax1.set_ylabel(key)
            ax2.scatter(x, straightened[key], s=.1)
            ax2.set_ylabel(key + ' (straightened)')
        for ax in axs[-1]:
            ax.set_xlabel(x_key)

    if return_poly_fn==False:
        return straightened
    if return_poly_fn==True:
        return straightened, poly, fits