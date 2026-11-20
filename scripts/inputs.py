# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import cmasher as cmr
import multiprocessing as mp

from scipy.stats import binned_statistic
from scipy.signal import argrelextrema, find_peaks
from scipy.special import logsumexp, ndtr, ndtri_exp, log_ndtr
from scipy.optimize import brentq


# plt.style.use('vedant') #<-- won't work on cannon. need vedant.mplstyle in scripts
import sys
from paths import repo_path, MPLSTYLE, DISSOLVED_STREAMS_DIR  # machine-specific paths, see paths.py
plt.style.use(MPLSTYLE)
import functions as f
import tsetse as tse
from streamframe import StreamFrame
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
from matplotlib.gridspec import GridSpecFromSubplotSpec

import astropy.units as u
import astropy.constants as const
import astropy.coordinates as coord
from astropy.coordinates import Galactocentric, ICRS, CartesianRepresentation, CartesianDifferential
from astropy.coordinates import SkyCoord


import agama
agama.setUnits(length=1, velocity=1, mass=1)   # kpc, km/s, Msun

_tu = agama.getUnits()['time']
# timeUnitGyr = agama.getUnits()['time'] / 1e3  # time unit is 1 kpc / (1 km/s)
timeUnitMyr = getattr(_tu, 'value', _tu)     # 977.79
timeUnitGyr = timeUnitMyr / 1e3            # 0.9778
KMS_TO_KPCMYR = (u.km/u.s).to(u.kpc/u.Myr)   # agama velocities -> jake units

import gala.coordinates as gc
import gala.dynamics as gd
import gala.potential as gp
from gala.dynamics import mockstream as ms
from gala.units import galactic
from gala.coordinates import reflex_correct
from astropy.table import Table
import pickle

from pygaia.errors.astrometric import parallax_uncertainty, proper_motion_uncertainty, total_proper_motion_uncertainty, total_position_uncertainty
import viamock  # viamock + extra site-packages are put on sys.path by paths.py
import artpop

# import filter_v2 as filter
# %%
def get_noise(ssc_table, noise_seed=42):
    """
    adds via (viamock) and gaia (pygaia) like noise to data
    columns with the error, the sampled noise, and the data+noise are added
    to the input table.

    noise_seed seeds the gaussian noise draws, so the same table and seed
    always give the same noised columns.
    """
    rv_err, feh_err, afe_err = viamock.get_viaspec_errors(ssc_table['phot_g_mean_mag'],ssc_table['feh'], ssc_table['eep']) #<--  1h exposure 1" seeing 1.2 airmass, dark
    ssc_table['v_gsr_err'] = rv_err

    pm_err = total_proper_motion_uncertainty(ssc_table['phot_g_mean_mag'], 'dr4') / np.sqrt(2)
    pos_err = total_position_uncertainty(ssc_table['phot_g_mean_mag'], 'dr4') / np.sqrt(2)

    ssc_table['pm_phi1_err'] = pm_err * (u.microarcsecond/u.yr).to(u.mas/u.yr)
    ssc_table['pm_phi2_err'] = pm_err * (u.microarcsecond/u.yr).to(u.mas/u.yr)
    ssc_table['phi1_err'] = pos_err * u.microarcsecond.to(u.degree)
    ssc_table['phi2_err'] = pos_err * u.microarcsecond.to(u.degree)

    rng = np.random.default_rng(seed=noise_seed)
    ssc_table['v_gsr_noise'] = rng.normal(loc=0, scale=ssc_table['v_gsr_err'], size=len(ssc_table))
    ssc_table['pm_phi1_noise'] = (rng.normal(0, pm_err) * u.microarcsecond / u.yr).to(u.mas/u.yr).value
    ssc_table['pm_phi2_noise'] = (rng.normal(0, pm_err) * u.microarcsecond / u.yr).to(u.mas/u.yr).value
    ssc_table['phi1_noise'] = (rng.normal(0, pos_err) * u.microarcsecond).to(u.degree).value
    ssc_table['phi2_noise'] = (rng.normal(0, pos_err) * u.microarcsecond).to(u.degree).value

    # now make a noised versions of the phase space dimension columns. 
    ssc_table['phi1_noised'] = ssc_table['phi1'] + ssc_table['phi1_noise']
    ssc_table['phi2_noised'] = ssc_table['phi2'] + ssc_table['phi2_noise']
    ssc_table['pm_phi1_noised'] = ssc_table['pm_phi1'] + ssc_table['pm_phi1_noise']
    ssc_table['pm_phi2_noised'] = ssc_table['pm_phi2'] + ssc_table['pm_phi2_noise']
    ssc_table['v_gsr_noised'] = ssc_table['v_gsr'] + ssc_table['v_gsr_noise']

    return ssc_table




def inputs_agama(orbit, pert_table, index=1636, paint_seed=42, noise_seed=42, Glim=24., m_pop=None):
    """
    given a pickled dictionary
    made for perturbation [index] in pert_table,
    paint a simple stellar population onto the data
    and calculate gaia and via-like errors.
    return the resulting data + uncertainty, with
    columns for [quantity]_noised to run through the filter
    and [quantity]_err to use as in the likelihood variance

    paint_seed seeds viamock.paint_ssp (which particles become stars, and their
    magnitudes, hence their errors and which pass the precision cuts). noise_seed
    seeds the noise draws in get_noise. None for either is not reproducible.

    Glim: the magnitude cut after painting, as in paint_and_noise (None = no cut).

    returns both the ssc_pert table that goes into the filter
    and phi1_loc the location of the true perturbation.
    """

    # i = 1636 # <-- a "great" example. 

    i=index
    row = pert_table[pert_table['ID']==i]

    indir = DISSOLVED_STREAMS_DIR + "%s/"%orbit
    fname = "%s_%i.pickle"%(orbit, i)

    with open(indir+fname, 'rb') as handle:
        pert_dict_in = pickle.load(handle)

    ssc = pert_dict_in["ssc"]
    ssc_pert = pert_dict_in["ssc_pert"] # <-- observed location of perturbation
    phi1_loc = ssc_pert.phi1.to(u.degree).value

    ssc_table = paint_and_noise(ssc, paint_seed=paint_seed, noise_seed=noise_seed, Glim=Glim, m_pop=m_pop)

    return ssc_table, phi1_loc


GLIM_NONE = 99. # <-- the Glim used for Glim=None. viamock.paint_ssp wants a finite Glim.

def paint_and_noise(ssc, paint_seed=42, noise_seed=42, Glim=24.,
                    m_pop=None):
    """
    paint a simple stellar population onto a stream with viamock.paint_ssp
    (feh = -2, 10 Gyr, total_mass = 0.5 Msun per particle), then add errors
    and noise with get_noise. ssc is anything indexable by column name (a dict
    of arrays or a Table) and needs a 'distance' column in kpc.

    Glim: the apparent Gaia G cut paint_ssp applies after painting (default 24).
        None = no cut (GLIM_NONE, i.e. G < 99): every particle that gets a star is
        kept, for noiseless fits that should see the whole stream. paint_ssp gives
        one star per particle, so at 0.5 Msun per particle (~1.2 stars) that's
        every particle. errors and noise are still computed for the faint stars,
        but they mean little past the survey limits (see get_noise).

    m_pop: mass of simple stellar population. if None (default) just use 0.5xNstars
            (i.e., stars have average mass 0.5 Msun). Increasing it above this number 
            will give you more bright stars and could be tuned to the estimated 
            stream progenitor masses, which might be >5000 Msun. 

    returns a new Table of the painted stars only: paint_ssp shuffles the rows
    and keeps the stars brighter than Glim, so it can be shorter than ssc.

    the mass per particle is fixed, so streams with different particle numbers
    get the same luminosity function per particle. a dense null painted this
    way has the same distribution of magnitudes (so of errors and of who passes
    the precision cuts) at a given distance as a 10k-particle perturbed stream.

    seeds as in inputs_agama.
    """
    tab = Table()
    for key in ssc.keys():
        tab[key] = ssc[key]


    total_mass = 0.5*len(tab) if m_pop is None else m_pop # 
    ssc_table_noiseless = viamock.paint_ssp(tab, feh=-2.0, log_age=np.log10(10e9),
                                total_mass = total_mass, shuffle_seed=paint_seed,
                                Glim=GLIM_NONE if Glim is None else Glim)

    return get_noise(ssc_table_noiseless, noise_seed=noise_seed)

def fit_streamframe(ICRS_coords, try_every=20):
    """
    ported from an old repository
    get phi1, phi2 from transforming to ICRS coordinates, fitting a great circle coordinate frame
    then fitting a quadratic, cubic, or spline and subtracting it. 
    for finding stream widths; progenitor will not necessarily be centered.

    returns gc_frame_best, xformed_coords
    """
    obfun_best = np.inf
    gc_frame_best = None
    point1_best = None
    point2_best = None

    point1 = SkyCoord(
        ra = np.nanmedian(ICRS_coords.ra),
        dec = np.nanmedian(ICRS_coords.dec),
        frame='icrs'
    )
    
    print("iterating to find best frame to minimize phi2")
    for i in tqdm(range(0, len(ICRS_coords), try_every)):
        point2 = ICRS_coords[i]

        try:
            gc_frame = gc.GreatCircleICRSFrame.from_endpoints(point1, point2)
            stars_sc_gc = ICRS_coords.transform_to(gc_frame)
            obfun = np.sum(stars_sc_gc.phi2.value**2)

            if obfun < obfun_best:
                obfun_best = obfun # update objective function
                gc_frame_best = gc_frame
                point2_best = point2

        except Exception as e:
            continue # skipped failed point2s
    
    xformed_coords = ICRS_coords.transform_to(gc_frame_best)
    
    streamframe = gc_frame_best
    streamframe_data = xformed_coords

    return streamframe, streamframe_data

def ICRS_from_galcen(xv):
    """
    ICRS from galactocentric coords xv
    """
    pos = xv[:,:3] * u.kpc
    vel = xv[:,3:] * u.km/u.s
 
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

def ICRS_from_table(tab):
    """
    return astropy coordinate object from viamock table 
    """
    coords_ICRS = SkyCoord(
        ra=tab['ra']*u.degree,
        dec=tab['dec']*u.degree,
        pm_ra_cosdec=tab['pmra']*u.mas/u.yr,
        pm_dec=tab['pmdec']*u.mas/u.yr,
        radial_velocity = tab['rv']*u.km/u.s,
        distance=tab['distance']*u.kpc
    )
    return coords_ICRS


def jake_streamtab_from_galcen(stars_w, prog_w):
        stream_x = stars_w[:,:3]## first three columns of stars_w
        stream_v = stars_w[:,3:] *(u.km/u.s).to(u.kpc/u.Myr)## last three columns of stars_w * (u.km/u.s).to(u.kpc/u.Myr)

        stream_xv =  np.hstack([stream_x, stream_v])# column stack stream_x and stream_v

        prog_x = prog_w[:3]
        prog_r = np.sqrt(np.sum(prog_x**(2)))

        prog_v = prog_w[3:] * (u.km/u.s).to(u.kpc/u.Myr)
        prog_xv = np.hstack([prog_x, prog_v])

        sf = StreamFrame(stream_xv, prog_xv)
        sc = sf.GetStreamFrame()

        tab = Table()
        tab['phi1'] = sc['phi1']
        tab['phi2'] = sc['phi2']
        tab['pm_phi1'] = sc['pm_phi1']
        tab['pm_phi2'] = sc['pm_phi2']
        tab['v_gsr'] = sc['vr'] #<-- technically mislabeled. this is _not_ rv in galactic standard of rest
        tab['r'] = sc['r']

        return tab, prog_r


def inputs_galactocentric(stars_w, prog_w=None, potential=None, paint_seed=42, noise_seed=42,
                          Glim=24., m_pop=None, 
                          stream_frame='great_circle', #<--- 'great_circle' or 'galactocentric'
                          distance=None, # if the streamframe is 'galactocentric' re-scale the magnitudes as if the progenitor is at this distance (if not none). TODO helper function for earth-progenitor distance.  
                          **track_kwargs):
    """
    TODO: OPTION TO RETURN THE TABLE IN GALACTOCENTRIC "STREAMFRAME" COORDINATES.
          FOR DISTANCE I'D LIKE TO USE THE MEAN DISTANCE OF THE STREAM FROM THE EARTH, 
          AND THEN JUST TAKE LIKE RENORMALIZE THE MAGNITUDES TO BE AT THAT DISTANCE PLUS
          OR MINUS WHATEVER OFFSET FROM THE MEAN IN THE GALACTOCENTRIC STREAMFRAME. 


    mock observations from galactocentric phase space: viamock.xv_to_table ->
    ICRS_from_table -> a great-circle frame fit to the stars (fit_streamframe) -> paint +
    noise -> (if prog_w is given) subtract the progenitor orbit.

    stars_w: (N, 6) present-day galactocentric stars (kpc, km/s), e.g. tse.clean_stream,
        or a gen_perturbed_stream pickle's 'stars_w'.
    prog_w: (6,) present-day galactocentric progenitor (kpc, km/s), e.g. the pickle's
        'prog_w'. if given, its orbit is integrated in `potential` with agama, put through
        the same steps as the stars (prog_track_in_frame, i.e. tse.progenitor_track_coords
        with frame=streamframe) and subtracted from GRID_KEYS (subtract_prog_track; the
        _noised columns too). phi1 and distance are left alone. None = no straightening.
    potential: what to integrate the progenitor in, anything tse.load_potential takes
        (None = data/MWPotential2014.ini, a .ini filename/path, or an agama.Potential).
        should be the potential the stream was made in.
    paint_seed, noise_seed: as in inputs_agama.
    Glim: the magnitude cut after painting, as in paint_and_noise (None = no cut, so
        all the particles are kept).
    track_kwargs: to tse.progenitor_track_coords (phi1_max_deg, n_half, pad).

    returns (ssc_table, streamframe, track): the painted + noised table, the fitted
    great-circle frame, and the progenitor track that was subtracted (a dict like
    prog_track_observed's; None without prog_w). NOT reflex corrected, stars or orbit
    (so 'v_gsr' here is the heliocentric rv, and the pms include the solar reflex).
    """

    ##### OPTION 1: stream frame based on ICRS coordinates
    if stream_frame=='great_circle':
        tab = viamock.xv_to_table(stars_w)                     # ra, dec, distance, dm, rv, ...

        icrs_data = ICRS_from_table(tab) # create an astropy coordinates object with ra, dec, proper motions, v_gsr,...
        streamframe, streamframe_data = fit_streamframe(icrs_data) #<-- streamframe (no underscore) becomes the astropy coordinates object for transformation. 

        tab['phi1']     = streamframe_data.phi1.to(u.degree).value #degree / remove units
        tab['phi2']     = streamframe_data.phi2.to(u.degree).value #deg
        tab['pm_phi1']  = streamframe_data.pm_phi1_cosphi2.to(u.mas/u.yr).value #mas/yr
        tab['pm_phi2']  = streamframe_data.pm_phi2.to(u.mas/u.yr).value
        tab['v_gsr']    = streamframe_data.radial_velocity.to(u.km/u.s).value # km/s
        tab['distance'] = streamframe_data.distance.to(u.kpc).value #kpc



    if stream_frame=='galactocentric':
        tab, prog_r = jake_streamtab_from_galcen(stars_w, prog_w)

        if distance is None:
            tab['distance'] = tab['r']
        else:
            tab['distance'] = tab['r'] - prog_r + distance #<--- put the stream as if the progenitor is at this distance... a little wack maybe. 
        

    # then paint and noise only cares about the distance column! 
    ssc_table = paint_and_noise(tab, paint_seed=paint_seed, noise_seed=noise_seed, Glim=Glim, m_pop=m_pop)

    # straighten after painting + noising: the noise is additive, so this is the same as
    # noising the straightened columns, and paint_ssp sees the true distances either way.
    track = None
    if prog_w is not None:
        ##### NEED AN OPTION FOR GALACTOCENTRIC VS OBSERVED FRAME!!!
        if stream_frame=='great_circle':
            sf = streamframe #<-- the astropy coordinate frame object
        if stream_frame=='galactocentric':
            sf = None #<-- so that the prog track will default to jake's frame
        track = prog_track_in_frame(prog_w, sf, potential=potential, **track_kwargs)
        if stream_frame=='galactocentric':
            track['v_gsr']=track['vr']
            streamframe = None #<-- so that the return is okay. 
        ssc_table = subtract_prog_track(ssc_table, track)

    return ssc_table, streamframe, track



#### TRACK STUFF
GRID_KEYS = ['phi2', 'pm_phi1', 'pm_phi2', 'v_gsr'] 

def prog_track_observed(orbit, Dt=500):
    """
    the progenitor orbit track that f.straightened_obscoords_orbit_interp subtracts
    (and so the one subtracted from every saved ssc in tse.gen_perturbed_stream):
    the prog_tab (FINAL_ics_nolmc.csv) progenitor, integrated +/- Dt Myr in gala's
    BovyMWPotential2014, reflex corrected, in the published stream frame, and chopped
    to the single monotonic phi1 branch through the progenitor.

    returns a dict of 1-D arrays: 'phi1' (increasing, deg) and GRID_KEYS.
    """
    prog_tab = tse.prog_tab
    row = prog_tab[prog_tab['name']==orbit]
    xv_prog = np.array([[row[c][0] for c in ('x', 'y', 'z', 'vx', 'vy', 'vz')]])

    # the frame only depends on orbit (+ prog_tab), so build it from the progenitor itself
    _, obs_streamframe = f.streamframe_coords_observed(orbit, xv_prog, prog_tab)
    w0 = gd.PhaseSpacePosition(xv_prog[0, :3]*u.kpc, xv_prog[0, 3:]*u.km/u.s)
    orbit_pos, orbit_vel = f.prog_orbit_track(w0, Dt)
    orbit_sf = f.observed_orbit_track(orbit_pos, orbit_vel, obs_streamframe)
    return _track_dict(orbit_sf)

def _track_dict(orbit_sf):
    """
    chop a progenitor orbit (a coordinate object in a stream frame, progenitor at the
    middle index) to the single monotonic phi1 branch through the progenitor, as in
    f.straightened_obscoords_orbit_interp, and return it as a dict of 1-D arrays:
    'phi1' (increasing, deg) and GRID_KEYS.
    """
    idx = f.chop_orbit_track(orbit_sf.phi1.to(u.degree).value)
    sf = orbit_sf[idx]
    return {'phi1': sf.phi1.to(u.degree).value,
            'phi2': sf.phi2.to(u.degree).value,
            'pm_phi1': sf.pm_phi1_cosphi2.to(u.mas/u.yr).value,
            'pm_phi2': sf.pm_phi2.to(u.mas/u.yr).value,
            'v_gsr': sf.radial_velocity.to(u.km/u.s).value}

def prog_track_in_frame(prog_w, streamframe, potential=None, **track_kwargs):
    """
    the progenitor orbit track for data made by inputs.inputs_galactocentric, i.e. in
    the great-circle frame fit_streamframe fit to the stars, and NOT reflex corrected.
    a wrapper for tse.progenitor_track_coords(frame=streamframe): the orbit is integrated
    with agama and converted with tse.galcen_to_frame, which uses the same galactocentric
    parameters and the same route through ICRS as the stars (viamock.xv_to_table ->
    inputs.ICRS_from_table -> streamframe), with no reflex correction either.

    prog_w: the progenitor's present-day [x, y, z, vx, vy, vz] (kpc, km/s), e.g. its
        prog_tab row, or a gen_perturbed_stream pickle's 'prog_w'.
    streamframe: the frame inputs.inputs_galactocentric returns (or any astropy stream
        frame with phi1/phi2).
    potential: anything tse.load_potential takes (None = data/MWPotential2014.ini, a .ini
        filename/path, or an agama.Potential).
    track_kwargs: to tse.progenitor_track_coords (phi1_max_deg, n_half, pad).

    returns a dict like prog_track_observed's (plus 'distance'), for subtract_prog_track.
    """
    _, track = tse.progenitor_track_coords(prog_w, pot_host=potential, frame=streamframe,
                                           **track_kwargs)
    return track


def _shift_by_track(ssc, track, sign, keys=GRID_KEYS):
    """
    key + sign*interp(phi1, track) for each key (and <key>_noised, if present), at the
    noiseless phi1. returns a copy.
    """
    out = ssc.copy()
    x = np.asarray(ssc['phi1'], dtype=float)
    n_out = np.count_nonzero((x < np.min(track['phi1'])) | (x > np.max(track['phi1'])))
    if n_out:
        print('WARNING: %d/%d stars fall outside the track phi1 range [%.1f, %.1f]; '
              'np.interp is clamping them' % (n_out, x.size, np.min(track['phi1']), np.max(track['phi1'])))
    for key in keys:
        y_track = np.interp(x, track['phi1'], track[key])
        out[key] = np.asarray(ssc[key], dtype=float) + sign*y_track
        if key+'_noised' in ssc.keys():
            out[key+'_noised'] = np.asarray(ssc[key+'_noised'], dtype=float) + sign*y_track
    return out

def subtract_prog_track(ssc, track, keys=GRID_KEYS):
    """
    straighten a stream: key - interp(phi1, track), the same subtraction as
    f.straightened_obscoords_orbit_interp, for coordinates given with the progenitor
    orbit still in them. track is from prog_track_in_frame (or prog_track_observed for
    the published frames). <key>_noised columns get the same track subtracted, which
    is the same as noising the straightened column, since the noise is additive.
    phi1 and distance are left alone. returns a copy; ssc isn't modified.
    """
    return _shift_by_track(ssc, track, -1., keys=keys)

def add_prog_track(ssc, orbit, track=None, keys=GRID_KEYS, Dt=500):
    """
    undo the progenitor-orbit subtraction in a saved (straightened) ssc, i.e. return
    the observed-frame coordinates: key + interp(phi1, track). exact, since phi1 itself
    was never straightened and np.interp is evaluated at the same phi1.


    !!! THIS ONLY WORKS FOR MY PERTURBED STREAMS! NOT IN GENERAL! 
    
    ssc: a dict or astropy Table with 'phi1' and keys (e.g. the inputs.inputs_agama table).
        if <key>_noised columns exist they get the same track added (the noise is
        additive, so this equals noising the un-straightened column). the track is
        evaluated at the noiseless 'phi1', which is what the subtraction used.
    track: the output of prog_track_observed(orbit, Dt); pass it in when looping over
        many streams on the same orbit to skip re-integrating.

    'distance' is left alone: tse.gen_perturbed_stream overwrites it with the absolute
    distance after straightening. v_phi1/v_phi2 are also left alone (they're the
    straightened pm times distance).

    returns a copy; ssc isn't modified.
    """
    if track is None:
        track = prog_track_observed(orbit, Dt=Dt)
    return _shift_by_track(ssc, track, +1., keys=keys)
# %%
###### SUBTRACT A POLYNOMIAL FROM THE TABLE DAWG
def subtract_polynomial(ssc, n=5, keys=GRID_KEYS, show_plot=False):
    """
    straighten a stream by fitting a degree-n polynomial in phi1 to each key and
    subtracting it, using f.straighten_stream_polynomial.

    ssc: a dict or astropy Table with 'phi1' and keys.
    n: polynomial degree (default 5).
    keys: the phase space dimensions to straighten (default GRID_KEYS). phi1 and
        distance are left alone.

    if <key>_noised columns exist, they get their own fit, to <key>_noised against
    'phi1_noised' (or 'phi1' if there's no 'phi1_noised'), so the noised version only
    uses "observed" quantities. non-finite values (e.g. the NaN v_gsr_err stars) are
    left out of the fits and stay NaN.

    returns a copy; ssc isn't modified.
    """
    out = ssc.copy()
    keys = list(keys)

    straight = f.straighten_stream_polynomial(ssc, keys=keys, x_key='phi1',
                                              degree=n, show_plot=show_plot)
    for key in keys:
        out[key] = straight[key]

    keys_noised = [key+'_noised' for key in keys if key+'_noised' in ssc.keys()]
    if len(keys_noised) > 0:
        x_key = 'phi1_noised' if 'phi1_noised' in ssc.keys() else 'phi1'
        straight = f.straighten_stream_polynomial(ssc, keys=keys_noised, x_key=x_key,
                                                  degree=n, show_plot=show_plot)
        for key in keys_noised:
            out[key] = straight[key]
    return out