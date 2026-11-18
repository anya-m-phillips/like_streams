# %% 
import agama
import numpy as np
# import matplotlib.pyplot as plt
# plt.style.use('vedant') #<-- won't work on cannon. need vedant.mplstyle in scripts
import sys
repo_path = '/n/home02/amphillips/like_streams'
sys.path.append(repo_path+'/scripts/')
from streamframe import StreamFrame
import astropy.units as u
from tqdm import tqdm

sys.path.append(repo_path+"/scripts")
import functions as f
### agama setup: 
agama.setUnits(length=1, velocity=1, mass=1)   # kpc, km/s, Msun

_tu = agama.getUnits()['time']
# timeUnitGyr = agama.getUnits()['time'] / 1e3  # time unit is 1 kpc / (1 km/s)
timeUnitMyr = getattr(_tu, 'value', _tu)     # 977.79
timeUnitGyr = timeUnitMyr / 1e3            # 0.9778
KMS_TO_KPCMYR = (u.km/u.s).to(u.kpc/u.Myr)   # agama velocities -> jake units
# print('time unit: %.3f Gyr' % timeUnitGyr)

from astropy.table import Table
from astropy.coordinates import SkyCoord, Galactocentric, galactocentric_frame_defaults
import pickle
import os
# %%
### stuff that we'll need:
prog_tab = Table.read(repo_path+'/data/FINAL_ics_nolmc.csv') #<-- TODO: get this but actual conditions with LMC from vedant.
o = f.extended_grid_info()

DEFAULT_POTENTIAL = repo_path + '/data/MWPotential2014.ini'

def load_potential(potential=None):
    """
    an agama.Potential from any of:
      None             -> DEFAULT_POTENTIAL (data/MWPotential2014.ini)
      agama.Potential  -> returned as is
      str              -> an agama .ini file. a path that doesn't exist as given is
                          looked for in the repo's data/ directory, so
                          'MWPotential2014.ini' works from anywhere.
    """
    if potential is None:
        potential = DEFAULT_POTENTIAL
    if isinstance(potential, agama.Potential):
        return potential
    if isinstance(potential, str):
        path = potential if os.path.exists(potential) else os.path.join(repo_path, 'data', potential)
        if not os.path.exists(path):
            raise FileNotFoundError('no potential file %r (also looked in %s/data/)' % (potential, repo_path))
        return agama.Potential(path)
    raise TypeError('potential must be None, a filename or an agama.Potential, not %s' % type(potential))


# utilities from bill chen:
def get_rot_mat(x, y, z, vx, vy, vz):
    Lx = y * vz - z * vy
    Ly = z * vx - x * vz
    Lz = x * vy - y * vx
    r = (x*x + y*y + z*z)**0.5
    L = (Lx*Lx + Ly*Ly + Lz*Lz)**0.5
    # rotation matrices transforming from the host to the satellite frame for each point on the trajectory
    R = np.zeros((len(x), 3, 3))
    R[:,0,0] = x/r
    R[:,0,1] = y/r
    R[:,0,2] = z/r
    R[:,2,0] = Lx/L
    R[:,2,1] = Ly/L
    R[:,2,2] = Lz/L
    R[:,1,0] = R[:,0,2] * R[:,2,1] - R[:,0,1] * R[:,2,2]
    R[:,1,1] = R[:,0,0] * R[:,2,2] - R[:,0,2] * R[:,2,0]
    R[:,1,2] = R[:,0,1] * R[:,2,0] - R[:,0,0] * R[:,2,1]
    return R, L, r

def get_d2Phi_dr2(pot_host, x, y, z, t=0):
    # compute  the second derivative of potential by spherical radius
    # t: time (scalar or one per point) for time-dependent hosts; 0 is agama's default
    r = (x*x + y*y + z*z)**0.5
    der = pot_host.forceDeriv(np.column_stack([x,y,z]), t=t)[1]
    d2Phi_dr2 = -(x**2  * der[:,0] + y**2  * der[:,1] + z**2  * der[:,2] +
                  2*x*y * der[:,3] + 2*y*z * der[:,4] + 2*z*x * der[:,5]) / r**2
    return d2Phi_dr2

def create_ic_chen25(rng, pot_host, orb_sat, mass_sat, t=0):
    # t: times of the orb_sat points, only matters if pot_host is time-dependent
    # mass_sat: a scalar, or one mass per orb_sat point (the progenitor's mass when that pair is
    #           released). it sets both r_tidal and v_esc, so offsets and dispersion go as M**(1/3).
    N = len(orb_sat)
    x, y, z, vx, vy, vz = orb_sat.T
    R, L, r = get_rot_mat(x, y, z, vx, vy, vz)
    d2Phi_dr2 = get_d2Phi_dr2(pot_host, x, y, z, t=t)
    mass_sat = np.broadcast_to(np.asarray(mass_sat, dtype=float), (N,))   # scalar -> same value per point

    # compute the tidal radius at this radius for each point on the trajectory
    Omega = L / r**2
    r_tidal = (agama.G * mass_sat / (Omega**2 - d2Phi_dr2))**(1./3)

    # assign positions and velocities (in the satellite reference frame) of particles
    r_tidal = np.repeat(r_tidal, 2)
    mass_sat = np.repeat(mass_sat, 2)   # one per particle (2 per point), to line up with Dr below

    mean = np.array([1.6, -30, 0, 1, 20, 0])

    cov = np.array([
        [0.1225,   0,   0, 0, -4.9,   0],
        [     0, 529,   0, 0,    0,   0],
        [     0,   0, 144, 0,    0,   0],
        [     0,   0,   0, 0,    0,   0],
        [  -4.9,   0,   0, 0,  400,   0],
        [     0,   0,   0, 0,    0, 484],
    ])
    
    posvel = rng.multivariate_normal(mean, cov, size=2*N)

    Dr = posvel[:, 0] * r_tidal
    v_esc = np.sqrt(2 * agama.G * mass_sat / Dr)
    Dv = posvel[:, 3] * v_esc

    # convert degrees to radians
    phi = posvel[:, 1] * np.pi / 180
    theta = posvel[:, 2] * np.pi / 180
    alpha = posvel[:, 4] * np.pi / 180
    beta = posvel[:, 5] * np.pi / 180

    dx = Dr * np.cos(theta) * np.cos(phi)
    dy = Dr * np.cos(theta) * np.sin(phi)
    dz = Dr * np.sin(theta)
    dvx = Dv * np.cos(beta) * np.cos(alpha)
    dvy = Dv * np.cos(beta) * np.sin(alpha)
    dvz = Dv * np.sin(beta)

    dq = np.column_stack([dx, dy, dz])
    dp = np.column_stack([dvx, dvy, dvz])
    
    ic_stream = np.tile(orb_sat, 2).reshape(2*N, 6)
    
    # trailing arm
    ic_stream[::2,0:3] += np.einsum('ni,nij->nj', dq[::2], R)
    ic_stream[::2,3:6] += np.einsum('ni,nij->nj', dp[::2], R)
    
    # leading arm
    ic_stream[1::2,0:3] += np.einsum('ni,nij->nj', -dq[1::2], R)
    ic_stream[1::2,3:6] += np.einsum('ni,nij->nj', -dp[1::2], R)

    return ic_stream
# %%
# functions for tuning the perturbation
def impact_site_triad(x_s, v_s):
    """
    orthonormal triad at an impact site, built exactly as vedant's catalog builds it
    (data/'Stream Encounter Reconstruction.pdf', "The frame"):
        e1 = v_hat                                along the stream's motion
        e2 = r_hat with the e1 component removed  outward radial, in the orbital plane
        e3 = e1 x e2                              right-handed
    bhat_t/bhat_r/bhat_L and what_t/what_r/what_L are components on (e1,e2,e3) in order.

    !! e3 is ANTIPARALLEL to L despite the `_L` name -- e3 = -(x cross v)/|x cross v|.
    getting it backwards mirrors the encounter through the orbital plane, flipping the
    sign of the kick in phi2/pm_phi2, and no magnitude-only check will catch it.
    """
    e1 = v_s / np.linalg.norm(v_s) # normalize e1
    r_hat = x_s / np.linalg.norm(x_s) # define r_hat
    e2 = r_hat - np.dot(r_hat, e1)*e1 # subtract v-hat component out of rhat to get e2
    n = np.linalg.norm(e2)
    if n < 1e-8:                       # handle the case of purely radial motion; "outward in-plane" undefined
        e2 = np.cross(e1, [0., 0., 1.])
        n = np.linalg.norm(e2)
        if n < 1e-8:
            e2 = np.cross(e1, [1., 0., 0.])
            n = np.linalg.norm(e2)
    return e1, e2/n, np.cross(e1, e2/n)

def get_perturber_location(
        xv_stream_today, # <-- location in galactocentric space for the perturber location to be today
        b, v, #<-- definitely need magnitudes of the impact parameter and relative velocity. 
        t_impact, #<-- impact time (in Myr before present)
        pot_host, #<-- host potential for back-integration purposes
        bhat_trl=None, what_trl=None, #<-- might want to specify b, v vectors in the stream-aligned 'trl' (tangential, radial, angular momentum) frame
        b_angle=None, v_angle=None #<-- or, might want to specify b, v vectors by (b) angle to r-hat and (v) angle to the stream velocity vector @ impact site
):
    """
    returns perturber phase space position at closest approach. 
    will feed this into create_stream via a perturber_dict. 

    xv_stream_today: 6x1 six-vector of position, velocity on stream track today. 
    b: perturbation impact parameter
    b_angle: impact parameter angle relative to rhat (position of back-integrated pert site)
    v: relative speed of perturber
    v_angle: angle of perturber motion relative to back-integrated stream velocity
    t_impact: time to back integrate
    pot_host: potential to back integrate in
    """

    # STEP 1: back-integrate xv_stream_today by t_impact within pot_host
    # to get "xv_impact_location" -- this is the location on the stream that is most closely approached. 
    xv_impact_location = np.asarray(agama.orbit(
        potential=pot_host, ic=xv_stream_today, time=-abs(t_impact), trajsize=1)[1][0], dtype=float)



    # STEP 1.5: get 'r_hat' the galactocentric direction of the xv_impact_location position, 
    #   as well as v_hat, the direction of v in xv_impact_location velocity.
    x_s, v_s = xv_impact_location[:3], xv_impact_location[3:]
    e1, e2, e3 = impact_site_triad(x_s, v_s) #<-- call the impact site triad function. 

    if bhat_trl is not None: # in the case where I specified impact triads (as in vedant's catalog)
        b_hat = bhat_trl @ np.vstack([e1, e2, e3])
        v_rel_hat = what_trl @ np.vstack([e1,e2,e3])

    else:
        if v_angle is None or b_angle is None:
            raise ValueError("if not specifying impact triads, specify v_angle, b_angle (less rigorous)")
        b_hat = np.cos(b_angle)*e2 + np.sin(b_angle)*e3 # <-- perpendicular to velocity, along r and L
        v_rel_hat = np.cos(v_angle)*e1 + np.sin(v_angle)*(np.cross(b_hat, e1))



    # STEP 4: basically we're done, return xv_perturber_flyby,
    #    which is xv_impact_location plus the impact parameter 
    #    vector plus the perturber relative velocity vector
    xv_perturber_flyby = np.empty(6)
    xv_perturber_flyby[:3] = x_s + b*b_hat
    xv_perturber_flyby[3:] = v_s + v*v_rel_hat

    return xv_perturber_flyby

def along_track_angle(xv, xv_ref, L_hat):
    """
    orbit angle ahead/behind prog. this is basically 
    phi1 from jake's stream frame. eventually could 
    just put stuff in streamframe coords and literally 
    use phi1 i guess. 
    """
    xv = np.atleast_2d(np.asarray(xv, dtype=float))
    x = xv[:, 0:3]
    xp = np.asarray(xv_ref[0:3], dtype=float)
    xp_hat = xp - np.dot(xp, L_hat) * L_hat
    xp_hat /= np.linalg.norm(xp_hat)
    u = x - np.outer(x @ L_hat, L_hat)              # positions projected into the orbital plane
    return np.arctan2(np.cross(np.broadcast_to(xp_hat, u.shape), u) @ L_hat, u @ xp_hat)

def get_xv_stream_today(unperturbed_stream, xv_progenitor,
                        phi1_today=None,
                        frac_along_tail=None,
                        lead_trail='lead',
                        tip_percentile=99., window_frac=0.05, window_deg=None,
                        min_count=8):
    """
    returns the phase space location of the perturbation today given an unperturbed stream.

    works in jake's stream frame (`get_streamcoords`), so the along-track coordinate
    here IS phi1 in degrees, with the same sign convention as everywhere else in the
    repo: **phi1 > 0 is the leading tail, phi1 < 0 is the trailing tail** (streamframe's
    y_hat = L_hat x x_hat points along the progenitor's direction of motion).

    two ways to say where you want the perturbation, pick one:
      phi1_today:       signed phi1 in degrees. the sign picks the tail, so
                        lead_trail is ignored (and -20 is 20 deg down the trailing tail).
      frac_along_tail:  fraction of the way out to the tip of the tail named by
                        lead_trail. the old interface; kept so existing perturber
                        dicts still work.

    unperturbed_stream: stream particle six-vectors today, Mx6, kpc & km/s
    xv_progenitor:      present-day progenitor six-vector -- this DEFINES the frame
    lead_trail:         'lead' or 'trail'; only used with frac_along_tail
    tip_percentile:     which percentile of the tail's phi1 extent counts as "the tip".
                        99 rather than 100 so one escaped particle cannot define the scale.
    window_frac:        half-width of the averaging window, as a fraction of the tip phi1
    window_deg:         half-width in degrees instead, if you would rather set it absolutely.
                        overrides window_frac.
    min_count:          grow the window until at least this many particles fall in it
    """
    stream = np.asarray(unperturbed_stream, dtype=float)
    xv_p   = np.asarray(xv_progenitor, dtype=float).reshape(6)

    # phi1 of every stream particle, in the frame the progenitor defines (deg)
    _, coords = get_streamcoords(None, stream, prog_w=xv_p)
    phi1 = coords['phi1']
    if np.any(np.abs(phi1) > 162.):
        print('WARNING: stream reaches |phi1| > 162 deg; phi1 is an arctan2 and is close '
              'to wrapping, so the lead/trail split is no longer reliable')

    # work in psi = (sign of the requested tail) * phi1, so psi > 0 is always the tail we want
    if phi1_today is not None:
        if phi1_today == 0.:
            raise ValueError("phi1_today=0 is the progenitor itself; pick a sign "
                             "(>0 leading, <0 trailing)")
        sign = np.sign(phi1_today)
        psi_target = abs(float(phi1_today))
        tail_name = 'lead' if sign > 0 else 'trail'
    else:
        if frac_along_tail is None or lead_trail is None:
            raise ValueError("must supply frac_along_tail and lead_trail if phi1_today is None")
        if lead_trail not in ('lead', 'trail'):
            raise ValueError("lead_trail must be 'lead' or 'trail', got %r" % (lead_trail,))
        sign = +1.0 if lead_trail == 'lead' else -1.0
        psi_target = None       # set below, once we know where the tip is
        tail_name = lead_trail

    psi = sign * phi1
    if not np.any(psi > 0):
        raise ValueError('no stream particles on the %s tail at all' % tail_name)
    psi_tip = np.percentile(psi[psi > 0], tip_percentile)

    if psi_target is None:
        psi_target = np.clip(frac_along_tail, 1e-3, 1.0) * psi_tip   # clipped off 0 to stay in one tail
    elif psi_target > psi_tip:
        print('WARNING: requested phi1_today=%g deg is past the %s tail tip '
              '(|phi1| ~ %.1f deg); the window will just pick up the tip particles'
              % (phi1_today, tail_name, psi_tip))

    # average the particles in a window around psi_target; widen it if the tail is sparse there
    half = window_deg if window_deg is not None else window_frac * psi_tip
    for _ in range(8):
        sel = (psi > 0) & (np.abs(psi - psi_target) < half)      # (psi > 0) keeps the other tail out
        if sel.sum() >= min_count:
            break
        half *= 1.6
    if sel.sum() == 0:
        raise ValueError('no stream particles within %g deg of phi1 = %g on the %s tail'
                         % (half, sign * psi_target, tail_name))
    return stream[sel].mean(axis=0)
# %%
def evolve_snapshots(pot_tot, ic_stream, time_seed, snap_times, time_end,
                     verbose=True, breaks=None):
    """
    for integrating every stream particle and outputting at common snapshot times

    breaks: optional times at which every orbit is stopped and restarted. agama's integrator
            has no max timestep, so a perturber that only exists for a few Myr can fall entirely
            between two steps and never be felt. a fine grid of breaks across the perturber's
            window caps the step there. None (the default) is the plain one-call integration.

    the integration runs a relative 1e-9 past time_end: time_seed + (time_end - time_seed) can
    land a hair below time_end in floating point, and then orb(time_end) is outside the
    interpolator and comes back NaN (this used to drop 10-45% of particles today).
    """
    time_end_pad = time_end + 1e-9 * (time_end - np.min(time_seed))
    if breaks is not None:
        return _evolve_snapshots_breaks(pot_tot, ic_stream, time_seed, snap_times, time_end_pad,
                                        breaks, verbose=verbose)
    orbits = agama.orbit(potential=pot_tot, ic=ic_stream,
                         time=time_end_pad-time_seed, timestart=time_seed,
                         dtype=object, verbose=False)
    out=np.full((len(snap_times), len(ic_stream), 6), np.nan)

    if verbose==True:
        iterator = enumerate(tqdm(orbits, desc='sampling orbits'))
    if verbose==False:
        iterator = enumerate(orbits)
    for i, orb in iterator:
        live = snap_times>=time_seed[i]
        if live.any():
            out[live, i] = orb(snap_times[live]) #<-- this should be interpolating to get the position at this time. 
        
    return out


    # n_snap, M = len(snap_times), len(ic_stream) # number of snaps, number of particles at end
    # out = np.full((n_snap, M, 6), np.nan) #<-- fill with nans and populate as particles are created
    # for k, tk in enumerate(tqdm(snap_times, desc='snapshots')):
    #     born = time_seed <= tk #<-- has been born flag
    #     if not born.any():
    #         continue
    #     r=np.asarray(agama.orbit(
    #         potential=pot_tot, ic=ic_stream[born],
    #         time=tk-time_seed[born], 
    #         timestart=time_seed[born], trajsize=1
    #     ), dtype=object)

    #     out[k, born] = np.vstack(r[:,1])
    # return out


def _evolve_snapshots_breaks(pot_tot, ic_stream, time_seed, snap_times, time_end, breaks,
                             verbose=True):
    """
    evolve_snapshots, but integrating in legs between consecutive break times. a particle's
    state is carried leg to leg in float64; legs with no snapshot in them only keep the endpoint.
    time_end comes in already padded. a snapshot sitting exactly on a break can come back NaN
    from the leg ending there, but the next leg starts on it and overwrites it.
    """
    edges = np.append(np.sort(breaks[(breaks > np.min(time_seed)) & (breaks < time_end)]), time_end)
    xv = np.array(ic_stream, dtype=float)
    t_now = np.array(time_seed, dtype=float)
    out = np.full((len(snap_times), len(ic_stream), 6), np.nan)
    tiny = 1e-12 * (time_end - np.min(time_seed))

    for E in (tqdm(edges, desc='integrating legs') if verbose else edges):
        go = np.flatnonzero(t_now < E)   # born, and not already at E
        # a particle born a hair before a break has a leg too short for agama to take a step in;
        # just move its clock on (and fill any snapshot in that sliver with its current state)
        short = go[E - t_now[go] < tiny]
        for i in short:
            out[(snap_times >= t_now[i]) & (snap_times <= E), i] = xv[i]
        t_now[short] = E
        go = np.setdiff1d(go, short)
        # NaN ICs (create_ic_chen25 can make a few) stay NaN, as in the one-call integration
        bad = go[~np.isfinite(xv[go]).all(axis=1)]
        t_now[bad] = E
        go = np.setdiff1d(go, bad)
        if len(go) == 0:
            continue
        has_snaps = np.any((snap_times >= t_now[go].min()) & (snap_times <= E))
        if has_snaps:
            orbits = agama.orbit(potential=pot_tot, ic=xv[go], time=E-t_now[go], timestart=t_now[go],
                                 dtype=object, verbose=False)
            if not isinstance(orbits, np.ndarray):   # a lone orbit comes back bare, not in an array
                orbits = [orbits]
            for k, i in enumerate(go):
                orb = orbits[k]
                live = (snap_times >= t_now[i]) & (snap_times <= E)
                if live.any():
                    out[live, i] = orb(snap_times[live])
                xv[i] = orb(orb[len(orb)-1])
        else:
            _, x_end = agama.orbit(potential=pot_tot, ic=xv[go], time=E-t_now[go], timestart=t_now[go],
                                   trajsize=1, dtype='float64', separateTime=True, verbose=False)
            xv[go] = np.reshape(x_end, (len(go), 6))
        t_now[go] = E
    return out


def orbit_with_breaks(potential, ic, timestart, time, trajsize, breaks=None):
    """
    agama.orbit(potential, ic, time=time, timestart=timestart, trajsize=trajsize) for ONE orbit,
    returning (times, traj) on the same regular grid, but stopped and restarted at `breaks`
    so the integrator can't step over a short-lived perturber (see evolve_snapshots).
    works in either time direction. breaks=None is exactly the plain agama call.
    """
    if breaks is None:
        return agama.orbit(potential=potential, ic=ic, time=time, timestart=timestart, trajsize=trajsize)
    # take the output grid from agama itself (one cheap orbit): it matches no simple linspace/arange
    # bit for bit, and time_seed values landing a hair either side of a snapshot time change
    # which particles count as born in that snapshot.
    t_grid = agama.orbit(potential=potential, ic=ic, time=time, timestart=timestart, trajsize=trajsize)[0]
    sgn = np.sign(time)
    t_stop = timestart + time + sgn * 1e-9 * abs(time)   # padded so the last grid point isn't lost to roundoff
    inside = breaks[(sgn*(breaks - timestart) > 0) & (sgn*(t_stop - breaks) > 0)]
    edges = np.append(np.sort(inside)[::int(sgn)], t_stop)

    traj = np.full((trajsize, 6), np.nan)
    traj[0] = ic
    x, t = np.array(ic, dtype=float), timestart
    for E in edges:
        orb = agama.orbit(potential=potential, ic=x, time=E-t, timestart=t, dtype=object)
        g = np.flatnonzero((sgn*(t_grid - t) >= 0) & (sgn*(E - t_grid) >= 0))
        if len(g):
            vals = orb(t_grid[g])
            ok = np.isfinite(vals).all(axis=1)   # grid points on a break come back from one leg or the other
            traj[g[ok]] = vals[ok]
        x, t = orb(orb[len(orb)-1]), E
    return t_grid, traj


def full_perturber_orbit(pot_host, xv_perturber_flyby, time_flyby, time_begin, time_end):
    """
    the perturber orbit over the whole run, integrated backwards and forwards out of the
    flyby point (so the flyby is hit exactly, not by a long integration that arrives there).

    returns
        traj:       (N, 7) [t, x, y, z, vx, vy, vz], increasing t, every integrator step --
                    the table for the potential's center= modifier
        orbit_fn:   callable t -> (len(t), 6), agama's interpolant of the same integrations,
                    for resampling onto whatever grid you want to plot on
    both spans are padded by 1e-9 of the run so the ends aren't lost to roundoff
    (see the evolve_snapshots NaN issue in the readme).
    """
    t_lo, t_hi = min(time_begin, time_flyby), max(time_end, time_flyby)
    pad = 1e-9 * (t_hi - t_lo)
    pieces, orb_back, orb_fwd = [], None, None
    if time_flyby > t_lo:
        span = (t_lo - pad) - time_flyby
        t_b, x_b = agama.orbit(potential=pot_host, ic=xv_perturber_flyby,
                               time=span, timestart=time_flyby, trajsize=0)
        pieces.append(np.column_stack([t_b, x_b])[::-1])   # back to increasing t, ends on the flyby
        orb_back = agama.orbit(potential=pot_host, ic=xv_perturber_flyby,
                               time=span, timestart=time_flyby, dtype=object)
    if t_hi > time_flyby:
        span = (t_hi + pad) - time_flyby
        t_f, x_f = agama.orbit(potential=pot_host, ic=xv_perturber_flyby,
                               time=span, timestart=time_flyby, trajsize=0)
        traj_f = np.column_stack([t_f, x_f])
        pieces.append(traj_f[1:] if pieces else traj_f)     # don't repeat the flyby point
        orb_fwd = agama.orbit(potential=pot_host, ic=xv_perturber_flyby,
                              time=span, timestart=time_flyby, dtype=object)
    traj = np.vstack(pieces)

    def orbit_fn(t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        use_back = (t < time_flyby) if orb_fwd is not None else np.ones(len(t), dtype=bool)
        out = np.empty((len(t), 6))
        if use_back.any():
            out[use_back] = orb_back(t[use_back])
        if (~use_back).any():
            out[~use_back] = orb_fwd(t[~use_back])
        return out

    return traj, orbit_fn


def smoothstep(x):
    x = np.clip(x, 0., 1.)
    return x*x*(3. - 2.*x)


def perturber_mass_switch(time_flyby, window, ramp, t_lo, t_hi, n_ramp=20, n_flat=5):
    """
    table for agama's time-dependent scale= modifier, columns [t, mass factor A, size factor S]:
    A = 1 for |t - time_flyby| <= window/2, smoothstep ramps of length `ramp` outside each end,
    A = 0 beyond them; S = 1 throughout.

    the table is a cubic spline in t that extrapolates linearly past its ends, so it runs on
    flat nodes from before t_lo to after t_hi (and past the ramps, if the window hangs off the
    run). with the ramps densely sampled A comes out exactly 0 / 1 on the flat parts and stays
    in [0, 1] on the ramps -- no ringing.
    """
    if not ramp > 0:
        raise ValueError("ramp has to be > 0 (repeated spline nodes otherwise)")
    half = window / 2.
    up0, up1 = time_flyby - half - ramp, time_flyby - half      # switching on
    dn0, dn1 = time_flyby + half, time_flyby + half + ramp      # switching off
    pad = 0.01 * (t_hi - t_lo) + ramp
    lo, hi = min(t_lo, up0) - pad, max(t_hi, dn1) + pad

    t_nodes = np.concatenate([np.linspace(lo,  up0, n_flat)[:-1],
                              np.linspace(up0, up1, n_ramp)[:-1],
                              np.linspace(up1, dn0, n_flat)[:-1],
                              np.linspace(dn0, dn1, n_ramp)[:-1],
                              np.linspace(dn1, hi,  n_flat)])
    A = np.where(t_nodes < time_flyby,
                 smoothstep((t_nodes - up0) / ramp),
                 smoothstep((dn1 - t_nodes) / ramp))
    return np.column_stack([t_nodes, A, np.ones_like(t_nodes)])


def release_orbit(potential, posvel_today, time_begin, time_end, time_dis, n_release, breaks=None):
    """
    the progenitor's phase-space position at n_release release times, evenly spaced from
    time_begin to time_dis (the dissolution time, <= time_end), for a progenitor that is at
    posvel_today at time_end. integrated back from today to time_dis, then from there back to
    time_begin on a regular grid (orbit_with_breaks, so breaks= works as for the progenitor).

    only needed when time_dis < time_end: otherwise the release points are just the orbit grid.
    returns (t_release, xv_release), increasing in time.
    """
    _, xv_dis = orbit_with_breaks(potential, ic=posvel_today, timestart=time_end,
                                  time=time_dis - time_end, trajsize=2, breaks=breaks)
    t_rel, xv_rel = orbit_with_breaks(potential, ic=xv_dis[-1], timestart=time_dis,
                                      time=time_begin - time_dis, trajsize=n_release, breaks=breaks)
    return t_rel[::-1], xv_rel[::-1]


def progenitor_mass_table(time_begin, time_dis, dt_release, mass_loss, t_lo, t_hi, n_dense=20, n_flat=5):
    """
    table for agama's time-dependent scale= modifier on the progenitor potential, columns
    [t, mass factor A, size factor S=1]. A is the progenitor's mass over its initial mass.
    the mass is all gone at time_gone = time_dis + dt_release, one release interval after the
    last pair goes out at time_dis (so each pair carries 1/n_release of the initial mass).

    mass_loss=True:  A = (time_gone - t)/(time_gone - time_begin), falling linearly from 1 at
                     time_begin to 0 at time_gone, and 0 after. the line is continued (not
                     clipped) before time_begin, so its only kink is at time_gone.
    mass_loss=False: A = 1 until time_dis, smoothstep down to 0 by time_gone, 0 after.

    the table is a cubic spline: it reproduces the straight/flat parts exactly and only rings
    at a kink, so the nodes are dense (n_dense per dt_release) for one release interval either
    side of [time_dis, time_gone].
    """
    time_gone = time_dis + dt_release
    pad = 0.01 * (t_hi - t_lo) + dt_release
    lo, hi = t_lo - pad, max(t_hi, time_gone + dt_release) + pad
    d0, d1 = time_dis - dt_release, time_gone + dt_release
    t_nodes = np.concatenate([np.linspace(lo, d0, n_flat)[:-1],
                              np.linspace(d0, d1, 3*n_dense)[:-1],
                              np.linspace(d1, hi, n_flat)])
    if mass_loss:
        A = np.clip((time_gone - t_nodes) / (time_gone - time_begin), 0., None)
    else:
        A = smoothstep((time_gone - t_nodes) / dt_release)
    return np.column_stack([t_nodes, A, np.ones_like(t_nodes)])


def moving_progenitor(pot_sat, traj, scale=None):
    """pot_sat centred on traj ([t, x, y, z, vx, vy, vz] rows); scale=None keeps its mass fixed."""
    if scale is None:
        return agama.Potential(potential=pot_sat, center=traj)
    return agama.Potential(potential=pot_sat, center=traj, scale=scale)


# final stream creation (chen + vasiliev + perturber dict stuff)
def create_stream(seed, time_total, num_particles, 
                  pot_host, posvel_sat, mass_sat, pot_sat=None, 
                  add_perturber=True, perturber_dict=None, update_prog_orbit=False,
                  gen_unperturbed=True, xv_stream_input=None,
                  n_snapshots=None,
                  create_ic_method=create_ic_chen25,
                  return_flyby_location=False,
                  tdis=0., mass_loss=False,
                   verbose=True, **kwargs):
    """
    TODO: in perturbed case, perhaps the progenitor orbit should respond to the perturber.

    frankensteining vasiliev+chen functions so that
    I can direct impacts to a chen stream.

    seed:                           integer seed for np random number generator
    update_prog_orbit:              whether to re-integrate orbit_sat
    num_particles:                  how many particles to release
    pot_host:                       host potential
    posvel_sat:                     specified satellite position (at final time or present day or whatever)
    mass_sat:                       satellite mass (for chen ps)
    pot_sat:                        potential of stream progenitor
    add_perturber:                  whether to add a perturber (if so returns perturbed and unperturbed streams)
    perturber_dict:                 perturbation properties; see below
    update_prog_orbit:              whether to re-integrate progenitor orbit including perturber potential
    n_snapshots:                    how many snapshots to output; evenly-spaced in time
    create_ic_method:               should just be create_ic_chen25 but in principle could sub in a different stream df generator
    tdis:                           lookback time (agama units, like time_total; e.g. 0.1/timeUnitGyr) at which the
                                    progenitor dissolved. the num_particles//2 release points are spread evenly from
                                    -time_total to tdis ago instead of to today, and nothing is released after that.
                                    the progenitor orbit (orbit_sat etc.) still covers the whole run on the same grid,
                                    so snapshots line up with it as before. 0 (default) = releasing until today.
    mass_loss:                      if True, mass_sat is the INITIAL mass, and the mass handed to create_ic_method
                                    falls linearly, mass_sat*(1 - i/n_release) for release point i, so each pair carries
                                    mass_sat/n_release and the progenitor is empty one release interval after the last
                                    release. False (default) = constant mass_sat.
                                    either option (tdis>0 or mass_loss) also puts the same mass history on pot_sat
                                    (see progenitor_mass_table); with mass_loss=False it switches off right after tdis.
    verbose:                        to be verbose or not to be verbose.

    perturber dict example:

    perturber_dict = {
        "mass": 1e6, # Msun?
        "scaleRadius": 0.05, # kpc?
        't_impact_fraction':0.5,
        'b':0.00,#1, #kpc
        'b_angle':np.pi/4,
        'v':50., #km/s?
        'v_angle':np.pi/2, #<--ie very perpendicular to the stream velocity. 
        't_impact':0.3/timeUnitGyr, #<-- idk
        'phi1_today':-20., #<-- where perturbed region should end up today, in deg of
                           #    stream-frame phi1. >0 leading tail, <0 trailing tail.
    }

    instead of 'phi1_today' you can still use the older pair
        'frac_along_tail':0.2, 'lead_trail':'lead'
    which puts the perturbation that fraction of the way out to the tip of the named tail.
    supply one or the other; phi1_today wins if both are present.

    the perturber only exists for the Lu+26 impact window, 4*sqrt(rs**2+b**2)/v centred on the
    flyby, plus smoothstep ramps outside each end. optional keys controlling that:
        'window_factor':4., # the 4 in Lu+26's window
        'ramp_frac':0.25,   # ramp length as a fraction of the window
        'pop':True,         # False keeps the perturber at full mass on its orbit for the whole run
        'n_breaks':33,      # integration breakpoints across window + ramps (see evolve_snapshots)

    time_total may be either sign: negative integrates the progenitor back from today,
    positive integrates it forward. Either way the returned arrays are ordered in
    increasing time, and "the end of the window" is the epoch the stream is observed at.

    Returns:
        time_sat, orbit_sat:            progenitor times (increasing) and Nx6 posvel
        xv_stream:                      Mx6 stream particles, unperturbed
        xv_stream_perturbed:            Mx6 stream particles with the subhalo (None if add_perturber=False)
        time_perturber, orbit_perturber: perturber orbit on the SAME time grid as time_sat
        ic_stream:                      Mx6 seed initial conditions
    """
    # time_total is a positive number. we are back-integrating so that the returned time_sat
    # will go from time_total to zero
    time_sat, orbit_sat = agama.orbit(potential=pot_host, ic=posvel_sat,
                                    time=-time_total, timestart=time_total,
                                    trajsize=num_particles//2)
    
    # then reverse the times and orbits so that they go from zero to time_total. 
    time_sat  = time_sat[::-1]
    orbit_sat = orbit_sat[::-1]
 
    
    
    if time_total <= 0:
        # reverse the arrays to make them increasing in time
        # time_sat  = time_sat [::-1]
        # orbit_sat = orbit_sat[::-1]
        # print("WARNING! time negative; perturbation time will be messed up. ")
        raise ValueError("please use a time_total > 0 ;-;")

    # the simulated time window, always in increasing order whatever the sign of time_total:
    # time_total<0 -> [time_total, 0] (present day at the end, t=0)
    # time_total>0 -> [0, time_total] (present day at the end, t=time_total)
    # everything below is expressed in terms of these two, so no sign juggling is needed.
    time_begin, time_end = time_sat[0], time_sat[-1] #< time begin is zero, time_end is time_total

    # dissolution + mass loss. the release points are decoupled from the orbit grid: orbit_sat
    # always spans the whole run (it centres pot_sat, and snapshots/movies index it), while the
    # n_release release points span [time_begin, time_dis]. with tdis=0 they ARE the orbit grid,
    # so the default is unchanged bit for bit.
    if not 0. <= tdis < time_end - time_begin:
        raise ValueError("tdis has to be in [0, time_total)")
    n_release = num_particles//2
    time_dis = time_end - tdis
    dt_release = (time_dis - time_begin) / (n_release - 1)
    if tdis > 0:
        time_release, orbit_release = release_orbit(pot_host, posvel_sat, time_begin, time_end,
                                                    time_dis, n_release)
    else:
        time_release, orbit_release = time_sat, orbit_sat
    # the progenitor's mass just before releasing pair i: each pair takes mass_sat/n_release
    mass_release = mass_sat * (1. - np.arange(n_release)/n_release) if mass_loss else mass_sat
    # the same mass history for the progenitor potential (None = constant mass, as before)
    if tdis > 0 or mass_loss:
        scale_sat = progenitor_mass_table(time_begin, time_dis, dt_release, mass_loss,
                                          t_lo=time_begin, t_hi=time_end)
    else:
        scale_sat = None

    # at each release point, create a pair of seed initial conditions
    # for particles released at Lagrange points
    rng=np.random.default_rng(seed) #<-- seed here
    ic_stream = create_ic_method(rng,
                                 pot_host,
                                 orbit_release,
                                 mass_release, **kwargs)
    time_seed = np.repeat(time_release, 2) #<-- when to release points.

    if pot_sat is None:
        pot_tot = pot_host
    else:
        # include the progenitor's potential
        traj = np.column_stack([time_sat, orbit_sat])
        pot_traj = moving_progenitor(pot_sat, traj, scale_sat)
        pot_tot = agama.Potential(pot_host, pot_traj)
        
    if n_snapshots is None:
        snap_times = np.array([time_end])              # present day only, as before
    elif np.isscalar(n_snapshots):
        snap_times = np.linspace(time_begin, time_end, int(n_snapshots))
    else:
        snap_times = np.asarray(n_snapshots, dtype=float)   # or pass your own epochs


    if gen_unperturbed==True:
        xv_snaps = evolve_snapshots(pot_tot, ic_stream, time_seed, snap_times, time_end,verbose=verbose)
        xv_stream = xv_snaps[-1]   # present day; bit-identical to the old trajsize=1 result
        
        ### use the unperturbed stream we generated to get perturbation we want:


    if gen_unperturbed==False:
        xv_snaps=None
        if xv_stream_input is None:
            raise ValueError("if not generating an unperturbed stream, pass one as an input xv_stream_input=...")
        
        xv_stream=xv_stream_input


    # Next, we'll do stuff with the perturber. 
    if add_perturber:
        # this is the location on the present day stream that I want the perturbation 
        # to show up at. 
        # 'phi1_today' (signed, deg) is the preferred way to say where the perturbation
        # lands; 'frac_along_tail' + 'lead_trail' is the older interface and still works.
        if perturber_dict.get('phi1_today') is None and perturber_dict.get('frac_along_tail') is None:
            raise ValueError("perturber_dict needs either 'phi1_today' (deg, >0 leading, "
                             "<0 trailing) or 'frac_along_tail' + 'lead_trail'")
        xv_stream_today = get_xv_stream_today(unperturbed_stream=xv_stream,
                                              xv_progenitor = orbit_sat[-1],
                                              phi1_today=perturber_dict.get('phi1_today'),
                                              frac_along_tail=perturber_dict.get('frac_along_tail'),
                                              lead_trail=perturber_dict.get('lead_trail', 'lead'))

        # get the position and velocity of the perturber at closest approach:
        xv_perturber_flyby = get_perturber_location(xv_stream_today, #<-- possibly an issue, as this is being back-integrated to a location of closest approach without accounting for the perturber. 
                                                    b=perturber_dict['b'], b_angle=perturber_dict.get('b_angle'),
                                                    v=perturber_dict['v'], v_angle=perturber_dict.get('v_angle'),
                                                    bhat_trl = perturber_dict.get('b_trl'), what_trl = perturber_dict.get('v_trl'),
                                                    t_impact=perturber_dict['t_impact'],
                                                    pot_host=pot_host)
        time_flyby = time_end - perturber_dict['t_impact'] # <-- when the impact is in the grand scheme of times. time_end is time_total. 


        #################################################
        ########### THE PERTURBER ONLY EXISTS FOR THE IMPACT TIME.
        # the impact duration is from Lu+26, depends on the perturber size, impact parameter, and relative speed.
        # 'window_factor' is the 4 in Lu+26; see the readme before trusting the default.
        impact_duration = perturber_dict.get('window_factor', 4.) * np.sqrt(perturber_dict.get('scaleRadius')**2 + perturber_dict.get('b')**2) / perturber_dict.get('v')
        impact_duration_Gyr = impact_duration * (u.kpc/(u.km/u.s)).to(u.Gyr)
        impact_duration_nbodyUnits = impact_duration_Gyr / timeUnitGyr #<--

        # the perturber orbit covers the WHOLE run, integrated both ways out of the flyby point so
        # the flyby itself is exact. a trajectory that only spans the impact window does NOT switch
        # the potential off outside it -- agama extrapolates the centre in a straight line at full mass.
        # the switching is done by the mass scale table below instead.
        traj_perturber, orbit_perturber_fn = full_perturber_orbit(
            pot_host, xv_perturber_flyby, time_flyby, time_begin, time_end)

        # mass factor A(t): 1 for the Lu+26 window centred on t_flyby, 0 outside, with smoothstep
        # ramps of ramp_frac*window tacked on outside each end. 'pop':False keeps A=1 throughout
        # (the perturber on its real orbit for the whole run), for checking what the window does.
        if perturber_dict.get('pop', True):
            ramp = perturber_dict.get('ramp_frac', 0.25) * impact_duration_nbodyUnits
            scale_perturber = perturber_mass_switch(
                time_flyby, impact_duration_nbodyUnits, ramp=ramp, t_lo=time_begin, t_hi=time_end)
            # stop/restart every stream orbit on a fine grid across the window + ramps, so the
            # integrator can't step clean over a perturber that's only around for a few Myr.
            half_on = impact_duration_nbodyUnits/2 + ramp
            breaks = np.linspace(time_flyby - half_on, time_flyby + half_on,
                                 perturber_dict.get('n_breaks', 33))
        else:
            scale_perturber = [1., 1.]
            breaks = None
        #################################################


        # and finally, create the NFW* potential of the perturber moving on its orbit
        # *option for perturber to be a regular NFW*
        # pot_perturber = agama.Potential(type='nfw', mass=perturber_dict['mass'], scaleRadius=perturber_dict['scaleRadius'],
        #                                 center=traj_perturber) #<-- center of perturber potential moves with its orbit.
        # *option for perturber to be a truncated nfw
        pot_perturber = agama.Potential(type='Spheroid',
                                        scaleRadius=perturber_dict['scaleRadius'],
                                        mass=perturber_dict['mass'],
                                        outerCutoffRadius=perturber_dict['truncationRadius'],
                                        gamma=1, beta=3, alpha=1,
                                        center=traj_perturber,
                                        scale=scale_perturber
                                        )


        # the same perturber orbit, re-sampled on the progenitor's time grid: this is the one to plot,
        # since it lines up point-for-point with (time_sat, orbit_sat) for direct comparison.
        # it's the same integration the potential is centred on, just interpolated, so the plotted
        # orbit and the one the stream feels can't disagree.
        time_perturber = time_sat.copy()
        orbit_perturber = orbit_perturber_fn(time_perturber)


        # new potential external to the satellite cluster includes the host + the perturber. 
        pot_ext = agama.Potential(pot_host, pot_perturber)

        if update_prog_orbit: # might want to re-integrate progenitor orbit including the perturber! 
            # <-- note that in this case the placement of the present-dat location of the perturbation
            #     is not guaranteed. 

            #### here i also want to change -- back-integrate from a fixed present-day position ??
            # same breaks as the stream particles: a single agama.orbit call can step clean over
            # a short window, which leaves the progenitor unperturbed (update_prog_orbit=False in effect)
            time_sat_perturbed, orbit_sat_perturbed = orbit_with_breaks(pot_ext,
                                                                    ic=posvel_sat,
                                                                    time=time_begin - time_end,
                                                                    timestart=time_end,
                                                                    trajsize=num_particles//2,
                                                                    breaks=breaks)
            time_sat_perturbed  = time_sat_perturbed[::-1]     # already absolute -- no + time_total
            orbit_sat_perturbed = orbit_sat_perturbed[::-1]


            if pot_sat is None:
                pot_total = pot_ext
            else:
                # include the progenitor's potential
                traj_perturbed = np.column_stack([time_sat_perturbed, orbit_sat_perturbed])
                pot_traj_perturbed = moving_progenitor(pot_sat, traj_perturbed, scale_sat)
                pot_total = agama.Potential(pot_ext, pot_traj_perturbed)

            time_begin_perturbed, time_end_perturbed = time_sat_perturbed[0], time_sat_perturbed[-1]

            # release points on the perturbed orbit: same times and masses as the unperturbed stream
            if tdis > 0:
                time_release_perturbed, orbit_release_perturbed = release_orbit(
                    pot_ext, posvel_sat, time_begin_perturbed, time_end_perturbed,
                    time_dis, n_release, breaks=breaks)
            else:
                time_release_perturbed, orbit_release_perturbed = time_sat_perturbed, orbit_sat_perturbed

            rng=np.random.default_rng(seed) # re-seed rng for IC generation -- will get same streamDF but center of rtids will change
            ic_stream_perturbed = create_ic_method(rng, pot_ext,
                                                   orbit_release_perturbed,
                                                   mass_release, **kwargs)
            time_seed_perturbed = np.repeat(time_release_perturbed, 2) #<-- when to release points.

        else:
            time_sat_perturbed, orbit_sat_perturbed = time_sat, orbit_sat
            time_begin_perturbed, time_end_perturbed = time_begin, time_end      
            ic_stream_perturbed = ic_stream
            time_seed_perturbed = time_seed
            pot_total = agama.Potential(pot_tot, pot_perturber) #<-- ie using the unperturbed orbit of the progenitor potential. 

        xv_snaps_perturbed = evolve_snapshots(pot_total, ic_stream_perturbed, time_seed_perturbed, 
                                              snap_times, time_end_perturbed,verbose=verbose,
                                              breaks=breaks)
        xv_stream_perturbed = xv_snaps_perturbed[-1]
    
    else: # <-- case of no perturbation
        xv_snaps_perturbed  = None
        xv_stream_perturbed = None
        time_sat_perturbed  = None
        orbit_sat_perturbed = None
        # perturber-only outputs: None so the return tuples below keep their usual length
        time_perturber      = None
        orbit_perturber     = None
        xv_stream_today     = None
        xv_perturber_flyby  = None


    if return_flyby_location==True:
        return (time_sat, time_sat_perturbed, 
                orbit_sat, orbit_sat_perturbed, 
                xv_stream, xv_stream_perturbed,
                time_perturber, orbit_perturber, 
                ic_stream, xv_stream_today, xv_perturber_flyby,
                xv_snaps, xv_snaps_perturbed)
    else:
        return (time_sat, time_sat_perturbed, 
                orbit_sat, orbit_sat_perturbed, 
                xv_stream, xv_stream_perturbed,
                time_perturber, orbit_perturber, 
                ic_stream, xv_stream_today,
                xv_snaps, xv_snaps_perturbed)
###


def clean_stream(orbit, num_particles, seed=0, verbose=False):
    """
    unperturbed particle-spray stream for one of the orbits in o.retrieve_sim_info
    ('gd1','c19','aau','jet','m3','pa5','circ'), same setup as gen_unperturbed_stream
    (MWPotential2014, 4 pc Plummer progenitor, chen25 ICs) but with no perturber at all.

    returns xv_stream: (num_particles, 6) present-day galactocentric positions and
    velocities (kpc, km/s).
    """
    pot = agama.Potential(repo_path+"/data/MWPotential2014.ini")

    apo, age, init_displacement, mprog = o.retrieve_sim_info(orbit)
    prog_w0 = np.array(init_displacement)
    prog_pot = agama.Potential(type='Plummer', mass=mprog, scaleRadius=4e-3) # kpc
    time_total = age / timeUnitMyr

    out = create_stream(
        create_ic_method=create_ic_chen25, seed=seed,
        time_total=time_total, num_particles=num_particles,
        pot_host=pot, posvel_sat=prog_w0, mass_sat=mprog,
        pot_sat=prog_pot, add_perturber=False,
        gen_unperturbed=True, return_flyby_location=False, verbose=verbose
    )
    xv_stream = out[4]
    return xv_stream



def get_streamcoords(orbit_sat, xv_stream, prog_w=None):
    """
    express xv_stream (Mx6, kpc & km/s) in jake's stream frame.

    prog_w: the present-day progenitor six-vector that DEFINES the frame.
            defaults to orbit_sat[-1]. pass it explicitly whenever two
            different particle sets have to land in the SAME frame -- the
            basis vectors come entirely from this one point, so a different
            prog_w means a different phi1/phi2 and the two sets are no
            longer comparable (let alone subtractable).
    """
    if prog_w is None:
        prog_w = orbit_sat[-1]
    prog_w = np.asarray(prog_w, dtype=float).reshape(6)   # copy: never alias orbit_sat
    xv_stream = np.atleast_2d(np.asarray(xv_stream, dtype=float))

    # in kpc, km/s; needs to be in kpc, kpc/Myr
    prog_w_jake_units   = np.hstack([prog_w[:3], prog_w[3:] * KMS_TO_KPCMYR])
    stream_w_jake_units = np.hstack([xv_stream[:, :3], xv_stream[:, 3:] * KMS_TO_KPCMYR])

    stream_obj = StreamFrame(sim_coords=stream_w_jake_units,
                             prog_sim_coord=prog_w_jake_units
                             )
    coords = stream_obj.GetStreamFrame()
    return stream_obj, coords

# %%

def integrate_orbit_agama(init_displacement, potential=None, time_Gyr=None, cadence_Myr=1.0,
                          trajsize=None):
    """
    potential: anything load_potential takes (None = MWPotential2014.ini, a .ini
        filename/path, or an agama.Potential).
    time_Gyr: signed interval in Gyr (required). cadence_Myr: spacing of the stored output.
    trajsize is derived from the requested cadence rather than guessed -- the old
    int(time_Gyr*10) actually gave ~100 Myr spacing, not the 1 Myr the caller assumed.
    pass trajsize to set the number of output points directly (cadence_Myr is then ignored).
    returns (times in agama time units, Nx6) -- times so nobody has to infer the
    cadence from an index again.
    """
    if time_Gyr is None:
        raise ValueError('time_Gyr is required')
    if trajsize is None:
        trajsize = max(2, int(round(abs(time_Gyr) * 1e3 / cadence_Myr)) + 1)
    return agama.orbit(potential=load_potential(potential), ic=init_displacement,
        time=time_Gyr/timeUnitGyr, trajsize=trajsize)

def galcen_to_frame(xv, frame):
    """
    galactocentric phase space (N x 6, kpc & km/s) -> a dict of coordinates in an astropy
    stream frame with phi1/phi2 components (a gala GreatCircleICRSFrame such as the one
    inputs.fit_streamframe fits, or a published one like gc.GD1Koposov10()).

    uses astropy's 'v4.0' galactocentric parameters, pinned as in viamock.xv_to_table,
    and goes through ICRS, so the result matches what inputs.inputs_galactocentric does
    to the stars (viamock.xv_to_table -> inputs.ICRS_from_table -> frame).
    NOT reflex corrected: 'v_gsr' is the heliocentric radial velocity (named to match
    inputs_galactocentric's column) and the pms include the solar reflex.

    returns {'phi1', 'phi2' (deg), 'pm_phi1', 'pm_phi2' (mas/yr; pm_phi1 is
    pm_phi1_cosphi2), 'v_gsr' (km/s), 'distance' (kpc)}, plain float arrays.
    """
    xv = np.atleast_2d(np.asarray(xv, dtype=float))
    with galactocentric_frame_defaults.set('v4.0'):
        coo = SkyCoord(x=xv[:, 0]*u.kpc, y=xv[:, 1]*u.kpc, z=xv[:, 2]*u.kpc,
                       v_x=xv[:, 3]*u.km/u.s, v_y=xv[:, 4]*u.km/u.s, v_z=xv[:, 5]*u.km/u.s,
                       frame=Galactocentric())
    sf = coo.icrs.transform_to(frame)
    return {'phi1':     sf.phi1.to(u.degree).value,
            'phi2':     sf.phi2.to(u.degree).value,
            'pm_phi1':  sf.pm_phi1_cosphi2.to(u.mas/u.yr).value,
            'pm_phi2':  sf.pm_phi2.to(u.mas/u.yr).value,
            'v_gsr':    sf.radial_velocity.to(u.km/u.s).value,
            'distance': sf.distance.to(u.kpc).value}

def progenitor_track_coords(prog_w, pot_host=None, frame=None, phi1_max_deg=180., n_half=2000, pad=2.5):
    """
    the progenitor's own orbit, expressed in a stream frame, ready to be used as the
    interpolant that straightening subtracts off.

    prog_w: present-day progenitor six-vector (kpc, km/s). MUST be the same
            point used to build the stream's coords, or the frames disagree.
    pot_host: the potential to integrate in, anything load_potential takes
            (None = MWPotential2014.ini, a .ini filename/path, or an agama.Potential).
            should be the potential the stream was made in.
    frame:  the frame to put the orbit in.
            None: jake's galactocentric StreamFrame defined by prog_w (get_streamcoords).
                keys phi1, phi2, pm_phi1, pm_phi2, r, vr. the present day is at phi1 = 0.
            an astropy stream frame (e.g. the GreatCircleICRSFrame from
                inputs.fit_streamframe): seen from Earth, via galcen_to_frame. keys phi1,
                phi2, pm_phi1, pm_phi2, v_gsr, distance, NOT reflex corrected. phi1 is
                wherever the present day lands in that frame, on the (-180, 180] branch.

    integrates symmetrically backwards and forwards from prog_w (integrate_orbit_agama),
    so the present-day epoch is the middle point.

    returns (phi1, track): phi1 unwrapped, strictly increasing, in degrees; track is the
    coords dict on the same points (track['phi1'] is phi1). trimmed to the single monotonic
    phi1 branch through the present day (f.chop_orbit_track) and to
    |phi1 - phi1_today| <= phi1_max_deg. in jake's frame the orbit is monotonic in phi1,
    so that's just the phi1_max_deg cut. seen from Earth the orbit can turn around in
    phi1 (parallax), and the chop keeps the branch the progenitor is on.
    """
    pot_host = load_potential(pot_host)
    prog_w = np.asarray(prog_w, dtype=float).reshape(6)
    x, v = prog_w[:3], prog_w[3:]

    # how long to integrate to sweep phi1_max_deg of azimuth: Omega = |L|/r^2 at
    # the present position, padded because Omega varies a lot over an eccentric orbit.
    # (integrating a fixed number of Gyr is what let the old version wrap ~3.5 times.)
    Omega = np.linalg.norm(np.cross(x, v)) / np.dot(x, x)
    t_half = pad * np.deg2rad(phi1_max_deg) / Omega          # agama time units

    _, back = integrate_orbit_agama(prog_w, pot_host, time_Gyr=-t_half*timeUnitGyr, trajsize=n_half+1)
    _, fwd  = integrate_orbit_agama(prog_w, pot_host, time_Gyr=+t_half*timeUnitGyr, trajsize=n_half+1)
    track_xv = np.vstack([back[::-1], fwd[1:]])   # increasing in time, present day in the middle
    i0 = n_half                                   # index of the present-day point

    if frame is None:
        _, track = get_streamcoords(None, track_xv, prog_w=prog_w)
    else:
        track = galcen_to_frame(track_xv, frame)
    track = {k: np.asarray(val, dtype=float) for k, val in track.items()}

    # phi1 is wrapped to a 360 degree range, so it jumps by ~360 every revolution.
    # unwrap along the (time-ordered) track. jake's frame: put the present day at
    # exactly 0. any other frame: keep the present day on the (-180, 180] branch the
    # stars' phi1 is on.
    phi1 = np.rad2deg(np.unwrap(np.deg2rad(track['phi1'])))
    if frame is None:
        phi1 = phi1 - phi1[i0]
    else:
        phi1 = phi1 - 360.*np.ceil((phi1[i0] - 180.)/360.)
    phi1_today = phi1[i0]

    # the monotonic branch through the present day (index n//2 = i0), increasing in phi1
    idx = f.chop_orbit_track(phi1)
    phi1, track = phi1[idx], {k: val[idx] for k, val in track.items()}

    keep = np.abs(phi1 - phi1_today) <= phi1_max_deg
    phi1, track = phi1[keep], {k: val[keep] for k, val in track.items()}
    track['phi1'] = phi1
    if not np.all(np.diff(phi1) > 0):
        raise ValueError('progenitor phi1 is not monotonic after unwrapping; '
                         'np.interp would silently return garbage')
    return phi1, track

def straighten_stream_orbit_interp_agama(
        coords,
        prog_w,          #<-- present-day progenitor six-vector, the SAME one behind `coords`
        pot_host=None,   #<-- anything load_potential takes; None = MWPotential2014.ini
        return_orbit_chunk=False,
        **track_kwargs   #<-- to progenitor_track_coords, e.g. frame=
):
    """
    straighten a stream by subtracting the progenitor's orbit at matched phi1.
    phi1 passes through untouched; every other key in coords that the orbit track
    also has gets the orbit's value at that phi1 subtracted off. keys the track
    doesn't have are copied over unchanged.

    coords must be in the same frame as the track: jake's frame (get_streamcoords with
    this prog_w) by default, or pass frame= (see progenitor_track_coords).
    """
    phi1_track, orbit_coords = progenitor_track_coords(prog_w, pot_host, **track_kwargs)
    phi1 = np.asarray(coords['phi1'], dtype=float)

    # np.interp clamps to the end values outside xp instead of raising, which would
    # quietly leave the tails un-straightened rather than tell you the track is short
    n_out = np.count_nonzero((phi1 < phi1_track.min()) | (phi1 > phi1_track.max()))
    if n_out:
        print('WARNING: %d/%d particles fall outside the orbit track phi1 range [%.1f, %.1f]; '
              'np.interp is clamping them' % (n_out, phi1.size, phi1_track.min(), phi1_track.max()))

    straightened_coords = {'phi1': phi1}
    for key in coords.keys():          # .keys() so an astropy Table works too
        if key == 'phi1':
            continue
        if key not in orbit_coords:
            straightened_coords[key] = coords[key]
            continue
        straightened_coords[key] = (np.asarray(coords[key], dtype=float)
                                    - np.interp(phi1, phi1_track, orbit_coords[key]))

    if return_orbit_chunk:
        return straightened_coords, phi1_track, orbit_coords
    else:
        return straightened_coords

#------------------------------------------------------------#
#     composite/wrapper functions for stream generation      #
#------------------------------------------------------------#
def gen_unperturbed_stream(orbit, fname, Nstars=10000,
                           outdir='/n/netscratch/conroy_lab/Lab/amphillips/tsetse_analysis/perturbed_ps_streams/',
                           mass_loss=True, tdis_gyr=0.1):
    """
    the null stream for an orbit (massless subhalo), saved as its ssc dict.
    mass_loss, tdis_gyr (GYR) as in gen_perturbed_stream; they have to match the perturbed
    streams it's compared to (with the same Nstars, it's then the same stream as the
    unperturbed half of every gen_perturbed_stream call: same seed, same settings).
    """
    pot = agama.Potential(repo_path+"/data/MWPotential2014.ini")

    ### retrieve info about this simulation.
    apo, age, init_displacement, mprog = o.retrieve_sim_info(orbit)
    prog_w0 = np.array(init_displacement)
    prog_mass = mprog # Msun 
    prog_pot = agama.Potential(
        type='Plummer', 
        mass=prog_mass, 
        scaleRadius=4e-3 # kpc. so this is 4 pc. note that this doesn't control internal progenitor dynamics/release conditions, only the potential felt by stream particles
    )
    time_total = age / timeUnitMyr

    # --- STEP 0: generate a densely-sampled unperturbed stream --- #
    many_particles = Nstars
    ### a little dumb but let's just have a massless subhalo
    pdict_null = {
        "mass":0.,
        "scaleRadius":0.1,
        "truncationRadius":0.1,
        "b":100000.,
        "v": 100000.,
        "t_impact":0.,
        'phi1_today':10.,
        'b_trl':np.array([1,0,0]),
        'v_trl':np.array([1,0,0]),
        'ramp_frac':0.5 # idk
    }
    (time_sat_null, time_sat_perturbed_null, 
                orbit_sat_null, orbit_sat_perturbed_null, 
                xv_stream_null, xv_stream_perturbed_null,
                time_perturber_null, orbit_perturber_null, 
                ic_stream_null, pert_location_today_null, pert_location_flyby_null,
                xv_snaps_null, xv_snaps_perturbed_null) = create_stream(
        create_ic_method=create_ic_chen25, seed=0,
        time_total=time_total, num_particles=many_particles, 
        pot_host=pot, posvel_sat=prog_w0, mass_sat=prog_mass,
        update_prog_orbit=True,
        pot_sat=prog_pot, add_perturber=True, perturber_dict=pdict_null,
        gen_unperturbed=True, xv_stream_input=None,
        return_flyby_location=True, verbose=False,
        tdis=tdis_gyr/timeUnitGyr, mass_loss=mass_loss
    )



    ssc_null = f.straightened_obscoords_orbit_interp(orbit, xv_stream_null, prog_tab)

    ### straightened_obscoords_orbit_interp does not nominally give tangential velocities or distances, but i might want them
    coords_null, _ = f.streamframe_coords_observed(orbit, xv_stream_null, prog_tab)
    d_kpc = coords_null.distance.to(u.kpc).value
    pmphi1 = ssc_null['pm_phi1'] * (u.mas/u.yr).to(u.rad/u.yr)
    pmphi2 = ssc_null['pm_phi2'] * (u.mas/u.yr).to(u.rad/u.yr)

    vphi1 = pmphi1 * d_kpc * (u.kpc/u.yr).to(u.km/u.s)
    vphi2 = pmphi2 * d_kpc * (u.kpc/u.yr).to(u.km/u.s)

    ssc_null['v_phi1'] = vphi1
    ssc_null['v_phi2'] = vphi2
    ssc_null['distance'] = d_kpc

    with open(outdir+fname, 'wb') as handle:
        pickle.dump(ssc_null, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return ssc_null

def gen_perturbed_stream(orbit, perturber_dict, fname, Nstars=4000,
                         save=True,
                           outdir='/n/netscratch/conroy_lab/Lab/amphillips/tsetse_analysis/perturbed_ps_streams/',
                           mass_loss=True, tdis_gyr=0.1
                           ):
    """
    perturbed stream for one impact, saved to outdir+fname as a dict:
        ssc:        straightened observed-frame coordinates of the perturbed stream
        ssc_pert:   the impact site today, in the observed frame
        stars_w:    (Nstars, 6) present-day galactocentric perturbed stream (kpc, km/s),
                    ready for inputs.inputs_galactocentric
        prog_w:     (6,) present-day galactocentric progenitor (kpc, km/s)
        mass_loss, tdis_gyr: the settings it was made with

    mass_loss:  progenitor mass falls linearly from mprog to zero (create_stream's mass_loss)
    tdis_gyr:   how long ago the progenitor dissolved, in GYR (converted to agama units here)
    gen_unperturbed_stream has to be run with the same mass_loss/tdis_gyr for the null to match.
    """
    ### i'll generate the perturber dict outside of this function.
    pot = agama.Potential(repo_path+"/data/MWPotential2014.ini")

    ### retrieve info about this simulation.
    apo, age, init_displacement, mprog = o.retrieve_sim_info(orbit)
    prog_w0 = np.array(init_displacement)
    prog_mass = mprog # Msun 
    prog_pot = agama.Potential(
        type='Plummer', 
        mass=prog_mass, 
        scaleRadius=4e-3 # kpc. so this is 4 pc. note that this doesn't control internal progenitor dynamics/release conditions, only the potential felt by stream particles
    )
    time_total = age / timeUnitMyr

    do_unpert=True
    xv_stream_input=None
    num_particles=Nstars
    (time_sat, time_sat_perturbed, 
                orbit_sat, orbit_sat_perturbed, 
                xv_stream, xv_stream_perturbed,
                time_perturber, orbit_perturber, 
                ic_stream, pert_location_today, pert_location_flyby,
                xv_snaps, xv_snaps_perturbed) = create_stream(
        create_ic_method=create_ic_chen25, seed=0,
        time_total=time_total, num_particles=num_particles, 
        pot_host=pot, posvel_sat=prog_w0, mass_sat=prog_mass,
        update_prog_orbit=True,
        pot_sat=prog_pot, add_perturber=True, perturber_dict=perturber_dict,
        gen_unperturbed=do_unpert, xv_stream_input=xv_stream_input,
        return_flyby_location=True, verbose=False,
        tdis=tdis_gyr/timeUnitGyr, mass_loss=mass_loss
    )

    ssc = f.straightened_obscoords_orbit_interp(orbit, xv_stream_perturbed, prog_tab)
    mega_pert_location_today = np.array([pert_location_today])
    ssc_pert, sf = f.streamframe_coords_observed(orbit, mega_pert_location_today, prog_tab) 

    ### straightened_obscoords_orbit_interp does not nominally give tangential velocities or distances,
    #    but i might want them
    coords, _ = f.streamframe_coords_observed(orbit, xv_stream_perturbed, prog_tab) #<-- the PERTURBED stream (was xv_stream)
    d_kpc = coords.distance.to(u.kpc).value
    pmphi1 = ssc['pm_phi1'] * (u.mas/u.yr).to(u.rad/u.yr)
    pmphi2 = ssc['pm_phi2'] * (u.mas/u.yr).to(u.rad/u.yr)

    vphi1 = pmphi1 * d_kpc * (u.kpc/u.yr).to(u.km/u.s)
    vphi2 = pmphi2 * d_kpc * (u.kpc/u.yr).to(u.km/u.s)

    ssc['v_phi1'] = vphi1
    ssc['v_phi2'] = vphi2
    ssc['distance'] = d_kpc


    pert_dict_out = {"ssc":ssc, "ssc_pert":ssc_pert,
                     "stars_w":xv_stream_perturbed,          # galactocentric, today -> inputs.inputs_galactocentric
                     "prog_w":prog_w0.copy(), # the progenitor today; perturbed or not, both orbits end here (posvel_sat)
                     "mass_loss":mass_loss, "tdis_gyr":tdis_gyr}

    if save==True:
        with open(outdir+fname, 'wb') as handle:
            pickle.dump(pert_dict_out, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return pert_dict_out



#----------------------------------------------------#
#   functions for measuring pert locations and       #
#   amplitudes from generated streams/data           #
#----------------------------------------------------#
def weighted_median(y, w):
    """
    median of y with weights w
    """
    idx = np.argsort(y)
    y, w = y[idx], w[idx]
    cw = np.cumsum(w)
    # midpoint convention: each sample "sits" at the centre of its weight interval
    return np.interp(0.5*cw[-1], cw - 0.5*w, y)

def running_median(phi1, y, grid, bw, kernel='gauss', min_neff=50):
    """
    running median of y along phi1, evaluated at each point of `grid`.

    kernel='tophat': particles with |phi1 - g| < bw/2 get weight 1 (overlapping bins).
    kernel='gauss':  Gaussian weights with FWHM = bw (sigma = bw/2.355), cut at 3 sigma.
                     comparable resolution to a top-hat of width bw, but no hard edges.

    returns (med, err, neff), each len(grid). NaN where neff < min_neff.
    """
    phi1 = np.asarray(phi1, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(phi1) & np.isfinite(y)
    order = np.argsort(phi1[ok])
    phi1, y = phi1[ok][order], y[ok][order]

    if kernel == 'gauss':
        sig = bw / 2.355
        half = 3*sig
    else:
        half = bw / 2

    med = np.full(len(grid), np.nan)
    err = np.full(len(grid), np.nan)
    neff = np.zeros(len(grid))

    for k, g in enumerate(grid):
        # phi1 is sorted, so searchsorted pulls out the window without scanning everything
        lo, hi = np.searchsorted(phi1, [g - half, g + half])
        if hi - lo == 0:
            continue
        yy = y[lo:hi]
        if kernel == 'gauss':
            w = np.exp(-0.5*((phi1[lo:hi] - g)/sig)**2)
        else:
            w = np.ones(hi - lo)

        # "effective" number of objects in this bin
        neff[k] = w.sum()**2 / (w**2).sum()   # = N for top-hat
        if neff[k] < min_neff:
            continue

        m = weighted_median(yy, w)
        mad = weighted_median(np.abs(yy - m), w) # median absolute deviation around the weighted median
        med[k] = m

        err[k] = 1.253 * 1.4826 * mad / np.sqrt(neff[k]) 
        # what are those weird factors for the error? 
        # for normally destributed data, the median of (x-mean) is 
        #   1.4826 * mad (from above), and it estimates the standard deviation of the distribution
        # for N normally distributed points, the sample median has variance (pi/2_^2)*sigma^2 / N, 
        #   so the error on the median is
        #   sigma_median = sqrt(pi/2)*sigma/sqrt(N) 
        #                = 1.253 * sigma/sqrt(N)  

    return med, err, neff

def measure_amps_locs(ssc, ssc_null, 
                      dims=['phi2','v_gsr','pm_phi1','pm_phi2'], #<--  might want to use 'v_phi1/phi2' at some point. 
                      bw=10, step=1, win_fraction=1.5, 
                      **running_med_kwargs):
    """
    running_med_kwargs might be min_neff, kernel='tophat' (nominally 'gauss')
    """

    all_res, all_err = {}, {}
    grid = np.arange(-100, 100+step, step) #<-- is this sufficient for all streams?

    for ii, d in enumerate(dims):
        m0, e0, n0 = running_median(ssc_null['phi1'], ssc_null[d], grid, bw)
        m, e, n = running_median(ssc['phi1'], ssc[d], grid, bw)

        res = m-m0
        err = np.hypot(e, e0) #<--- i think this is just a quadrature sum of the errors. 
        all_res[d] = res
        all_err[d] = err

    ### get perturbation location using where the residuals are maximized...
    z2 = np.zeros(len(grid))
    for d in dims:
        z2 += np.nan_to_num((all_res[d]/all_err[d])**2)   # NaN windows contribute nothing
    i_loc = np.argmax(z2)
    phi1_loc = grid[i_loc]


    ### and measure the perturbation amplitude 
    ### a boolean window around the fit phi1_loc that is win_fraction x bin width, default 1.5
    win = np.abs(grid - phi1_loc) < win_fraction*bw # this is supposed to be ~wiggle half-width; amp will be kind of sensitive to the window, so might want to plot some answers to see if i'm getting it right.)      
    amp, amp_err = {}, {}
    for d in dims:
        r = np.where(win, all_res[d], np.nan)
        imax, imin = np.nanargmax(r), np.nanargmin(r)
        amp[d] = r[imax] - r[imin]
        amp_err[d] = np.hypot(all_err[d][imax], all_err[d][imin]) 


    return phi1_loc, amp, amp_err