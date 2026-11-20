# %%
import agama
import os
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
# machine-specific paths live in scripts/paths.py (picks cannon vs. my machine on its own)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd(), 'scripts'))
from paths import repo_path, MPLSTYLE
plt.style.use(MPLSTYLE)
import functions as f
import tsetse as tse
from streamframe import StreamFrame

import inputs
import filter 

from tqdm import tqdm
from matplotlib.gridspec import GridSpec
from matplotlib.gridspec import GridSpecFromSubplotSpec
import astropy.units as u
import astropy.constants as const
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

# %%
# #--------------------------------------------------------------#
# #            example with contrived perturbations              #
# #--------------------------------------------------------------#
# orbit='gd1'

# #  example of good perturbation that i generate
# perturber_dict = {
#     "mass":1e7,
#     # "mass":1e7,
#     "scaleRadius":0.5,
#     "truncationRadius":0.5,
#     "b":0,
#     "v": 100,
#     "t_impact":0.1 / tse.timeUnitGyr,
#     'phi1_today':20,
#     # "phi1_today":6,
#     'b_trl':np.array([0,0,1]),
#     'v_trl':np.array([1/np.sqrt(2),1/np.sqrt(2),0]),
#     'ramp_frac':0.1 # idk
# }




# # #   example of no perturbation
# # perturber_dict = {
# #     "mass":5e1,
# #     "scaleRadius":0.5,
# #     "truncationRadius":0.5,
# #     "b":1000000000000,
# #     "v": 10000000000,
# #     "t_impact":0.05 / tse.timeUnitGyr,
# #     'phi1_today':20,
# #     'b_trl':np.array([0,0,1]),
# #     'v_trl':np.array([1/np.sqrt(2),1/np.sqrt(2),0]),
# #     'ramp_frac':0.1 # idk
# # }



# print("generating perturbed stream. ")
# # one bad row (e.g. a NaN subrt, which agama refuses as a truncation radius) used to kill
# # the task and lose the rest of the chunk. now it's logged and skipped.
# pert_dict_out = tse.gen_perturbed_stream(orbit,
#                                             perturber_dict,
#                                             fname=None,
#                                             Nstars=10000, #<-- since I'm submitting as an array, idrc about making this a little slow...
#                                             save=False, outdir=None,
#                                             # outdir=OUTDIR+'%s/'%orbit,
#                                             mass_loss=True, tdis_gyr=0.1
#                                             )

# stars_w, prog_w = pert_dict_out['stars_w'], pert_dict_out['prog_w'] #<-- eventually this will be something that I just get from Lucas. 
# pot_total = None
# prog_mass = 1e4
# %%
#---------------------------------------------------------------------#
#        examples with Lucas' streams in dreams potentials            #
#---------------------------------------------------------------------#
host_n = 1012
host_dir = f'./lucas_streams/run_%i/'%host_n
fsave = host_dir+'stream_catalog_run_%i.pkl'%host_n
with open(fsave, 'rb') as f:
    catalog=pickle.load(f)

pot_bar = tse.load_potential(host_dir+'pot_bar_z0.ini')
pot_dm = tse.load_potential(host_dir+'pot_dm_z0.ini')
pot_total = pot_bar+pot_dm
R_peris = catalog['Rperi']

prog_masses = catalog['prog_mass']
prog_ws = catalog['prog_coords_z0']
stars_ws = catalog['st_coords_z0']


index = 105## 436 #1000 wraps the galaxy!
stars_w = stars_ws[index]
prog_w = prog_ws[index]
prog_mass = prog_masses[index]
Rperi = R_peris[index]

fig,ax = plt.subplots()
ax.scatter(stars_w[:,0], stars_w[:,1], s=1, c='k')

# %%

# print("creating stream for progenitor mass %i; rperi=%.1f"%(prog_mass, Rperi))
ssc_table, streamframe, track = inputs.inputs_galactocentric(stars_w, prog_w, 
                                                             potential=pot_total, 
                                                             distance=10, #kpc
                                                             stream_frame='galactocentric',
                                                             subtract_orbit=True,
                                                             Glim=24, m_pop=prog_mass) # TODO: LET THE SSP TOTAL MASS CHANGE!

# %%
additional_clips = np.ones(len(ssc_table), dtype=bool)# None #(np.abs(ssc_table['phi2'])<10) & (np.abs(ssc_table['v_gsr'])<50)

fig, ax = plt.subplots(figsize=(8,3))
ax.scatter(ssc_table['phi1'][additional_clips], 
           ssc_table['pm_phi1'][additional_clips], 
           s=1, 
           c='k')

# r = np.mean(ssc_table['r']) 
# scale = 3
# phi1_scale = scale/r * u.radian.to(u.degree)
# ax.axvspan(perturber_dict['phi1_today']-phi1_scale/2, perturber_dict['phi1_today']+phi1_scale/2,
#            alpha=0.2)


# %%
#---------------------------------------------------#
#   creating the grid that the filter will search   #
#   over; depends on the span of the stream in      #
#   phi1, how many points i want                    #
#---------------------------------------------------#
# create the grid to search over
lo, hi = np.quantile(ssc_table['phi1'], [0.01,0.99])
stream_length = hi-lo
if stream_length>200:
    lo, hi=-100,100
    stream_length=hi-lo

d_phi1 = stream_length/100
loc_grid = np.arange(lo, hi, d_phi1) # degree increments. 

n_points = 100
wmin = stream_length / 2000

width_max_physical = 1 # kpc
dist_avg = np.mean(ssc_table['r']) 
maxwidth_angle = width_max_physical / dist_avg * u.radian.to(u.degree)
wmax = maxwidth_angle# stream_length / 20
print(wmax)
w_grid = np.geomspace(wmin, wmax, n_points) # this basically makes sense for GD1; trying to keep it proportional for the stubbier streams. 

# perturbation locations and widths:
grid_dict = {
"l": loc_grid,
"w": w_grid,
}

# add intrinsic stream dispersions in each dimension; hard-coded for now but you could change the grid search
n_sig = n_points
for key in filter.GRID_KEYS:
    grid_dict['sig_'+key] = np.linspace(0.2, 3, n_sig) * filter.streamwidths_test[key]

unc=True #<-- whether to use errors and place precision constraints. 
filter_window_factor = 2. # filter window half-width = filter_window_factor*max(w grid)


# %%
# RUN THE grid search
res = filter.gridsearch(grid_dict, ssc_table[additional_clips],
                        uncertainties=unc, n_proc=1,
                        filter_window_factor=filter_window_factor,
                        min_stars=15)
# the result 'res' is a dictionary. A, m, and b have a highest likelihood value and uncertainty for each point on the grid in (w, l, intrinsic width) and in each dimension. 
#   so e.g., res['A_hat'] and res['sigma_A'] have shape [pert location grid, w grid, intrinsic width grid, 4 dimensions]
#   and are the highest likelihood amp and its uncertainty in each dimension at each grid point
# then res['lnL_max'] gives you the maximum likelihood reachable across the grid in location, perturbtation width, stream dispersion per dimension (taking the maximum likelihood A, m, b per grid point)
#   lnL_null is the same but fixing the amplitude to zero, i.e., in each window what's the maximum likelihood you can have fitting just a straight line through the data with no wiggle
# %%
### compute median/quantile values of the marginal likelihoods of amplitudes in each dimension:
margs = [filter.marginal_A(res, grid_dict, k) for k in tqdm(range(len(filter.GRID_KEYS)))] 
# to get margs['z'] (or the detection statistic) we've inverted a standard gaussian CDF at P_+
# %%
# detections: local maxima of |z(l)| that reach z_thresh -- 
z_thresh = 10
peaks = filter.find_z_peaks(margs, z_thresh=z_thresh) # finds peaks in the amplitude (in units of its uncertainty) along the stream
# ghost tests from the shape of p(w | l). rerun=True searches the peaks with no parent
# again on a w grid extended past max(w), to see if p(w | l) follows the edge.
ghosts = filter.ghost_flags(margs, grid_dict, res, peaks,
                            rerun=True, ssc=ssc_table, uncertainties=unc)
show_ghosts = True # True: ghost-flagged peaks in light coral. False: hide them.

# make diagnostic figure.
fig, axs, _ = filter.plot_detections(res, grid_dict, ssc_table[additional_clips], uncertainties=unc,
                                z_thresh=z_thresh, phi1_true=None,#phi1_true,
                                # subtract_poly=False, poly_deg=5,
                                # prog_subtracted=prog_subtracted,
                                keys=['phi2', 'v_gsr', 'pm_phi1'],
                                # keep_peaks={'v_gsr':[16]},
                                margs=margs, 
                                prof=None, w_meds=None, peaks=peaks,
                                ghosts=ghosts, show_ghosts=show_ghosts)

# axs[-1,-1].set_ylim(-50,50)
for row in axs:
    for ax in row:
        # ax.set_xlim(lo, hi)
        ax.set_xlim(-10,8)
# %%

# %%
# shapes of p(w | l) at each z peak, one panel per dimension, with A and sig marginalised
# out (the same (w, sig) weights as p(A | l), summed over sig). each curve is the
# probability per w grid point. the dotted line marks max(w): weight piled up there is
# the ghost signature. each label ends with the ghost_flags class (flagged peaks dashed).
# peaks, z and var_between/var_within come from margs above
# (flat w prior), so if w_prior here is 'log' the curves change but the labels don't.
w_prior = 'log'
w_ax = np.asarray(grid_dict['w'])
fig, axs = plt.subplots(len(filter.GRID_KEYS), 1, figsize=(7, 2.6*len(filter.GRID_KEYS)),
                        sharex=True, constrained_layout=True)
for k, key in enumerate(filter.GRID_KEYS):
    ax = axs[k]
    _, p_w = filter.marginal_quantiles(res, grid_dict, k, keep='w', q=(0.5,), w_prior=w_prior)
    pk = peaks[k]
    colors = plt.cm.viridis(np.linspace(0, 0.9, max(len(pk), 1)))
    for i, (c, il) in enumerate(zip(colors, pk)):
        ratio = margs[k]['var_between'][il] / margs[k]['var_within'][il]
        ax.plot(w_ax, p_w[il], color=c, lw=1.5,
                 ls='--' if ghosts[k]['flag'][i] else '-',
                label=r'$l = %.0f$, $z = %.1f$, btw/wtn $= %.2g$, %s'
                      % (grid_dict['l'][il], margs[k]['z'][il], ratio, ghosts[k]['class'][i]))
    ax.axvline(w_ax[-1], color='gray', ls=':', lw=1)
    ax.set_ylim(bottom=0)
    ax.set_ylabel(r'$p(w \mid l)$, $%s$' % filter.SYMBOLS[key])
    if len(pk):
        ax.legend(fontsize=7, loc='upper left', frameon=False)
    else:
        ax.text(0.5, 0.5, r'no peaks with $|z| \geq %g$' % z_thresh,
                transform=ax.transAxes, ha='center', va='center', color='gray')
axs[-1].set_xlabel(r'$w$ [deg]')

# %%