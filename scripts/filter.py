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
# repo_path = '/Users/anyaphillips/Desktop/harvard/research/THE_STREAM_IMPACT_EXPLORER'
repo_path = '/n/home02/amphillips/THE_STREAM_IMPACT_EXPLORER'
plt.style.use(repo_path+'/scripts/vedant.mplstyle')
sys.path.append(repo_path+'/scripts')
import functions as f
import tsetse as tse
from streamframe import StreamFrame
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
sys.path.append("/n/home02/amphillips/software/viamock")
import viamock
sys.path.append("/n/home02/amphillips/.conda/envs/petar_env/lib/python3.10/site-packages") #<-- possibly a rly crazy thing to do lol. 
import artpop

sys.path.append(repo_path)
import inputs
# %%
def generic_signal(x, A, w, loc=0, m=0, b=0):
    """

    returns a*(x-loc) / ((x-loc)^2 + w^2) + m*(x-loc) + b,

    i.e. the wiggle plus a linear baseline. the line is centred on loc, so b is the
    baseline at the centre of the wiggle and m its slope there. (centring on loc
    rather than x=0 doesn't change the fit for A, but keeps b meaningful and m, b
    uncorrelated for a window far from phi1 = 0.) loc stays the 4th positional
    argument so generic_signal(x, A, w, loc) calls still work.

    where w = b v_enc / v_perp (larger->wider perturbation. 
                     but since is divided out larger
                     will also shrink the signal)

    and  a = -2GMsub/v_enc (larger -> higher amplitude)
    
    a is specified from A (the actual amplitude of the signal) and w by
    a = 2*w*A 


    A is the actual amplitude of the perturbation, +/- w are the locations of the extrema. 
        
    """
    a = 2 * w * A
    wiggle_component = (a * (x-loc) / ((x-loc)**2 + w**2))
    linear_component = (m*(x-loc) + b)
    return wiggle_component + linear_component
# %%
if __name__=="__main__":

    n=15
    amps = np.linspace(0.1, 2, n)
    widths = np.linspace(0.1, 3, n)
    ms = np.linspace(-0.5, 0.5, n)

    x = np.linspace(-6,6,500)

    fig, axs = plt.subplots(1,3,figsize=[24,8], sharey=True)
    plt.subplots_adjust(wspace=0, hspace=0)
    colors=cmr.torch(np.linspace(0, 0.9, n))
    for ii, a in enumerate(amps):
        axs[0].plot(x, generic_signal(x, a, w=1), color=colors[ii])
    for ii, w in enumerate(widths):
        axs[1].plot(x, generic_signal(x,A=2, w=w ), color=colors[ii])

    colors = plt.cm.coolwarm(np.linspace(0, 1, n))
    for ii, m in enumerate(ms):
        axs[2].plot(x, generic_signal(x, A=1, w=0.75, m=m), color=colors[ii])
        axs[2].plot(x, m*x, color=colors[ii], ls=':', lw=2)

    axs[0].set_title(r'Varying $A$')
    axs[1].set_title(r'Varying $w$')
    axs[2].set_title(r'Varying $m$')
    plt.savefig("example_filter.png")
# %%

def rms_residual(data, model, **model_args):
    """
    model is a Function with args model_args
    """
    r = np.sum((data - model(**model_args))**2) #/ len(data)
    return r

# accounting for the fact that the stream has a characteristic width sigma
def nll_gaussian(data, width, model, uncertainties=None, **model_args):
    """
    compute a gaussian negative log likelihood 
    given data, an intrinsic width, the model function and its **model_args
    and the data uncertainties. 

    width is one number that the stream has 
    characteristically along phi1.

    if the data have uncertainties then I would add them in
    quadrature with the characteristic width of the stream.

    delta_lnL compares the model to the same model with A = 0, i.e. the wiggle
    against the linear baseline (m, b) alone, at the same (m, b). (with m = b = 0
    that's the old comparison to a flat line at zero.)
    """

    obs = data
    expected = model(**model_args)
    null = model(**{**model_args, 'A': 0.0}) * np.ones(len(obs))

    sigma = width if uncertainties is None else np.sqrt(width**2 + uncertainties**2)

    # f1 = 1 / np.sqrt(2 * np.pi * sigma**2)
    # exponential_arg = - (obs - expected)**2 / (2 * sigma**2)
    # lnL_i = np.log(f1 * np.exp(exponential_arg))

    ### negative log likelihood analytically instead. 
    # lnL_i = -0.5 * (data - expected)**2 / sigma - 0.5 * np.log(2 * np.pi * sigma)
    lnL_i = -0.5 * (data - expected)**2 / sigma**2 - 0.5 * np.log(2 * np.pi * sigma**2)
    lnL_i0 = -0.5 * (data - null)**2 / sigma**2 - 0.5 * np.log(2 * np.pi * sigma**2)


    delta_lnL = np.sum(lnL_i - lnL_i0)
    neg_lnL = -np.sum(lnL_i)

    return delta_lnL, neg_lnL

# intrinsic stream widths used when none are given (nll_model_full), and for any
# dimension without a 'sig_'+key grid in gridsearch
DEFAULT_STREAMWIDTHS = {'phi2':0.1, "pm_phi1":0.01, "pm_phi2":0.001, "v_gsr":0.1}

def _window_data(ssc, key, l, window_width, uncertainties=False, precision_limits=None):
    """
    the stars the likelihood uses in dimension key for a window of half-width
    window_width around l: returns (x, y, err), each an array over those stars.
    without uncertainties err is all zeros (not None), so sig**2 + err**2 is always
    one value per star. with uncertainties, x is phi1_noised, y is ssc[key+'_noised'],
    and only stars passing obs_limit for key are kept. same selection as gridsearch.
    """
    pls = {} if precision_limits is None else precision_limits
    x = np.asarray(ssc['phi1_noised'] if uncertainties else ssc['phi1'], dtype=float)
    use = (x < (l + window_width)) & (x > (l - window_width))
    if not uncertainties:
        return x[use], np.asarray(ssc[key], dtype=float)[use], np.zeros(np.sum(use))
    use = use & obs_limit(ssc, key, precision_limit=pls.get(key))
    return (x[use], np.asarray(ssc[key+"_noised"], dtype=float)[use],
            np.asarray(ssc[key+"_err"], dtype=float)[use])

def nll_model_full(ssc, params,
                   streamwidths=None,
                   uncertainties=False,
                   window_width=None,
                   precision_limits=None,
                   min_stars=5,
                   baselines=None):
    """
    the brute-force likelihood of the full model, wiggle + line, in each dimension.
    params = (l, w, A_phi2, A_pm_phi1, A_pm_phi2, A_v_gsr).

    baselines: dict {key: (m, b)}, the slope and intercept of the linear baseline in
    each dimension (centred on l, see generic_signal). None, or a missing key, means
    (0, 0). the window median is NOT subtracted any more: b is the baseline, so with
    no baselines the data are compared to a wiggle about zero.

    returns (nll, delta_lnL) per dimension, where delta_lnL is the model against the
    same line with A = 0 (see nll_gaussian).

    window_width: half-width (in phi1, deg)
    of the window around phi1_loc that
    the likelihood is evaluated on. None = 2*w,
    so the window changes with the
    tested w; pass a number to hold it fixed
    (e.g. across a grid search).

    uncertainties: if False, use the noiseless columns ssc[key] and only the
    intrinsic stream widths. if True, use the noised columns ssc[key+"_noised"]
    (phi1 included), add ssc[key+"_err"] in quadrature with the stream width,
    and in each dimension keep only stars that pass obs_limit for that dimension.
    phi1 is treated as error-free in the likelihood.

    precision_limits: dict of precision cuts per dimension, handed to obs_limit.
    None (or a missing key) uses PRECISION_REQUIREMENTS_DICT. only used if
    uncertainties is True.

    min_stars: a dimension with fewer than this many stars in the window (after
    any precision cut) returns NaN for both nll and delta lnL. otherwise an empty
    window sums to nll = 0, which looks like a perfect fit.
    """
    sw = DEFAULT_STREAMWIDTHS if streamwidths is None else streamwidths
    bls = {} if baselines is None else baselines
    keys = ['phi2', 'pm_phi1','pm_phi2','v_gsr']

    phi1_loc, wpert, amp_phi2, amp_pmphi1, amp_pmphi2, amp_vgsr = params

    amps = {key:amp for key, amp in zip(keys, [amp_phi2, amp_pmphi1, amp_pmphi2, amp_vgsr])}

    window_size = 2*wpert if window_width is None else window_width

    nlls = []
    delta_lls = []
    for key in keys:
        # stars in the window (and, with uncertainties, precise enough in this dimension)
        xw, yw, errs = _window_data(ssc, key, phi1_loc, window_size,
                                    uncertainties=uncertainties, precision_limits=precision_limits)
        if len(xw) < min_stars:
            nlls.append(np.nan)
            delta_lls.append(np.nan)
            continue
        m, b = bls.get(key, (0.0, 0.0))
        delta_lnL, neg_lnL = nll_gaussian(yw,
                      width = sw[key],
                      model=generic_signal,
                      uncertainties=errs,
                      x = xw, A = amps[key], w = wpert, loc = phi1_loc, m = m, b = b)
        nlls.append(neg_lnL)
        delta_lls.append(delta_lnL)

    return np.array(nlls), np.array(delta_lls)


GRID_KEYS = ['phi2', 'pm_phi1', 'pm_phi2', 'v_gsr'] # same order as nll_model_full
streamwidths_test = {'phi2':0.1, "pm_phi1":0.1, "pm_phi2":0.02, "v_gsr":0.5}

def _sig_grid(grid_dict, key):
    """
    the intrinsic stream width grid for dimension key, grid_dict['sig_'+key].
    without one, the width is fixed at DEFAULT_STREAMWIDTHS[key] (a length-1 grid).
    """
    return np.atleast_1d(grid_dict.get('sig_'+key, DEFAULT_STREAMWIDTHS[key]))

# set per process (see _init_worker) so a Pool doesn't re-pickle ssc for every task.
# holds plain float arrays and the precision masks, computed once rather than on
# every likelihood evaluation:
#   'x'    phi1 (phi1_noised if uncertainties)
#   'data' {key: ssc[key]} (ssc[key+'_noised'] if uncertainties)
#   'err'  {key: ssc[key+'_err']} if uncertainties, else {key: None}
#   'good' {key: obs_limit mask} if uncertainties, else all True
_worker = {}

def _init_worker(ssc, uncertainties=False, precision_limits=None):
    global _worker
    pls = {} if precision_limits is None else precision_limits
    x = np.asarray(ssc['phi1_noised'] if uncertainties else ssc['phi1'], dtype=float)
    _worker = {'x': x, 'data': {}, 'err': {}, 'good': {}}
    for key in GRID_KEYS:
        if uncertainties:
            _worker['data'][key] = np.asarray(ssc[key+"_noised"], dtype=float)
            _worker['err'][key] = np.asarray(ssc[key+"_err"], dtype=float)
            _worker['good'][key] = obs_limit(ssc, key, precision_limit=pls.get(key))
        else:
            _worker['data'][key] = np.asarray(ssc[key], dtype=float)
            _worker['err'][key] = None
            _worker['good'][key] = np.ones(len(x), dtype=bool)

CF_NAMES = ('A_hat', 'sig_A', 'm_hat', 'b_hat', 'lnL_max', 'lnL_null', 'lnL_marg')

def _closed_form_one_loc(args):
    """
    closed-form fit at one location l, for every (w, stream width sig) and every
    dimension. same data selection and likelihood as nll_model_full (window,
    precision cut, sigma_i^2 = sig^2 + err_i^2), with the amplitude A and the linear
    baseline (slope m, intercept b) all fit exactly instead of sampled on a grid.

    the model is linear in all three: with xc = x - l,
        model_i = A*u_i + m*xc_i + b,    u_i = generic_signal(x_i, 1, w, l),
    so lnL(A, m, b) is an exact 3-d paraboloid: weighted linear least squares with
    design matrix [u, xc, 1] and normal matrix F (3x3). the nuisance block, for (m, b),
        F_nn = [[Sxx, Sx1], [Sx1, S11]],     Sab = sum a*b/s2,
    doesn't depend on w, so (m, b) are eliminated analytically with its 2x2 inverse
    (a schur complement), leaving the same one-parameter formulas as before with
    "effective" sums:
        null fit (A = 0):  (m0, b0) = F_nn^-1 (Sxy, S1y)
        Suu_eff = Suu - (Sux, Su1) F_nn^-1 (Sux, Su1)
        Suy_eff = Suy - (Sux*m0 + Su1*b0)
        A_hat   = Suy_eff/Suu_eff
        sig_A   = 1/sqrt(Suu_eff)            (= sqrt of (F^-1)_AA: m, b marginalised)
        (m_hat, b_hat) = (m0, b0) - A_hat * F_nn^-1 (Sux, Su1)
        lnL_null = -(Syy - m0*Sxy - b0*S1y)/2 - sum ln(2 pi s2)/2   (best line, no wiggle)
        lnL_max  = lnL_null + Suy_eff^2/(2 Suu_eff)                 (best wiggle + line)
    and, profiling (m, b) at each A, lnL(A) = lnL_max - (A - A_hat)^2 / (2 sig_A^2).

    lnL_marg is the peak over A of the likelihood with (m, b) integrated out under
    flat (improper) priors: lnL_max + ln(2 pi) - ln(det F_nn)/2. it's what the
    mixture weights need, since det F_nn depends on sig. the prior volume of (m, b)
    is the same for every (w, sig) and every model, so it drops out.

    the window doesn't depend on (w, sig), so it's set once per dimension. u depends
    only on w and 1/s2 only on sig, so each w-dependent sum is one
    (n_w, n_star) x (n_star, n_sig) matrix product (4 of them now, 2 before).

    returns a dict of CF_NAMES arrays, each (n_w, n_sig_max, 4). NaN where the
    window has fewer than min_stars stars in that dimension, or past the end of a
    shorter sig grid. lnL_null and lnL_marg's F_nn term don't depend on w; lnL_null
    is repeated along that axis.
    """
    l, w_grid, sig_grids, window_width, min_stars = args
    n_sig_max = max(len(g) for g in sig_grids)
    out = {name: np.full((len(w_grid), n_sig_max, len(GRID_KEYS)), np.nan) for name in CF_NAMES}

    x = _worker['x']
    window = (x < (l + window_width)) & (x > (l - window_width))
    for k, key in enumerate(GRID_KEYS):
        use = window & _worker['good'][key]
        if np.sum(use) < min_stars:
            continue
        xc = x[use] - l                                             # line centred on l
        y = _worker['data'][key][use]
        e2 = np.zeros(len(y)) if _worker['err'][key] is None else _worker['err'][key][use]**2

        sig = sig_grids[k]
        s2 = sig[:, None]**2 + e2[None, :]                          # (n_sig, n_star)
        ivar = 1.0 / s2
        u = generic_signal(xc[None, :], 1.0, w_grid[:, None])       # (n_w, n_star), wiggle only

        # w-independent sums, each (n_sig,)
        S11, Sx1, Sxx = ivar.sum(axis=1), ivar @ xc, ivar @ xc**2
        S1y, Sxy, Syy = ivar @ y, ivar @ (xc*y), ivar @ y**2
        # w-dependent sums, each (n_w, n_sig)
        Suu = u**2 @ ivar.T
        Suy = u @ (ivar * y[None, :]).T
        Su1 = u @ ivar.T
        Sux = u @ (ivar * xc[None, :]).T

        with np.errstate(divide='ignore', invalid='ignore'):
            # the nuisance block F_nn = [[Sxx, Sx1], [Sx1, S11]] and its 2x2 inverse.
            # det <= 0 only if every star has the same x: no information on the slope.
            det = S11*Sxx - Sx1**2
            det = np.where(det > 0, det, np.nan)
            # best line with no wiggle, and the F_nn^-1 (Sux, Su1) that couples it to A
            m0 = (S11*Sxy - Sx1*S1y) / det                           # (n_sig,)
            b0 = (Sxx*S1y - Sx1*Sxy) / det
            cm = (S11*Sux - Sx1*Su1) / det                           # (n_w, n_sig)
            cb = (Sxx*Su1 - Sx1*Sux) / det

            Suu_eff = Suu - (Sux*cm + Su1*cb)
            Suy_eff = Suy - (Sux*m0 + Su1*b0)
            # Suu_eff <= 0: the template is (numerically) a straight line here, no information on A
            Suu_eff = np.where(Suu_eff > 0, Suu_eff, np.nan)
            A_hat = Suy_eff / Suu_eff

            lnL_null = (-0.5*(Syy - m0*Sxy - b0*S1y)
                        - 0.5*np.sum(np.log(2*np.pi*s2), axis=1))    # (n_sig,)
            lnL_max = lnL_null[None, :] + 0.5 * Suy_eff**2 / Suu_eff

            n_s = len(sig)
            out['A_hat'][:, :n_s, k] = A_hat
            out['sig_A'][:, :n_s, k] = 1.0 / np.sqrt(Suu_eff)
            out['m_hat'][:, :n_s, k] = m0[None, :] - A_hat*cm
            out['b_hat'][:, :n_s, k] = b0[None, :] - A_hat*cb
            out['lnL_max'][:, :n_s, k] = lnL_max
            out['lnL_null'][:, :n_s, k] = lnL_null[None, :]
            out['lnL_marg'][:, :n_s, k] = lnL_max + np.log(2*np.pi) - 0.5*np.log(det)[None, :]
    return out

def gridsearch(grid_dict, ssc, uncertainties=False, n_proc=1,
               filter_window_factor=2., precision_limits=None, min_stars=5,
               progress=True):
    """
    grid search over (l, w, sig) with the amplitude A and a linear baseline
    m*(x - l) + b in each dimension fit in closed form (see _closed_form_one_loc).
    no amplitude grid needed, and no window median subtracted: b is the baseline.

    returns a dict of per-dimension arrays, each of shape

        (len(l), len(w), n_sig_max, 4)

    last axis in the order phi2, pm_phi1, pm_phi2, v_gsr (GRID_KEYS):
        'A_hat'    best-fit amplitude (jointly with m, b)
        'sig_A'    its uncertainty with m and b marginalised out, exact for a
                   gaussian likelihood (1/sqrt(Suu_eff))
        'm_hat'    best-fit baseline slope (per deg of phi1), jointly with A
        'b_hat'    best-fit baseline at x = l, jointly with A
        'lnL_max'  full lnL at (A_hat, m_hat, b_hat) (includes the -0.5 ln(2 pi sigma^2) terms)
        'lnL_null' full lnL of the best line with A = 0 (doesn't depend on w;
                   repeated along that axis)
        'lnL_marg' the peak over A of the lnL with m, b integrated out (flat priors),
                   = lnL_max + ln(2 pi) - ln(det F_nn)/2. used by mixture_weights.
    e.g. res['A_hat'][il, iw, isig, k] is the best amplitude in dimension k for a
    perturbation at l[il] with width w[iw] and intrinsic stream width
    grid_dict['sig_'+key][isig]. NaN-padded where sig grids differ in length, and NaN
    where the window has fewer than min_stars stars in that dimension.

    profiling (m, b) at each A (i.e. refitting the line for every A),
        lnL(A)   = lnL_max - (A - A_hat)**2 / (2 sig_A**2)
    and lnL_max - lnL_null = (A_hat/sig_A)**2 / 2. fit_baseline gives the (m, b)
    that goes with a given A.
    use lnL_max (full lnL), not the delta, to compare or marginalise across sig.

    grid_dict: 1-D arrays, keys 'l', 'w', plus optional 'sig_phi2', 'sig_pm_phi1',
    'sig_pm_phi2', 'sig_v_gsr' for the intrinsic stream width in each dimension. a
    dimension with no 'sig_' key is held at DEFAULT_STREAMWIDTHS (sig axis length 1).
    amplitude keys ('phi2', ...) are ignored if present.

    uncertainties: bool. False = noiseless columns ssc[key], no measurement errors.
    True = noised columns ssc[key+"_noised"] (and phi1_noised), ssc[key+"_err"] added
    in quadrature with the stream width, and per-dimension precision cuts with
    obs_limit. ssc then needs the _noised and _err columns (e.g. ssc_table).
    precision_limits: dict of per-dimension precision cuts for obs_limit when
    uncertainties is True. None uses PRECISION_REQUIREMENTS_DICT.

    filter_window_factor: the filter window half-width around l is
    filter_window_factor*max(w_grid) (default 2). it's the same for every (w, sig) at
    a given l, which the marginalisation needs (every model at a given l must be
    scored on the same stars for their lnL to be comparable). the old "2w" option is
    gone for that reason. the half-width used is returned as res['window_width'].

    min_stars: windows with fewer stars than this in a dimension (after any
    precision cut) come back NaN in that dimension.

    n_proc: number of worker processes (parallelised over the location grid).
    1 runs in this process with no multiprocessing at all.

    progress: show a tqdm progress bar over the location grid.
    """
    loc_grid = np.asarray(grid_dict['l'])
    w_grid = np.asarray(grid_dict['w'], dtype=float)
    sig_grids = [_sig_grid(grid_dict, key).astype(float) for key in GRID_KEYS]

    if not np.isscalar(filter_window_factor) or not filter_window_factor > 0:
        raise ValueError("filter_window_factor must be a positive number, got %r"
                         % (filter_window_factor,))
    window_width = float(filter_window_factor)*np.max(w_grid)

    tasks = [(l, w_grid, sig_grids, window_width, min_stars) for l in loc_grid]

    if n_proc <= 1:
        _init_worker(ssc, uncertainties, precision_limits)
        results = [_closed_form_one_loc(t) for t in tqdm(tasks, disable=not progress)]
    else:
        with mp.Pool(n_proc, initializer=_init_worker, initargs=(ssc, uncertainties, precision_limits)) as pool:
            results = list(tqdm(pool.imap(_closed_form_one_loc, tasks), total=len(tasks),
                                disable=not progress))

    out = {name: np.array([r[name] for r in results]) for name in CF_NAMES} # each (n_l, n_w, n_sig_max, 4)
    out['window_width'] = window_width # the filter window half-width used (deg)
    return out

def fit_baseline(ssc, key, l, w, A, sig, window_width,
                 uncertainties=False, precision_limits=None):
    """
    the best linear baseline (m, b) in dimension key given a wiggle (l, w, A): a
    weighted least-squares fit of y - A*u on [x - l, 1], over the same stars and with
    the same sigma_i^2 = sig^2 + err_i^2 as gridsearch. returns (m, b), or
    (nan, nan) if the window has fewer than 2 stars.

    at A = res['A_hat'] this gives res['m_hat'], res['b_hat'] (an independent check
    of the closed form). at any other A it's the line that goes with that A, e.g. for
    drawing a wiggle with the marginal-median A and w over the data.
    """
    xw, yw, errs = _window_data(ssc, key, l, window_width,
                                uncertainties=uncertainties, precision_limits=precision_limits)
    if len(xw) < 2:
        return np.nan, np.nan
    s = np.sqrt(sig**2 + errs**2)                 # errs is zeros without uncertainties
    resid = yw - generic_signal(xw, A, w, l)
    X = np.column_stack([xw - l, np.ones(len(xw))])
    (m, b), *_ = np.linalg.lstsq(X / s[:, None], resid / s, rcond=None)
    return m, b

def plot_stream(table, i, orbit, **scatter_kwargs):
    row = table[i]

    indir = "/n/netscratch/conroy_lab/Lab/amphillips/tsetse_analysis/perturbed_ps_streams_dissolved/%s/"%orbit

    fname = "%s_%i.pickle"%(orbit, i)

    with open(indir+fname, 'rb') as handle:
        pert_dict_in = pickle.load(handle)

    ssc = pert_dict_in["ssc"]
    ssc_pert = pert_dict_in["ssc_pert"] # <-- observed location of perturbation

    # plot a 4-row, 1-column subplots array of phi2, pmphi1, pmphi2, vgsr vs phi1 here. 
    # see way way below for example of how we would do this. 
    fig, axs = plt.subplots(4, 1, figsize=[8,12])
    for d_index in range(4):
        axs[d_index].scatter(ssc['phi1'], ssc[GRID_KEYS[d_index]],
                                **scatter_kwargs)
                            # c='k', s=1, rasterized=True)
        

        axs[d_index].axvline(ssc_pert.phi1.to(u.degree).value, c='tomato', lw=0.1, zorder=0)

    return fig, axs

PRECISION_REQUIREMENTS_DICT = {
    "phi1":500 * u.microarcsecond.to(u.degree), 
    "phi2":500 * u.microarcsecond.to(u.degree), 
    "pm_phi1":0.3, # mas/yr
    "pm_phi2":0.3,
    "v_gsr":1.0 # km/s
}
def obs_limit(table, key, precision_limit=None):
    """
    returns a boolean mask array that indexes 
    the elements in the table such that column 
    key's uncertainty is below the precision
    limit. 

    if precison limit is not explicitly given , pull 
    from the all caps dictionary defined above this function. 
    which I did so that the number of surviving stars roughly matches
    the number of GD1 members known now. idk. 
    """
    pl = precision_limit if precision_limit is not None else PRECISION_REQUIREMENTS_DICT[key]
    k = key+"_err"
    good = np.asarray(table[k])<=pl
    return good



def mixture_weights(res, grid_dict, k, w_prior='flat'):
    """
    log weights lnW of the (w, sig) grid points in p(A | l) for dimension k,
    normalised over (w, sig) at each l. shape (n_l, n_w, n_sig_max).

    at each (w, sig) =: j, with the baseline (m, b) integrated out (flat priors), the
    likelihood in A is L_marg,j * exp(-(A-A_hat_j)^2 / 2 sig_A,j^2), whose integral
    over A is L_marg,j * sqrt(2 pi) * sig_A,j: the evidence for j with A, m and b
    integrated out. so, with a flat prior on A,
        p(A | l) = sum_j W_j N(A; A_hat_j, sig_A,j),   W_j ∝ prior_j * L_marg,j * sig_A,j.
    (weighting by L_max alone would profile over A instead of marginalising.)
    L_marg,j = L_max,j * 2 pi / sqrt(det F_nn,j) (see _closed_form_one_loc); the
    det F_nn factor depends on sig but not w.

    uses the full lnL (lnL_marg), not a difference to lnL_null, since sig varies.

    w_prior: prior weight per w grid point. 'flat' = equal weights, 'log' = 1/w.
    on a linearly spaced w grid those are flat in w and flat in ln w; on a
    log-spaced grid 'flat' is already flat in ln w. sig always gets equal weights.

    NaN grid points (empty windows, padding) get zero weight. an l with no usable
    grid point gives a NaN row.
    """
    lnW = res['lnL_marg'][..., k] + np.log(res['sig_A'][..., k])
    if w_prior == 'log':
        lnW = lnW - np.log(np.asarray(grid_dict['w'], dtype=float))[None, :, None]
    elif w_prior != 'flat':
        raise ValueError("w_prior must be 'flat' or 'log', got %r" % (w_prior,))
    lnW = np.where(np.isfinite(lnW), lnW, -np.inf)
    with np.errstate(divide='ignore', invalid='ignore'):
        lnW = lnW - logsumexp(lnW, axis=(1, 2), keepdims=True) # all -inf l -> NaN row
    return lnW

def marginal_A(res, grid_dict, k, q=(0.05, 0.5, 0.95), w_prior='flat'):
    """
    the marginal posterior p(A | l) in dimension k, with w and sig marginalised out
    (flat prior on A, see mixture_weights for the rest). it is a mixture of
    gaussians, sum_j W_j N(A; A_hat_j, sig_A,j), so nothing here depends on an
    amplitude grid and nothing clips at a grid edge.

    returns a dict of arrays over l:
        'q'           (n_l, len(q)) quantiles of p(A | l), root-found (brentq) on
                      the exact mixture CDF, sum_j W_j Phi((A - A_hat_j)/sig_A,j),
                      to 1e-6 of the smallest sig_A. q must be strictly in (0, 1).
        'mean'        sum_j W_j A_hat_j
        'var_within'  sum_j W_j sig_A,j^2: the noise at fixed (w, sig)
        'var_between' sum_j W_j (A_hat_j - mean)^2: how much A_hat moves across
                      plausible (w, sig). big = a w-A degeneracy.
        'std'         sqrt(var_within + var_between)
        'P_plus'      P(A > 0 | data, l) = sum_j W_j Phi(A_hat_j / sig_A,j)
        'z'           Phi^-1(P_plus). the detection statistic: |z| >= 3 means 99.87%
                      of p(A | l) is on one side of zero; the sign is the sign of
                      the wiggle. with one (w, sig) it's A_hat/sig_A. computed from
                      whichever of P_plus, P_minus is smaller, entirely in logs
                      (log_ndtr / ndtri_exp), so it stays finite for strong signals
                      where P_plus rounds to exactly 1.
    an l with no usable window is NaN throughout.

    the mixture can be bimodal (some w prefer A ~ 0, others a real wiggle), in which
    case mean/std describe neither mode. plot mixture_pdf when z looks odd.
    """
    lnW = mixture_weights(res, grid_dict, k, w_prior=w_prior)
    A_hat = res['A_hat'][..., k]
    sig_A = res['sig_A'][..., k]
    n_l = lnW.shape[0]

    out = {name: np.full(n_l, np.nan) for name in
           ('mean', 'var_within', 'var_between', 'std', 'P_plus', 'z')}
    out['q'] = np.full((n_l, len(q)), np.nan)

    for il in range(n_l):
        m = np.isfinite(lnW[il])                     # components with nonzero weight
        if not m.any():
            continue
        lw, a, s = lnW[il][m], A_hat[il][m], sig_A[il][m]
        W = np.exp(lw)

        mean = np.sum(W * a)
        out['mean'][il] = mean
        out['var_within'][il] = np.sum(W * s**2)
        out['var_between'][il] = np.sum(W * (a - mean)**2)
        out['std'][il] = np.sqrt(out['var_within'][il] + out['var_between'][il])

        lnP_plus = logsumexp(lw + log_ndtr(a / s))
        lnP_minus = logsumexp(lw + log_ndtr(-a / s))
        out['P_plus'][il] = np.exp(lnP_plus)
        out['z'][il] = -ndtri_exp(lnP_minus) if lnP_minus < lnP_plus else ndtri_exp(lnP_plus)

        # quantiles: root-find the exact mixture CDF. the bracket is set by the
        # components that carry weight, so negligible ones can't stretch it.
        big = W > 1e-10
        lo, hi = np.min(a[big] - 8*s[big]), np.max(a[big] + 8*s[big])
        cdf = lambda A: W @ ndtr((A - a) / s)
        xtol = 1e-6 * np.min(s[big])
        out['q'][il] = [brentq(lambda A: cdf(A) - qq, lo, hi, xtol=xtol) for qq in q]

    return out

def mixture_pdf(res, grid_dict, k, il, A_ax, w_prior='flat'):
    """
    p(A | l = grid_dict['l'][il]) in dimension k, evaluated on A_ax. for plotting,
    e.g. to see whether the mixture is bimodal.
    """
    lnW = mixture_weights(res, grid_dict, k, w_prior=w_prior)[il]
    m = np.isfinite(lnW)
    a, s = res['A_hat'][il][..., k][m], res['sig_A'][il][..., k][m]
    gauss = np.exp(-0.5*((A_ax[None, :] - a[:, None]) / s[:, None])**2) / (np.sqrt(2*np.pi) * s[:, None])
    return np.exp(lnW[m]) @ gauss

def marginal_quantiles(res, grid_dict, k, keep='w', q=(0.05, 0.5, 0.95), w_prior='flat'):
    """
    shoutout chris k for your wisdom.
    quantiles of p(w | l) (keep='w') or p(sig | l) (keep='sig') in dimension k, with
    A (analytically) and the other grid parameter marginalised out. for p(A | l)
    use marginal_A.

    the weights are the same W_j as p(A | l) (see mixture_weights), summed over the
    other axis. the quantiles are read off the CDF with each grid value treated as
    a bin centre, so they're limited by the grid spacing: sig in particular is very
    well constrained with thousands of stars, and its interval is often just the
    spacing.

    returns the quantiles, (n_l, len(q)), and the normalised posterior p, (n_l, n_vals).
    an l with no usable window gives NaN rows.
    """
    if keep == 'A':
        raise ValueError("p(A | l) is a gaussian mixture now, use marginal_A")
    key = GRID_KEYS[k]
    sig_grid = _sig_grid(grid_dict, key)
    W = np.exp(mixture_weights(res, grid_dict, k, w_prior=w_prior)[:, :, :len(sig_grid)])
    p = {'w': np.sum(W, axis=2), 'sig': np.sum(W, axis=1)}[keep]
    vals = {'w': np.asarray(grid_dict['w']), 'sig': sig_grid}[keep]
    cdf = np.cumsum(p, axis=1) - p/2                  # cdf at each grid value as a bin centre
    return np.array([np.interp(q, c, vals) if np.all(np.isfinite(c)) else np.full(len(q), np.nan)
                     for c in cdf]), p

def profile_best(res, grid_dict):
    """
    profile over (w, sig) at every l, separately per dimension. A, m and b are
    already profiled: they're A_hat, m_hat, b_hat at each (w, sig).

    returns a dict of (n_l, 4) arrays:
        'w', 'sig', 'A', 'sig_A', 'm', 'b'  at the (w, sig) that maximise lnL_max
        'dll'  max_(w,sig) lnL_max - max_sig lnL_null: a likelihood ratio of wiggle +
               line against line alone, each getting its own best sig
        'z_profile'  A/sig_A at that point, the significance conditional on (w, sig)
    an l with no usable window in a dimension is NaN in that dimension.
    """
    n_l, n_w, n_s, n_k = res['lnL_max'].shape
    out = {name: np.full((n_l, n_k), np.nan) for name in ('w', 'sig', 'A', 'sig_A', 'm', 'b', 'dll', 'z_profile')}
    w_grid = np.asarray(grid_dict['w'])
    for k, key in enumerate(GRID_KEYS):
        sig_grid = _sig_grid(grid_dict, key)
        lnL = res['lnL_max'][..., k].reshape(n_l, -1)
        ok = np.any(np.isfinite(lnL), axis=1)
        if not ok.any():
            continue
        ib = np.nanargmax(lnL[ok], axis=1)
        iw, isig = np.unravel_index(ib, (n_w, n_s))
        il = np.flatnonzero(ok)
        out['w'][il, k] = w_grid[iw]
        out['sig'][il, k] = sig_grid[isig]
        out['A'][il, k] = res['A_hat'][il, iw, isig, k]
        out['sig_A'][il, k] = res['sig_A'][il, iw, isig, k]
        out['m'][il, k] = res['m_hat'][il, iw, isig, k]
        out['b'][il, k] = res['b_hat'][il, iw, isig, k]
        out['dll'][il, k] = lnL[il, ib] - np.nanmax(res['lnL_null'][il, 0, :, k], axis=1)
    out['z_profile'] = out['A'] / out['sig_A']
    return out

def _fit_wiggle_line(x, y, err, good, l, w, sig, window_width, min_stars=5):
    """
    weighted least-squares fit of A*u + m*(x - l) + b at one fixed (l, w, sig), over
    the stars in the window |x - l| < window_width that pass the mask good. the
    same model and weights as gridsearch, on arbitrary data arrays (e.g. a residual).
    returns (A, sig_A, m, b), all NaN if the window has fewer than min_stars stars.
    """
    use = good & (x < (l + window_width)) & (x > (l - window_width))
    if np.sum(use) < min_stars:
        return np.nan, np.nan, np.nan, np.nan
    xw, yw = x[use], y[use]
    s = np.sqrt(sig**2 + err[use]**2)
    X = np.column_stack([generic_signal(xw, 1.0, w, l), xw - l, np.ones(len(xw))]) / s[:, None]
    (A, m, b), *_ = np.linalg.lstsq(X, yw / s, rcond=None)
    sig_A = np.sqrt(np.linalg.inv(X.T @ X)[0, 0])
    return A, sig_A, m, b

def clean(ssc, grid_dict, z_thresh=5., max_iter=20, n_refit=5, refit_lw=True, dl_refit=3.,
          uncertainties=False, filter_window_factor=2., precision_limits=None,
          min_stars=5, w_prior='flat', n_proc=1, verbose=True):
    """
    iterative subtraction ("CLEAN") of detected wiggles, separately in each dimension.
    each round:
      1. run gridsearch + marginal_A on the current residual, and find the l with the
         highest |z| in each dimension that's still active.
      2. if that |z| < z_thresh, the dimension is done.
      3. otherwise record a detection there, with the profile's best (w, sig) at that
         l, and add its wiggle to that dimension's model.
      4. refit the amplitudes of all of that dimension's wiggles jointly, with their
         (l, w, sig) held fixed (see below), and subtract them all from the original
         data, over the whole stream. go back to 1.
    a ghost peak only exists because a stronger wiggle nearby has a lobe in its
    window, so once that wiggle is subtracted the ghost goes away, while independent
    wiggles survive. each round is a nested comparison: k+1 wiggles against k.

    what's subtracted: the profile's best (w, sig) at the peak, i.e. the maximum
    likelihood wiggle there. (the marginal medians of A and w come from different
    marginals, so they needn't fit the data together, and a mismatched pair leaves a
    residual wiggle behind.) the baseline line is not subtracted: each window fits
    its own line in every round.

    joint refit: each wiggle is refit to the data minus all the other wiggles,
    cycling through the wiggles n_refit times (gauss-seidel / coordinate descent).
    this matters when wiggles overlap: the first one's single-wiggle fit absorbed
    part of its neighbour, and the refit gives that back.
      refit_lw=True (default): each wiggle's (l, w, sig) is re-optimised too, by
        searching l within +/- dl_refit deg of its current l (on the l grid) and
        the whole (w, sig) grid. l is picked by the profile likelihood ratio
        against the line alone (profile_best's 'dll'), since lnL itself at different
        l is computed on different stars and can't be compared; (w, sig) are then
        the ones maximising lnL_max at that l. without this, two overlapping
        wiggles get found first as one wide blend (w at the top of the grid) whose
        shape can never change, and CLEAN patches the leftovers with extra
        components. if a wiggle moves onto the same (l, w) as another, it's merged
        into it (dropped; the other absorbs its amplitude on the next pass).
      refit_lw=False: only the amplitudes are refit, with (l, w, sig) fixed. linear,
        so this converges to the joint least-squares fit of the amplitudes.
    either way A, m, b come from _fit_wiggle_line (each wiggle with its own line, in
    its own window).

    stopping: a dimension is done when its highest |z| is below z_thresh, or if the
    peak lands on an (l, w) already in its model (a residual that the refit can't
    remove; flagged with 'repeat' in the history). the loop stops when every
    dimension is done or after max_iter rounds of subtraction.

    ssc: the stream table, as for gridsearch (with the _noised/_err columns if
    uncertainties). not modified. grid_dict, uncertainties, filter_window_factor,
    precision_limits, min_stars, n_proc: as for gridsearch. w_prior: as for
    marginal_A. z_thresh is a per-l threshold, same caveats as in the script.

    returns a dict:
        'detections'  list over GRID_KEYS of lists of dicts, one per wiggle, in the
                      order found: 'round', 'z' (marginal z when it was found),
                      'z_profile', 'l_found', 'w_found', 'A_found' (the profile values
                      when found), and after the final refit 'l', 'w', 'sig', 'A',
                      'sig_A', 'm', 'b' (its own line), 'z_refit' (= A/sig_A at
                      fixed l, w, sig, with the others subtracted), and 'dl_prev'
                      (distance in l from where it was found to the nearest wiggle
                      found before it in that dimension; NaN for the first).
                      merged wiggles are dropped.
        'resid'       a dict of the residual columns (data minus all wiggles), in
                      the same layout as ssc, so it can go straight into gridsearch
        'res'         gridsearch on the final residual
        'margs'       marginal_A per dimension on the final residual
        'history'     list over rounds of dicts {key: (l, z)} of the peak found in
                      each active dimension that round, with 'done' / 'repeat' notes
        'window_width', 'n_rounds'
    """
    pls = {} if precision_limits is None else precision_limits
    xcol = 'phi1_noised' if uncertainties else 'phi1'
    x = np.asarray(ssc[xcol], dtype=float)
    dcol = {key: key+"_noised" if uncertainties else key for key in GRID_KEYS}
    y0, err, good = {}, {}, {}
    for key in GRID_KEYS:
        y0[key] = np.asarray(ssc[dcol[key]], dtype=float)
        if uncertainties:
            err[key] = np.asarray(ssc[key+"_err"], dtype=float)
            good[key] = obs_limit(ssc, key, precision_limit=pls.get(key))
        else:
            err[key] = np.zeros(len(x))
            good[key] = np.ones(len(x), dtype=bool)

    # the residual table that gridsearch sees each round
    resid = {xcol: x}
    for key in GRID_KEYS:
        resid[dcol[key]] = y0[key].copy()
        if uncertainties:
            resid[key+"_err"] = err[key]

    window_width = float(filter_window_factor) * np.max(np.asarray(grid_dict['w'], dtype=float))
    l_grid = np.asarray(grid_dict['l'])
    comps = [[] for _ in GRID_KEYS]          # per dimension: dicts with l, w, sig, u, A, ...
    done = [False] * len(GRID_KEYS)
    final_margs = [None] * len(GRID_KEYS)
    history = []

    def best_lw_near(k, y, l0):
        """the best (l, w, sig) for one wiggle in dimension k, on data y (the data
        minus the other wiggles), with l within dl_refit of l0. None if no usable l."""
        near = l_grid[np.abs(l_grid - l0) <= dl_refit + 1e-9]
        tab = dict(resid)
        tab[dcol[GRID_KEYS[k]]] = y
        r = gridsearch({**grid_dict, 'l': near}, tab, uncertainties=uncertainties,
                       filter_window_factor=filter_window_factor,
                       precision_limits=precision_limits, min_stars=min_stars, progress=False)
        p = profile_best(r, {**grid_dict, 'l': near})
        if not np.any(np.isfinite(p['dll'][:, k])):
            return None
        il = np.nanargmax(p['dll'][:, k])
        return near[il], p['w'][il, k], p['sig'][il, k]

    for it in range(max_iter + 1):
        res = gridsearch(grid_dict, resid, uncertainties=uncertainties, n_proc=n_proc,
                         filter_window_factor=filter_window_factor,
                         precision_limits=precision_limits, min_stars=min_stars,
                         progress=False)
        prof = profile_best(res, grid_dict)
        hist = {}
        for k, key in enumerate(GRID_KEYS):
            if done[k]:
                continue
            mg = marginal_A(res, grid_dict, k, w_prior=w_prior)
            final_margs[k] = mg
            absz = np.nan_to_num(np.abs(mg['z']), nan=0.0)
            il = int(np.argmax(absz))
            hist[key] = (l_grid[il], mg['z'][il])
            if absz[il] < z_thresh or it == max_iter:
                done[k] = True
                hist[key] += ('done',)
                continue
            l_new, w_new, sig_new = l_grid[il], prof['w'][il, k], prof['sig'][il, k]
            if any(c['l'] == l_new and c['w'] == w_new for c in comps[k]):
                done[k] = True
                hist[key] += ('repeat',)
                continue
            prev = [c['l'] for c in comps[k]]
            comps[k].append({'round': it, 'z': mg['z'][il], 'z_profile': prof['z_profile'][il, k],
                             'l_found': l_new, 'w_found': w_new, 'A_found': prof['A'][il, k],
                             'l': l_new, 'w': w_new, 'sig': sig_new, 'A': prof['A'][il, k],
                             'dl_prev': np.min(np.abs(np.array(prev) - l_new)) if prev else np.nan,
                             'u': generic_signal(x, 1.0, w_new, l_new)})

            # joint refit of this dimension's wiggles (gauss-seidel), then subtract.
            # a lone wiggle stays where it was found; it's already the best fit there.
            cs = comps[k]
            for _ in range(n_refit if len(cs) > 1 else 1):
                j = 0
                while j < len(cs):
                    c = cs[j]
                    others = sum((ci['A'] * ci['u'] for i, ci in enumerate(cs) if i != j),
                                 np.zeros(len(x)))
                    if refit_lw and len(cs) > 1:
                        best = best_lw_near(k, y0[key] - others, c['l'])
                        if best is not None:
                            l_c, w_c, sig_c = best
                            if any(i != j and ci['l'] == l_c and ci['w'] == w_c
                                   for i, ci in enumerate(cs)):
                                cs.pop(j) # moved onto another wiggle: merge into it
                                continue
                            c.update(l=l_c, w=w_c, sig=sig_c, u=generic_signal(x, 1.0, w_c, l_c))
                    A, sA, m, b = _fit_wiggle_line(x, y0[key] - others, err[key], good[key],
                                                   c['l'], c['w'], c['sig'], window_width,
                                                   min_stars=min_stars)
                    c.update(A=A, sig_A=sA, m=m, b=b, z_refit=A/sA)
                    j += 1
            # a wiggle whose window ran out of stars can't be fit (NaN A): drop it
            cs[:] = [c for c in cs if np.isfinite(c['A'])]
            resid[dcol[key]] = y0[key] - sum((c['A'] * c['u'] for c in cs), np.zeros(len(x)))
        history.append(hist)
        if verbose:
            print("round %i: " % it + ", ".join(
                "%s l=%.0f z=%+.1f%s" % (key, h[0], h[1], " (%s)" % h[2] if len(h) > 2 else "")
                for key, h in hist.items()))
        if all(done):
            break

    detections = [[{n: v for n, v in c.items() if n != 'u'} for c in cs] for cs in comps]
    return {'detections': detections, 'resid': resid, 'res': res, 'margs': final_margs,
            'history': history, 'window_width': window_width, 'n_rounds': len(history)}


YLABELS = {'phi2': r'$\phi_2$ [deg]', 'pm_phi1': r'$\mu_{\phi_1}$ [mas/yr]',
           'pm_phi2': r'$\mu_{\phi_2}$ [mas/yr]', 'v_gsr': r'$v_{\rm gsr}$ [km/s]'}
# the bare math symbol per dimension, for composite labels like A_{phi2}/sigma
SYMBOLS = {'phi2': r'\phi_2', 'pm_phi1': r'\mu_{\phi_1}',
           'pm_phi2': r'\mu_{\phi_2}', 'v_gsr': r'v_{\rm gsr}'}


def find_z_peaks(margs, z_thresh=5.):
    """
    detections: the local maxima of |z(l)| that reach z_thresh, in each dimension.

    margs: list of marginal_A outputs, one per dimension (GRID_KEYS order).
    returns a list of index arrays into the l grid, one per dimension.

    |z| is padded with a zero at each end so a peak on the first or last l still counts
    (find_peaks never returns an endpoint). NaN z (no usable window) counts as 0.
    """
    peaks = []
    for mg in margs:
        absz = np.nan_to_num(np.abs(mg['z']), nan=0.0)
        ipk, _ = find_peaks(np.concatenate([[0.0], absz, [0.0]]), height=z_thresh)
        peaks.append(ipk - 1)
    return peaks


def _edge_stats(p_w, w_grid, edge_width):
    """for rows of p_w (p(w | l) on w_grid): the posterior mass in the edge band (the top
    edge_width of w_grid, at least the top grid point), and whether the mode is in it."""
    w_grid = np.asarray(w_grid, dtype=float)
    edge = w_grid >= np.max(w_grid) - edge_width
    edge[np.argmax(w_grid)] = True
    p_edge = np.sum(p_w[..., edge], axis=-1)
    finite = np.all(np.isfinite(p_w), axis=-1)
    mode_at_edge = finite & edge[np.argmax(np.nan_to_num(p_w, nan=-1.), axis=-1)]
    return p_edge, mode_at_edge

def ghost_flags(margs, grid_dict, res, peaks, w_prior='flat', edge_frac=0.05,
                edge_test='mode', p_edge_thresh=0.5, parent_dist=None,
                require_opposite_sign=True, parent_must_be_interior=False,
                flag_ambiguous=False, rerun=False, ssc=None, uncertainties=False,
                precision_limits=None, min_stars=5, w_extend=2., rerun_fixed_window=True,
                n_proc=1, verbose=True):
    """
    flags peaks of |z(l)| (find_z_peaks) that look like ghosts, from the shape of p(w | l).

    a ghost is a wide template whose window catches the outer lobe of a stronger
    wiggle nearby. that lobe looks like curvature the line baseline can't absorb, and a
    wider template fits it better, so p(w | l) rises all the way to max(w) instead of
    peaking at a resolved width. the tests, per dimension and per peak:
      1. edge: is p(w | l) up against the top of the w grid? the edge band is the top
         edge_frac of the w range (at least the top grid point). edge_test='mode': the
         mode of p(w | l) is in the band (catches broad posteriors that rise to the edge
         but don't pile up in the band). 'mass': p_edge, the mass in the band, is
         >= p_edge_thresh. not at the edge -> 'real'.
      2. parent: an edge peak is a 'ghost' if another peak in the same dimension has
         larger |z|, opposite sign (if require_opposite_sign), and lies within
         parent_dist. the parent can itself be at the edge (parent_must_be_interior=
         False), so ghosts of ghosts and ghosts of a real feature wider than max(w) are
         caught. otherwise 'ambiguous': a real feature wider than max(w), leftover
         curvature from the straightening, or a ghost whose parent didn't make the
         peak list.
      3. (rerun=True) move the edge: the ambiguous peaks' l are searched again with the
         w grid extended to w_extend*max(w) (the original grid plus more points at its
         last spacing). if p(w | l) is still at the (new) edge by the same edge_test, with
         the same absolute band width, it's 'edge' (a ghost or a trend, not a resolved
         wiggle); if it settles inside the new grid it's 'wide' (a real feature wider
         than the original grid).
    only edge peaks are ever flagged: a peak with a resolved width is 'real' whatever
    its neighbours are.

    margs, grid_dict, res, peaks: marginal_A list (one per dimension), the grid, the
        gridsearch output and find_z_peaks output. margs must have been made with the
        same w_prior as passed here.
    w_prior: as for mixture_weights.
    edge_frac: the edge band is the top edge_frac*(max(w) - min(w)) of the w grid.
    edge_test: 'mode' or 'mass', see test 1. p_edge_thresh is only used for 'mass'.
    parent_dist: max |l_peak - l_parent| (deg). default window_width + max(w): the
        ghost's window has to reach the parent's lobe.
    require_opposite_sign: a parent must have the opposite sign of z.
    parent_must_be_interior: only peaks with a resolved width (not at the edge) can be
        parents.
    flag_ambiguous: whether 'ambiguous' peaks are flagged (default False: shown as real).
    rerun: do test 3 (one extra gridsearch over the ambiguous peaks' l only; nothing is
        rerun if there are none). needs ssc, plus uncertainties, precision_limits,
        min_stars and n_proc as they were passed to the original gridsearch.
    w_extend: the extended grid goes up to w_extend*max(w).
    rerun_fixed_window: True keeps the window half-width of the original search, so the
        same stars are scored and the only change is the wider template family (the
        results at the original w are identical to res). False keeps
        filter_window_factor instead, so the window grows with max(w).
    verbose: print each peak's class.

    returns a list (one per dimension, GRID_KEYS order) of dicts of arrays aligned with
    peaks[k]:
        'il'          index into the l grid (= peaks[k])
        'l', 'z'      the peak's l and z
        'p_edge'      edge mass of p(w | l)
        'w_med'       median of p(w | l) (grid-limited, see marginal_quantiles)
        'parent'      index into the l grid of the parent, -1 if none
        'class'       'real', 'ghost', 'ambiguous', or after a rerun 'edge' / 'wide'
        'flag'        True for 'ghost' and 'edge' (and 'ambiguous' if flag_ambiguous)
        'p_edge_rerun', 'w_med_rerun', 'z_rerun'  the same on the extended grid (NaN
                      where the peak wasn't rerun)
        'w_rerun'     the extended w grid (None if nothing in this dimension was rerun)
    """
    if edge_test not in ('mode', 'mass'):
        raise ValueError("edge_test must be 'mode' or 'mass', got %r" % (edge_test,))
    l_grid = np.asarray(grid_dict['l'], dtype=float)
    w_grid = np.asarray(grid_dict['w'], dtype=float)
    w_max = np.max(w_grid)
    edge_width = edge_frac * (w_max - np.min(w_grid))
    window_width = res['window_width']
    if parent_dist is None:
        parent_dist = window_width + w_max

    def at_edge_of(p_w, w):
        p_edge, mode_at_edge = _edge_stats(p_w, w, edge_width)
        return p_edge, (mode_at_edge if edge_test == 'mode' else p_edge >= p_edge_thresh)

    out = []
    for k in range(len(GRID_KEYS)):
        ipk = np.asarray(peaks[k], dtype=int)
        n = len(ipk)
        q_w, p_w = marginal_quantiles(res, grid_dict, k, keep='w', q=(0.5,), w_prior=w_prior)
        z = margs[k]['z'][ipk]
        p_edge, at_edge = at_edge_of(p_w[ipk], w_grid)

        parent = np.full(n, -1)
        cls = np.where(at_edge, 'ambiguous', 'real').astype('<U9')
        for i in np.where(at_edge)[0]:
            cand = [j for j in range(n) if j != i
                    and not (parent_must_be_interior and at_edge[j])
                    and np.abs(z[j]) > np.abs(z[i])
                    and np.abs(l_grid[ipk[j]] - l_grid[ipk[i]]) <= parent_dist
                    and (not require_opposite_sign or np.sign(z[j]) == -np.sign(z[i]))]
            if cand:
                parent[i] = ipk[max(cand, key=lambda j: np.abs(z[j]))]
                cls[i] = 'ghost'

        out.append({'il': ipk, 'l': l_grid[ipk], 'z': z, 'p_edge': p_edge,
                    'w_med': q_w[ipk, 0], 'parent': parent, 'class': cls,
                    'p_edge_rerun': np.full(n, np.nan), 'w_med_rerun': np.full(n, np.nan),
                    'z_rerun': np.full(n, np.nan), 'w_rerun': None})

    # test 3: move the edge, for the ambiguous peaks only (one gridsearch over their l)
    il_amb = np.unique(np.concatenate([g['il'][g['class'] == 'ambiguous'] for g in out]
                                      + [np.array([], dtype=int)])).astype(int)
    if rerun and len(il_amb):
        if ssc is None:
            raise ValueError("rerun=True needs ssc (the table the gridsearch was run on)")
        w_sorted = np.sort(w_grid)
        dw = w_sorted[-1] - w_sorted[-2] if len(w_sorted) > 1 else w_sorted[-1]
        n_ext = max(int(np.ceil((w_extend - 1) * w_max / dw)), 1)
        w_new = np.concatenate([w_sorted, w_max + dw*np.arange(1, n_ext+1)])
        grid_new = dict(grid_dict, l=l_grid[il_amb], w=w_new)
        factor = window_width / (np.max(w_new) if rerun_fixed_window else w_max)
        res_new = gridsearch(grid_new, ssc, uncertainties=uncertainties, n_proc=n_proc,
                             filter_window_factor=factor, precision_limits=precision_limits,
                             min_stars=min_stars, progress=verbose)
        for k, g in enumerate(out):
            amb = np.where(g['class'] == 'ambiguous')[0]
            if not len(amb):
                continue
            rows = np.searchsorted(il_amb, g['il'][amb])
            q_new, p_new = marginal_quantiles(res_new, grid_new, k, keep='w', q=(0.5,), w_prior=w_prior)
            z_new = marginal_A(res_new, grid_new, k, w_prior=w_prior)['z']
            pe_new, still_at_edge = at_edge_of(p_new[rows], w_new)
            g['p_edge_rerun'][amb] = pe_new
            g['w_med_rerun'][amb] = q_new[rows, 0]
            g['z_rerun'][amb] = z_new[rows]
            g['class'][amb] = np.where(still_at_edge, 'edge', 'wide')
            g['w_rerun'] = w_new

    for g in out:
        g['flag'] = np.isin(g['class'], ['ghost', 'edge'] + (['ambiguous'] if flag_ambiguous else []))

    if verbose:
        for k, g in enumerate(out):
            for i in range(len(g['il'])):
                line = ("%-8s l = %6.1f: z = %+6.2f, p_edge = %.2f, w_med = %.2f -> %s"
                        % (GRID_KEYS[k], g['l'][i], g['z'][i], g['p_edge'][i], g['w_med'][i], g['class'][i]))
                if g['parent'][i] >= 0:
                    line += " (parent at l = %.1f)" % l_grid[g['parent'][i]]
                if np.isfinite(g['p_edge_rerun'][i]):
                    line += (" (rerun to w = %.2f: p_edge = %.2f, w_med = %.2f, z = %+.2f)"
                             % (np.max(g['w_rerun']), g['p_edge_rerun'][i],
                                g['w_med_rerun'][i], g['z_rerun'][i]))
                print(line)
    return out


def plot_detections(res, grid_dict, ssc, uncertainties=False, z_thresh=5., phi1_true=None,
                    subtract_poly=False, poly_deg=5, prog_subtracted=None,
                    margs=None, prof=None, w_meds=None, peaks=None, w_prior='flat',
                    precision_limits=None, ghosts=None, show_ghosts=True, verbose=True):
    """
    the 4x3 diagnostic figure for one gridsearch, one row per dimension:
        left:   the noiseless stream, every star (no precision cut), for reference
        middle: the detection statistic z(l), labelled A/sigma (it's the analogue of the
                amplitude over its uncertainty; see marginal_A), with +-z_thresh lines
                and the peaks (find_z_peaks) as green dots
        right:  the stream the filter saw, with a wiggle drawn at each peak of |z|
                (marginal medians of A and w, on the line fit_baseline gives at the
                profile's best sig)

    res, grid_dict, ssc: the gridsearch output, its grid, and the table it was run on.
        ssc needs the noiseless columns ('phi1' and GRID_KEYS) for the left column.
    uncertainties: as passed to gridsearch. True plots the <key>_noised columns against
        phi1_noised in the right column, with stars failing obs_limit in light grey.
        False plots the noiseless columns, all black (so the right column is the left
        one plus the wiggles).
    phi1_true: optional true impact location, drawn as a green vertical line. also turns
        on the "z at truth" printout.
    subtract_poly: fit a degree-poly_deg polynomial in phi1 to the stars that pass the
        precision cut (all stars if uncertainties=False), separately per dimension, and
        subtract it from the right-column data and green wiggles. the left column gets
        its own fit, to all the noiseless stars. display only: the fits and z are
        unchanged. meant for un-straightened streams, where the progenitor track swamps
        the y-range.
    prog_subtracted: whether ssc is already straightened (progenitor track subtracted).
        True with subtract_poly raises a ValueError. None skips the check, so pass it
        whenever you know it.
    margs, prof, w_meds, peaks: precomputed marginal_A list, profile_best dict, marginal
        median w per dimension, and find_z_peaks output. anything None is computed here
        (with w_prior).
    precision_limits: as passed to gridsearch (used for obs_limit and fit_baseline).
    ghosts: optional ghost_flags output for these peaks. flagged peaks are drawn in
        light coral (dots and wiggles) instead of green, or left out entirely if
        show_ghosts=False. None flags nothing.
    verbose: print, per dimension, the number of l with |z| >= z_thresh and each peak's
        l, z, A (5-95%) and w (and its ghost_flags class, if ghosts is given).

    returns (fig, axs, out), where out is a dict with 'margs', 'prof', 'w_meds', 'peaks',
    'ghosts' and 'poly' (per dimension, the fitted numpy Polynomial, or None).
    """
    if subtract_poly and prog_subtracted:
        raise ValueError("subtract_poly=True, but the stream is already straightened "
                         "(prog_subtracted=True). set subtract_poly=False, or use the "
                         "un-straightened stream.")
    unc = uncertainties
    nk = len(GRID_KEYS)
    if margs is None:
        margs = [marginal_A(res, grid_dict, k, w_prior=w_prior) for k in range(nk)]
    if prof is None:
        prof = profile_best(res, grid_dict)
    if w_meds is None:
        w_meds = [marginal_quantiles(res, grid_dict, k, keep='w', q=(0.5,), w_prior=w_prior)[0][:, 0]
                  for k in range(nk)]
    if peaks is None:
        peaks = find_z_peaks(margs, z_thresh=z_thresh)
    if ghosts is None:
        flags = [np.zeros(len(pk), dtype=bool) for pk in peaks]
    else:
        if any(not np.array_equal(g['il'], pk) for g, pk in zip(ghosts, peaks)):
            raise ValueError("ghosts doesn't match peaks: run ghost_flags on the same peaks")
        flags = [g['flag'] for g in ghosts]
    GHOST_C = 'lightcoral'

    l_grid = grid_dict['l']
    window_width = res['window_width']
    il_true = None if phi1_true is None else np.argmin(np.abs(l_grid - phi1_true))
    x_data = np.asarray(ssc['phi1_noised'] if unc else ssc['phi1'], dtype=float)
    x_true = np.asarray(ssc['phi1'], dtype=float)
    polys = []

    fig, axs = plt.subplots(nk, 3, figsize=[18, 3*nk], sharex=True)
    for k, key in enumerate(GRID_KEYS):
        mg = margs[k]
        ipk, flag = np.asarray(peaks[k], dtype=int), flags[k]

        # middle: detection statistic, with the peaks (ghost-flagged ones in light coral)
        ax = axs[k, 1]
        ax.plot(l_grid, mg['z'], c='k', lw=1)
        ax.scatter(l_grid[ipk[~flag]], mg['z'][ipk[~flag]], c='mediumseagreen', s=15, zorder=3)
        if show_ghosts:
            ax.scatter(l_grid[ipk[flag]], mg['z'][ipk[flag]], c=GHOST_C, s=15, zorder=3)
        for s in (1, -1):
            ax.axhline(s*z_thresh, c='tomato', lw=0.8, ls='--')
        ax.axhline(0, c='grey', lw=0.5)
        ax.set_ylabel(r'$A_{%s}/\sigma$' % SYMBOLS[key])

        # right: the stream the filter saw, with a wiggle at each peak
        ax = axs[k, 2]
        y_data = np.asarray(ssc[key+"_noised"] if unc else ssc[key], dtype=float)
        good = (obs_limit(ssc, key, None if precision_limits is None else precision_limits.get(key))
                if unc else np.ones(len(y_data), dtype=bool))
        if subtract_poly:
            fit_ok = good & np.isfinite(x_data) & np.isfinite(y_data)
            poly = np.polynomial.Polynomial.fit(x_data[fit_ok], y_data[fit_ok], poly_deg)
            polys.append(poly)
        else:
            poly = lambda xx: np.zeros_like(xx, dtype=float)
            polys.append(None)
        y_plot = y_data - poly(x_data)
        # stars dropped by the precision cut in this dimension, for reference
        ax.scatter(x_data[~good], y_plot[~good], c='lightgrey', s=.1, rasterized=True, zorder=0)
        ax.scatter(x_data[good], y_plot[good], c='k', s=1, rasterized=True)
        # the cut stars have big errors, so scale the axis to the ones that pass
        lo, hi = np.nanpercentile(y_plot[good], [0.5, 99.5])
        ax.set_ylim(lo - 0.1*(hi-lo), hi + 0.1*(hi-lo))
        for ip, is_ghost in zip(ipk, flag):
            if is_ghost and not show_ghosts:
                continue
            l_pk, A_pk, w_pk = l_grid[ip], mg['q'][ip, 1], w_meds[k][ip]
            # the best line given this wiggle, at the profile's best stream width
            m_pk, b_pk = fit_baseline(ssc, key, l_pk, w_pk, A_pk, prof['sig'][ip, k],
                                      window_width, uncertainties=unc,
                                      precision_limits=precision_limits)
            xx = np.linspace(l_pk - window_width, l_pk + window_width, 400)
            ax.plot(xx, generic_signal(xx, A_pk, w_pk, l_pk, m=m_pk, b=b_pk) - poly(xx),
                    c=GHOST_C if is_ghost else 'mediumseagreen', lw=2 if is_ghost else 3,
                    zorder=2 if is_ghost else 3)
        ax.set_ylabel(YLABELS[key])

        # left: the noiseless stream, every star. with subtract_poly it gets its own fit
        # (to all the noiseless stars), since the noised fit is poorly pinned at the ends
        ax = axs[k, 0]
        y_true = np.asarray(ssc[key], dtype=float)
        if subtract_poly:
            ok = np.isfinite(x_true) & np.isfinite(y_true)
            poly_true = np.polynomial.Polynomial.fit(x_true[ok], y_true[ok], poly_deg)
        else:
            poly_true = poly
        y_true_plot = y_true - poly_true(x_true)
        ax.scatter(x_true, y_true_plot, c='k', s=1, rasterized=True)
        lo, hi = np.nanpercentile(y_true_plot, [0.5, 99.5])
        ax.set_ylim(lo - 0.1*(hi-lo), hi + 0.1*(hi-lo))
        ax.set_ylabel(YLABELS[key])

        if phi1_true is not None:
            for ax in axs[k]:
                ax.axvline(phi1_true, c='g', lw=1, zorder=0)

        if verbose:
            n_det = np.sum(np.abs(mg['z']) >= z_thresh)
            line = "%-8s |z|>=%g at %i of %i l" % (key, z_thresh, n_det, len(l_grid))
            if il_true is not None:
                line += (", z at truth = %+.2f, between/within var at truth = %.2f"
                         % (mg['z'][il_true], mg['var_between'][il_true] / mg['var_within'][il_true]))
            print(line)
            for i, ip in enumerate(ipk):
                print("         peak at l = %6.1f: z = %+6.2f, A = %+.4g (5-95%%: %+.4g, %+.4g), w = %.2f%s"
                      % (l_grid[ip], mg['z'][ip], mg['q'][ip, 1], mg['q'][ip, 0], mg['q'][ip, 2], w_meds[k][ip],
                         "" if ghosts is None else ", " + ghosts[k]['class'][i]))

    poly_note = (" − degree-%i polynomial in $\\phi_1$" % poly_deg) if subtract_poly else ""
    axs[0, 0].set_title("noiseless, all stars" + poly_note, fontsize=10)
    ghost_note = ("" if ghosts is None else
                  ", ghost-flagged in light coral" if show_ghosts else ", ghost-flagged hidden")
    axs[0, 1].set_title(r"$|A/\sigma| \geq %g$ in green" % z_thresh + ghost_note, fontsize=10)
    axs[0, 2].set_title(("noised, passing precision cuts" if unc else "noiseless, all stars")
                        + poly_note, fontsize=10)
    axs[-1, 0].set_xlabel(r'$\phi_1$ [deg]')
    axs[-1, 1].set_xlabel(r'$l$ [deg]')
    axs[-1, 2].set_xlabel(r'$\phi_1$ [deg]')
    fig.tight_layout()
    return fig, axs, {'margs': margs, 'prof': prof, 'w_meds': w_meds, 'peaks': peaks,
                      'ghosts': ghosts, 'poly': polys}


# %%
if __name__=="__main__": # scratch below basically. 

    # very exciting. next let's load some stream data. 
    orbit='gd1'
    pert_table = Table.read(repo_path+"/data/%s_impacts_with_amps.fits"%orbit, format="fits")


    ssc_table, phi1_loc = inputs.inputs_agama(orbit, pert_table) # default i=1636

    # the saved streams have the progenitor orbit subtracted. False adds it back, so the
    # filter runs on the observed-frame coordinates (phi1, and so phi1_loc, are unchanged).
    prog_subtracted = False
    if not prog_subtracted:
        prog_track = inputs.prog_track_observed(orbit)
        ssc_table = inputs.add_prog_track(ssc_table, orbit, track=prog_track)
    # %%

    # for ii in range(len(GRID_KEYS)):
    #     k = GRID_KEYS[ii]
    #     fig, ax = plt.subplots(figsize=[8,3])
    #     ax.scatter(ssc_table['phi1'], ssc_table[k], s=1, c='k')
    # # %%
    # ssc_poly = inputs.subtract_polynomial(ssc_table, n=5)
    # for ii in range(len(GRID_KEYS)):
    #     k = GRID_KEYS[ii]
    #     fig, ax = plt.subplots(figsize=[8,3])
    #     ax.scatter(ssc_poly['phi1'], ssc_poly[k], s=1, c='k')

    # ssc_table = ssc_poly.copy()
    # %%
    # xv_stream = tse.clean_stream(orbit='gd1', num_particles=10000)

    # coordinates in a fitted great-circle frame (no reflex correction), with the progenitor
    # orbit put through the same frame and subtracted inside inputs_galactocentric.
    # prog_row = tse.prog_tab[tse.prog_tab['name']==orbit]
    # prog_w = np.array([prog_row[c][0] for c in ('x', 'y', 'z', 'vx', 'vy', 'vz')])
    # ssc_table, fitted_frame, prog_track = inputs.inputs_galactocentric(xv_stream, prog_w=prog_w)
    # phi1_loc=0.

    # fig, axs = plt.subplots(4,1,figsize=[8,12])

    # for ii, ax in enumerate(axs):
    #     ax.scatter(ssc_table['phi1'], ssc_table[GRID_KEYS[ii]], s=1, c='k')

    # %%
    #----------------------------------------------------------------------------------#
    #   and now define the grid to search
    loc_grid = np.arange(-100, 10, 1.0)#0.5)
    n_points = 100
    w_grid = np.linspace(0.1, 5, n_points)

    # no amplitude grids: A is fit in closed form at every (l, w, sig)
    grid_dict = {
        "l": loc_grid,
        "w": w_grid,
    }


    # intrinsic stream widths are free too: flat grids from 0.2x to 3x the hand-set
    # values above (1x is a grid point). with the closed form, the sig axis is one more
    # column in a matrix product, so it's nearly free.
    n_sig = 100
    for key in GRID_KEYS:
        grid_dict['sig_'+key] = np.linspace(0.2, 3, n_sig) * streamwidths_test[key]
    # %%
    unc=True
    # unc=False
    filter_window_factor = 3. # filter window half-width = filter_window_factor*max(w grid)
    res = gridsearch(grid_dict, ssc_table,
                    uncertainties=unc, n_proc=1,
                    filter_window_factor=filter_window_factor)
    # res['A_hat'], res['sig_A'], res['lnL_max'], res['lnL_null']: each (n_l, n_w, n_sig, 4)

    phi1_true = np.ravel(phi1_loc)[0]
    il_true = np.argmin(np.abs(grid_dict['l'] - phi1_true))
    # %%
    # check the closed form against the brute-force likelihood (nll_model_full) at a few
    # (w, sig) at the true location. the baseline is refit with fit_baseline (an
    # independent lstsq) at each A, so:
    #   at A = A_hat, fit_baseline should give (m_hat, b_hat)
    #   at A = A_hat + delta*sig_A, -nll should be lnL_max - delta^2/2 (lnL with m, b profiled)
    #   at A = 0, -nll should be lnL_null
    #   lnL_marg - lnL_max should be ln(2 pi) - ln(det F_nn)/2, with F_nn built directly
    # differences should be at the level of floating-point rounding on lnL ~ 1e3-1e4.
    window_width = res['window_width']
    l_chk = grid_dict['l'][il_true]
    max_diff = {'lnL': 0.0, 'lnL_null': 0.0, 'm': 0.0, 'b': 0.0, 'lnL_marg': 0.0}
    for iw in [0, n_points//2, n_points-1]:
        w_chk = grid_dict['w'][iw]
        for isig in [0, n_sig//2, n_sig-1]:
            sw = {key: _sig_grid(grid_dict, key)[isig] for key in GRID_KEYS}
            Ah = res['A_hat'][il_true, iw, isig]
            sA = res['sig_A'][il_true, iw, isig]
            for delta in [-1.0, 0.0, 2.5, None]:            # None = A = 0, the null
                A = np.zeros(len(GRID_KEYS)) if delta is None else Ah + delta*sA
                bls = {key: fit_baseline(ssc_table, key, l_chk, w_chk, A[k], sw[key], window_width,
                                        uncertainties=unc) for k, key in enumerate(GRID_KEYS)}
                nlls, _ = nll_model_full(ssc_table, np.array([l_chk, w_chk, *A]),
                                        streamwidths=sw, uncertainties=unc,
                                        window_width=window_width, baselines=bls)
                if delta is None:
                    max_diff['lnL_null'] = np.nanmax([max_diff['lnL_null'],
                                                    *np.abs(-nlls - res['lnL_null'][il_true, iw, isig])])
                    continue
                lnL_cf = res['lnL_max'][il_true, iw, isig] - 0.5*delta**2
                max_diff['lnL'] = np.nanmax([max_diff['lnL'], *np.abs(-nlls - lnL_cf)])
                if delta == 0.0:
                    m_bl, b_bl = np.array([bls[key] for key in GRID_KEYS]).T
                    max_diff['m'] = np.nanmax([max_diff['m'], *np.abs(m_bl - res['m_hat'][il_true, iw, isig])])
                    max_diff['b'] = np.nanmax([max_diff['b'], *np.abs(b_bl - res['b_hat'][il_true, iw, isig])])
            for k, key in enumerate(GRID_KEYS):
                xw, _, errs = _window_data(ssc_table, key, l_chk, window_width, uncertainties=unc)
                iv = 1/(sw[key]**2 + errs**2)
                Xn = np.column_stack([xw - l_chk, np.ones(len(xw))])
                _, lndet = np.linalg.slogdet(Xn.T @ (Xn * iv[:, None]))
                marg_cf = res['lnL_marg'][il_true, iw, isig, k] - res['lnL_max'][il_true, iw, isig, k]
                max_diff['lnL_marg'] = np.nanmax([max_diff['lnL_marg'], abs(marg_cf - (np.log(2*np.pi) - 0.5*lndet))])
    print("closed form vs brute force, max |difference|: " + ", ".join("%s %.2e" % kv for kv in max_diff.items()))
    # %%
    # profile over (w, sig), and the marginal p(A | l) with its detection statistic z(l)
    prof = profile_best(res, grid_dict)
    margs = [marginal_A(res, grid_dict, k) for k in tqdm(range(len(GRID_KEYS)))]
    # median of the marginal p(w | l), A and sig marginalised out. grid-limited (see marginal_quantiles)
    w_meds = [marginal_quantiles(res, grid_dict, k, keep='w', q=(0.5,))[0][:, 0] for k in range(len(GRID_KEYS))]

    # detections: local maxima of |z(l)| that reach z_thresh
    z_thresh = 5
    peaks = find_z_peaks(margs, z_thresh=z_thresh)
    # %%
    # the diagnostic figure. subtract_poly subtracts a degree-poly_deg polynomial (fit to the
    # stars passing the precision cut) from the stream column, display only; it raises if
    # the stream is already straightened.
    fig, axs, _ = plot_detections(res, grid_dict, ssc_table, uncertainties=unc,
                                  z_thresh=z_thresh, phi1_true=phi1_true,
                                  subtract_poly=True, poly_deg=5,
                                  prog_subtracted=prog_subtracted,
                                  margs=margs, prof=prof, w_meds=w_meds, peaks=peaks)

    # used by the CLEAN cells below
    ylabels = YLABELS
    x_data = np.asarray(ssc_table['phi1_noised'] if unc else ssc_table['phi1'])
    # %%
    # 
    #
    #   CLEAN STUFF; mixed results tbh. 
    #
    #
    #
    # iterative subtraction (CLEAN): in each dimension, find the highest |z| peak, subtract
    # its wiggle from the whole stream, refit all of that dimension's wiggles jointly (incl.
    # their l, w, sig), and repeat until no |z| reaches z_thresh. ghost peaks only exist
    # because of a stronger wiggle nearby, so they should vanish once it's subtracted.
    cl = clean(ssc_table, grid_dict, z_thresh=z_thresh, uncertainties=unc,
            filter_window_factor=filter_window_factor)

    for k, key in enumerate(GRID_KEYS):
        dets = cl['detections'][k]
        print("%-8s %i wiggle(s) after CLEAN (peaks before: %i). z at truth %+.2f -> %+.2f, max |z| left %.2f"
            % (key, len(dets), len(peaks[k]), margs[k]['z'][il_true], cl['margs'][k]['z'][il_true],
                np.nanmax(np.abs(cl['margs'][k]['z']))))
        for d in dets:
            print("         round %i: found at l = %6.1f (z = %+6.2f, w = %.2f), refit l = %6.1f, w = %.2f, "
                "A = %+.4g +- %.2g (z_refit = %+.1f), dl_prev = %s"
                % (d['round'], d['l_found'], d['z'], d['w_found'], d['l'], d['w'],
                    d['A'], d['sig_A'], d['z_refit'], "%.0f" % d['dl_prev'] if np.isfinite(d['dl_prev']) else '-'))
    # %%
    # before vs after CLEAN. left: z(l) on the data (grey) and on the final residual
    # (black), with each wiggle at the l and z where it was found. middle: the data, with
    # each CLEAN wiggle drawn on its own fitted line (fit with the other wiggles
    # subtracted, so where two overlap the curves are each other's residual baseline).
    # right: the residual, data minus every CLEAN wiggle.
    fig, axs = plt.subplots(4, 3, figsize=[18, 12], sharex=True)
    for k, key in enumerate(GRID_KEYS):
        dets = cl['detections'][k]
        dcol = key+"_noised" if unc else key
        y_data = np.asarray(ssc_table[dcol])
        y_res = np.asarray(cl['resid'][dcol])
        good = obs_limit(ssc_table, key) if unc else np.ones(len(y_data), dtype=bool)
        lo, hi = np.nanpercentile(y_data[good], [0.5, 99.5])
        ylim = (lo - 0.1*(hi-lo), hi + 0.1*(hi-lo))

        ax = axs[k, 0]
        ax.plot(grid_dict['l'], margs[k]['z'], c='grey', lw=1, label='data')
        ax.plot(grid_dict['l'], cl['margs'][k]['z'], c='k', lw=1, label='CLEAN residual')
        ax.scatter([d['l_found'] for d in dets], [d['z'] for d in dets], c='mediumseagreen', s=15, zorder=3)
        for s in (1, -1):
            ax.axhline(s*z_thresh, c='tomato', lw=0.8, ls='--')
        ax.axvline(phi1_true, c='g', lw=1)
        ax.set_ylabel(r'$z(l)$, ' + ylabels[key].split(' [')[0])

        ax = axs[k, 1]
        ax.scatter(x_data[~good], y_data[~good], c='lightgrey', s=.1, rasterized=True, zorder=0)
        ax.scatter(x_data[good], y_data[good], c='k', s=1, rasterized=True)
        for d in dets:
            xx = np.linspace(d['l'] - cl['window_width'], d['l'] + cl['window_width'], 400)
            ax.plot(xx, generic_signal(xx, d['A'], d['w'], d['l'], m=d['m'], b=d['b']),
                    c='mediumseagreen', lw=3)
        ax.axvline(phi1_true, c='g', lw=1, zorder=0)
        ax.set_ylim(ylim)
        ax.set_ylabel(ylabels[key] + (" (noised)" if unc else ""))

        ax = axs[k, 2]
        ax.scatter(x_data[~good], y_res[~good], c='lightgrey', s=.1, rasterized=True, zorder=0)
        ax.scatter(x_data[good], y_res[good], c='k', s=1, rasterized=True)
        ax.axvline(phi1_true, c='g', lw=1, zorder=0)
        ax.set_ylim(ylim)
        ax.set_ylabel('residual')
    axs[0, 0].legend(fontsize=8)
    axs[0, 1].set_title('data + CLEAN wiggles')
    axs[0, 2].set_title('data - CLEAN wiggles')
    for ax in axs[-1]:
        ax.set_xlabel(r'$\phi_1$ [deg]')
    axs[-1, 0].set_xlabel(r'$l$ [deg]')
    fig.tight_layout()
    # %%