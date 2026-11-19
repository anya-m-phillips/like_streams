"""
one place for every machine-specific path.

works out whether we're on cannon or my laptop, sets the paths to match, and puts
the extra package dirs on sys.path. every script does `from paths import ...`
instead of hard-coding its own paths.

machine detection (first match wins):
  1. env var LIKE_STREAMS_MACHINE = 'cannon' or 'local'  (force it either way)
  2. cannon if /n/home02/amphillips exists, otherwise local

to add a machine or change a path, edit MACHINES below.
"""
import os
import sys

# repo root = the folder above scripts/. found from this file's location, so it's
# right on any machine without hard-coding it.
SCRIPTS_PATH = os.path.dirname(os.path.abspath(__file__))
repo_path = os.path.dirname(SCRIPTS_PATH)

MACHINES = {
    'cannon': {
        'viamock': '/n/home02/amphillips/software/viamock',
        'site_packages': '/n/home02/amphillips/.conda/envs/petar_env/lib/python3.10/site-packages',
        'scratch': '/n/netscratch/conroy_lab/Lab/amphillips/tsetse_analysis/',
    },
    'local': {
        'viamock': '/Users/anyaphillips/Downloads/software/viamock',
        'site_packages': '/opt/anaconda3/envs/viamock/lib/python3.10/site-packages',
        # no local copy of the netscratch outputs yet. point this at one if you make it.
        'scratch': '/n/netscratch/conroy_lab/Lab/amphillips/tsetse_analysis/',
    },
}


def _detect_machine():
    forced = os.environ.get('LIKE_STREAMS_MACHINE')
    if forced:
        if forced not in MACHINES:
            raise ValueError('LIKE_STREAMS_MACHINE=%r, expected one of %s' % (forced, list(MACHINES)))
        return forced
    if os.path.isdir('/n/home02/amphillips'):
        return 'cannon'
    return 'local'


MACHINE = _detect_machine()
_p = MACHINES[MACHINE]

VIAMOCK_PATH = _p['viamock']
SITE_PACKAGES_PATH = _p['site_packages']  # extra env with artpop etc.
SCRATCH_PATH = _p['scratch']
PERTURBED_STREAMS_DIR = os.path.join(SCRATCH_PATH, 'perturbed_ps_streams') + '/'
DISSOLVED_STREAMS_DIR = os.path.join(SCRATCH_PATH, 'perturbed_ps_streams_dissolved') + '/'
MPLSTYLE = os.path.join(SCRIPTS_PATH, 'vedant.mplstyle')


def _add_path(p, front=False):
    if p and p not in sys.path:
        if front:
            sys.path.insert(0, p)
        else:
            sys.path.append(p)


# scripts/ and the repo root go first so local modules win; the extra package dirs
# go at the end so the active env's own packages win.
_add_path(SCRIPTS_PATH, front=True)
_add_path(repo_path)
_add_path(VIAMOCK_PATH)
_add_path(SITE_PACKAGES_PATH)
