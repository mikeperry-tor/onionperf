#!/usr/bin/env python3

import os
import math
import sys
import random

import numpy

from scipy.stats import pareto
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

numpy.seterr('raise')

SUBSAMPLING_SIZES = [100, 500, 1000]
NUM_SUBSAMPLES = 1000

PLOT_MAX_MS = 5000

XM_NUM_MODES = 5
XM_MAX_MODE = True
XM_AVG_MODES = False
XM_AVG_BELOW = True
XM_ALL_MODES = True

def parse_state_line(line, full_hist):
    """
    Parse a line from a Tor state line and return the data that we should plot in the histogram

    For example if it's (CircuitBuildTimeBin 342 4) return (342, 342, 342, 342)
    """
    items = line.split()

    # We only use CircuitBuildTimeBin lines
    if len(items) < 1:
        return None
    if items[0] == "CircuitBuildTimeBin":
        value = int(items[1])
        occurences = int(items[2])
#    elif items[0] == "CircuitBuildAbandonedCount":
#        value = float("NaN")
#        occurences = int(items[1])
    else:
        return None

    full_hist[value] = occurences

    return ([value] * occurences)

def extract_data(state_fname, subsampling_n):
    # Extract data from state file
    data = []
    hist = {}
    for line in open(state_fname,'r'):
        values = parse_state_line(line, hist)
        if values:
            data.extend(values)

    # For each abandoned circuits turn it into a circuit that has reached the
    # maximum timeout
    for i, value in enumerate(data):
        if math.isnan(value):
            data[i] = max(data)

    if subsampling_n != 1000:
        data = random.sample(data, subsampling_n)

    return (data, hist)

def _get_hist(hist, idx):
  if not idx in hist:
    return 0
  return hist[idx]

def get_xm(data, hist):
  avg_modes=XM_AVG_MODES
  avg_below=XM_AVG_BELOW
  all_modes=XM_ALL_MODES
  num_modes=XM_NUM_MODES
  max_mode=XM_MAX_MODE
  if not all_modes and len(data) < 600: # XXX: hack for abandoned circs
    num_modes = 1

  # obtain the nth highest modes
  nth_max_bin = [0] * num_modes
  for i in hist.keys():
    if hist[i] >= _get_hist(hist, nth_max_bin[0]):
      nth_max_bin[0] = i

    for n in range(1, num_modes):
      if (hist[i] >= _get_hist(hist, nth_max_bin[n]) and
           (not _get_hist(hist, nth_max_bin[n-1])
               or hist[i] < _get_hist(hist, nth_max_bin[n-1]))):
        nth_max_bin[n] = i;

  Xm = 0

  # XXX: Most common mode not tested, but it is not good
  if max_mode:
    Xm = max(nth_max_bin)
    assert not avg_modes
  else:
    Xm = min(nth_max_bin)

  assert Xm

  if avg_modes:
    tot_xm = 0
    xm_cnt = 0

    for i in range(0, num_modes):
      tot_xm += nth_max_bin[i]*hist[nth_max_bin[i]]
      xm_cnt += hist[nth_max_bin[i]]

    Xm = tot_xm/xm_cnt

  if avg_below:
    tot_xm = 0
    xm_cnt = 0

    for x in data:
      if x <= Xm:
        tot_xm += x
        xm_cnt += 1

    Xm = tot_xm / xm_cnt

  return Xm

def get_alpha(data, Xm):
  alpha = 0.0
  n = len(data)

  for d in data:
    if d < Xm:
      alpha += math.log(Xm)
    else:
      alpha += math.log(d)

  alpha -= n*math.log(Xm)
  alpha = n/alpha

  return alpha

def pareto_pdf(x, Xm, alpha):
  if x < Xm:
    return 0.0
  return alpha*math.pow(Xm,alpha)/math.pow(x,alpha+1)

def pareto_quantile(q, Xm, alpha):
  return Xm/math.pow(1.0-q,1.0/alpha)

def get_timeout_rate(timeout_ms, data):
  timeouts = 0.0
  for d in data:
    if d < timeout_ms:
      timeouts += 1

  return timeouts/len(data)

def test_timeout_rate(data, test_data, hist):
  # Compute our own pareto
  Xm = get_xm(data, hist)
  alpha = get_alpha(data, Xm)

  timeout_60 = pareto_quantile(0.60, Xm, alpha)
  timeout_70 = pareto_quantile(0.70, Xm, alpha)
  timeout_80 = pareto_quantile(0.80, Xm, alpha)

  rate_60 = get_timeout_rate(timeout_60, test_data)
  rate_70 = get_timeout_rate(timeout_70, test_data)
  rate_80 = get_timeout_rate(timeout_80, test_data)

  return (rate_60, rate_70, rate_80)

def plot_state_file(state_fname, data, hist, ax, full_data):

    # Make the bins for the histogram
    bins = list(range(0, PLOT_MAX_MS, 10)) # bins should be every 10ms
    #bins.append(max(data)) # also include the last one

    # Plot the histogram
    ax.hist(data, bins=bins, density=True, facecolor='green', alpha=1)

    # Try to fit a Pareto distribution to the data
    shape, loc, scale = pareto.fit(data, 1, loc=0, scale=1)
    y = pareto.pdf(bins, shape, loc=loc, scale=scale)
    # Plot the pareto
    l = ax.plot(bins, y, 'r--', linewidth=2)

    # Compute our own pareto
    Xm = get_xm(data, hist)
    alpha = get_alpha(data, Xm)
    l = ax.plot(bins, list(map(lambda x: pareto_pdf(x, Xm, alpha), bins)), 'g--', linewidth=2)

    # Set up the graph metadata
    plt.xticks((0,500, 1000, 1500) + tuple(range(2500, PLOT_MAX_MS, 5000)), rotation=90)
    ax.grid(alpha=0.3)
    ax.set_xlabel('Miliseconds')
    ax.set_ylabel('Probability')

    basename=os.path.splitext(os.path.basename(state_fname))[0]
    ax.set_title("%s [%d timeouts:" %(basename, len(data)))

    ax.grid(True)

    # Plot it! :)
    #plt.show()
    #basename=os.path.splitext(sys.argv[1])[0]
    #plt.savefig(basename + "_pareto.png", dpi=300)

def subsample_test(state_fnames_list):
    print("\nTimeout test with XM_NUM_MODES=%d; XM_MAX_MODE=%d XM_AVG_MODES=%d; XM_AVG_BELOW=%d; XM_ALL_MODES=%d" %
          (XM_NUM_MODES, XM_MAX_MODE, XM_AVG_MODES, XM_AVG_BELOW, XM_ALL_MODES))

    for i, state_fname in enumerate(state_fnames_list):
        (full_data,full_hist) = extract_data(state_fname, 1000)
        basename=os.path.splitext(os.path.basename(state_fname))[0]
        for s, subsampling_n in enumerate(SUBSAMPLING_SIZES):
          rates_60 = []
          rates_70 = []
          rates_80 = []
          for n in range(0,NUM_SUBSAMPLES):
            (data,hist) = extract_data(state_fname, subsampling_n)
            (rate_60, rate_70, rate_80) = test_timeout_rate(data, full_data, hist)
            rates_60.append(math.fabs(rate_60-0.6))
            rates_70.append(math.fabs(rate_70-0.7))
            rates_80.append(math.fabs(rate_80-0.8))
          print("%s-%d 60 avg/dev error: %f/%f" %
               (basename, subsampling_n,
                numpy.mean(rates_60), numpy.std(rates_60)))
          print("%s-%d 70 avg/dev error: %f/%f" %
               (basename, subsampling_n,
                numpy.mean(rates_70), numpy.std(rates_70)))
          print("%s-%d 80 avg/dev error: %f/%f" %
               (basename, subsampling_n,
                numpy.mean(rates_80), numpy.std(rates_80)))

def plot_all(state_fnames_list):
    n_states = len(state_fnames_list)
    fig, (ax_list) = plt.subplots(nrows=len(SUBSAMPLING_SIZES), ncols=n_states)

    for i, state_fname in enumerate(state_fnames_list):
        for s, subsampling_n in enumerate(SUBSAMPLING_SIZES):
            (data,hist) = extract_data(state_fname, subsampling_n)
            (full_data,full_hist) = extract_data(state_fname, 1000)
            plot_state_file(state_fname, data, hist, ax_list[s][i], full_data)


    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("Usage:\n $ ./analysis_state_timeout.py state-1.txt state-2.txt state-3.txt")
        sys.exit(1)

    state_fnames_list = sys.argv[1:]

    global XM_NUM_MODES, XM_MAX_MODE, XM_AVG_MODES, XM_AVG_BELOW, XM_ALL_MODES

    (XM_NUM_MODES, XM_MAX_MODE, XM_AVG_MODES, XM_AVG_BELOW, XM_ALL_MODES) = \
               (3,       False,        True,        False,         False)
    subsample_test(state_fnames_list)

    (XM_NUM_MODES, XM_MAX_MODE, XM_AVG_MODES, XM_AVG_BELOW, XM_ALL_MODES) = \
               (3,       False,         True,         True,         True)
    subsample_test(state_fnames_list)

    (XM_NUM_MODES, XM_MAX_MODE, XM_AVG_MODES, XM_AVG_BELOW, XM_ALL_MODES) = \
               (3,       False,        False,         True,         True)
    subsample_test(state_fnames_list)

    (XM_NUM_MODES, XM_MAX_MODE, XM_AVG_MODES, XM_AVG_BELOW, XM_ALL_MODES) = \
               (3,       True,         False,         True,         True)
    subsample_test(state_fnames_list)

    (XM_NUM_MODES, XM_MAX_MODE, XM_AVG_MODES, XM_AVG_BELOW, XM_ALL_MODES) = \
               (5,       True,         False,         True,         True)
    subsample_test(state_fnames_list)

    (XM_NUM_MODES, XM_MAX_MODE, XM_AVG_MODES, XM_AVG_BELOW, XM_ALL_MODES) = \
               (7,      True,         False,         True,         True)
    subsample_test(state_fnames_list)

    (XM_NUM_MODES, XM_MAX_MODE, XM_AVG_MODES, XM_AVG_BELOW, XM_ALL_MODES) = \
               (10,      True,         False,         True,         True)
    subsample_test(state_fnames_list)

    (XM_NUM_MODES, XM_MAX_MODE, XM_AVG_MODES, XM_AVG_BELOW, XM_ALL_MODES) = \
               (15,      True,         False,         True,         True)
    subsample_test(state_fnames_list)

    (XM_NUM_MODES, XM_MAX_MODE, XM_AVG_MODES, XM_AVG_BELOW, XM_ALL_MODES) = \
               (20,      True,         False,         True,         True)
    subsample_test(state_fnames_list)

    plot_all(state_fnames_list)

if __name__ == '__main__':
    main()
