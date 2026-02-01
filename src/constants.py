from src.parser import *
from src.folderconstants import *

# Threshold parameters
lm_d = {
		'SMD': [(0.99, 1)],
		'SWaT': [(0.95, 1)],
		'SMAP': [(0.923, 1)],
		'MSL': [(0.986, 1)],
		'MIT-BIH': [(0.913, 1)],
	}
lm = lm_d[args.dataset][0]

# Hyperparameters
lr_d = {
		'SMD': 0.0001, 
		'SWaT': 0.008, 
		'SMAP': 0.001, 
		'MSL': 0.002, 
		'MIT-BIH': 0.001, 
	}
lr = lr_d[args.dataset]

percentiles = {
		'SMD': (98, 2000),
		'SWaT': (95, 10),
		'SMAP': (97, 5000),
		'MSL': (97, 150),
		'MIT-BIH': (99, 2),
	}
percentile_merlin = percentiles[args.dataset][0]
cvp = percentiles[args.dataset][1]
preds = []
debug = 9