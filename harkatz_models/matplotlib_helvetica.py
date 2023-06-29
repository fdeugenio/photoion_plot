# coding: utf-8
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = (r'\usepackage{tgheros}'
#r'\usepackage{sansmath}'
#r'\sansmath'
r'\usepackage{upgreek}'
r'\usepackage{textgreek}'
)
# 14 from mpl_toolkits.axes_grid1 import make_axes_locatable
# 15 from matplotlib.ticker import MaxNLocator
#matplotlib.rcParams['mathtext.rm'] = 'serif'
matplotlib.rcParams['xtick.labelsize'] = 18
matplotlib.rcParams['ytick.labelsize'] = 18
matplotlib.rcParams['xtick.minor.visible'] = True
matplotlib.rcParams['ytick.minor.visible'] = True
matplotlib.rcParams['xtick.major.width'] = 1.2
matplotlib.rcParams['xtick.minor.width'] = 1.
matplotlib.rcParams['xtick.major.size'] = 5.
matplotlib.rcParams['xtick.minor.size'] = 3.
matplotlib.rcParams['ytick.major.width'] = 1.2
matplotlib.rcParams['ytick.minor.width'] = 1.
matplotlib.rcParams['ytick.major.size'] = 6.
matplotlib.rcParams['ytick.minor.size'] = 2.5
matplotlib.rcParams['axes.linewidth'] = 1.2
matplotlib.rcParams['xtick.direction'] = u'in'
matplotlib.rcParams['ytick.direction'] = u'in'
matplotlib.rcParams['xtick.top'] = True
matplotlib.rcParams['ytick.right'] = True
#plt.rcParams['ps.useafm'] = True
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rcParams['pdf.fonttype'] = 42
