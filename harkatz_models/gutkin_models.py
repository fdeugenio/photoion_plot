import glob
import os
import re

import numpy as np
from astropy import table

from .ratio_table import ratio_table

__all__ = ("read_gutkin_models",)



def read_gutkin_models(input_filename=None,
    param1='OIII5007/Hbeta', # [OIII]5007/H\beta
    param2='NII6584/Halpha', # [NII]6584/H\alpha
    downsample={},
    print_ranges=True,
    ):
    """
    Parameters
    ----------
    param1: str, ratio of two column names
        E.g. logU/logn, but ideally two emission-line fluxes
    param2: str, ratio of two column names
        E.g. logU/logn, but ideally two emission-line fluxes
    downsample: dict of {'colname': value or range,}
        E.g. downsample={'logU': (-4, -3.5, -3), 'lnh': 1} will return the
        grid only for these values (removing logU>-3, lnh>1).
    print_ranges: bool
        If set, print the range in all the physical parameters.

    Return
    ------

    grid : nd float array with the model values.
        The grid has shape
        (n1, n2, n3, n4, n5, n6, 8), where
        n1 - number of unique `Mup` values (0, 1) [0: single, 1: binary]
        n2 - number of unique `C` values (3, 5) [Myr]
        n3 - number of unique `xi` values (3, 5) [Myr]
        n4 - number of unique `Z` values (0.001, 0.002, ..., 0.04) [mass fraction TBC]
        n5 - number of unique `logn` values (1, 1.5, ..., 3) [dex cm^{-3}]
        n6 - number of unique `logU` values (-4, -3.5, ..., -0.5) [dex]
        8  - number of physical parameters (6) + number of emission-line ratios (2)

    Examples
    --------

    Grid with [OIII]5007/H\beta and [NII]6584/H\alpha as a function of all parameters.
    >>> grid0 = read_gutkin_models()

    For logU=-4, logn=1, Z=0.001, ssp=0, age=5:
    [OIII]5007/H\beta is grid0[0, 0, 0, 0, 0, 5]
    [NII]6584/H\alpha is grid0[0, 0, 0, 0, 0, 6]
    ssp  can be read from grid0[0, 0, 0, 0, 0, 0]
    age  can be read from grid0[0, 0, 0, 0, 0, 1]
    ...
    logU can be read from grid0[0, 0, 0, 0, 0, 4]

    To see all values of Mup in the grid
    >>> print(grid0[:, 0, 0, 0, 0, 0, 0])

    To see all values of C in the grid
    >>> print(grid0[0, :, 0, 0, 0, 0, 1])

    To see all values of xi in the grid
    >>> print(grid0[0, 0, :, 0, 0, 0, 2])

    To see all values of Z in the grid
    >>> print(grid0[0, 0, 0, :, 0, 0, 3])

    To see all values of logn in the grid
    >>> print(grid0[0, 0, 0, 0, :, 0, 4])

    To see all values of logU in the grid
    >>> print(grid0[0, 0, 0, 0, 0, :, 5])
    
    Grid with [NeIII]/[OII] and [OIII]5007/H\beta, but consider only logU>-3, age=5 and logn=2.
    >>> grid1 = read_gutkin_models(
    >>>     param1="OIII5007/Hbeta",
    >>>     param2="NeIII3869/OII3727",
    >>>     downsample={'logU': (-2.5, -2., -1.5, -1, -0.5), 'Mup': 100, 'logn': 2, 'C': 100})
    
    """

    if input_filename is None:
        data_path = os.path.dirname(__file__)
        input_filenames = glob.glob(os.path.join(
            data_path, 'data', 'gutkin_ne3he2',
            'SF_line_info*.dat'))

    gutkintab = [ratio_table.read(
        input_filename,format='ascii.commented_header')
        for input_filename in input_filenames]
    for tab,filename in zip(gutkintab, input_filenames):
        carbon_str = re.search(r'C[0-9]{1,}_', filename)
        assert carbon_str, f'Did not find C in string {filename}'
        tab['C'] = float(carbon_str.group()[1:-1])
    for tab,filename in zip(gutkintab, input_filenames):
        imf_mup = re.search(r'mup(1|3)00\.', filename)
        assert imf_mup, f'Did not find mup in string {filename}'
        tab['Mup'] = float(imf_mup.group()[3:-1])
    gutkintab = table.vstack(gutkintab)
            
    # Need a parser to do this operation for me.
    # gutkintab['O  2 3727A'] = (
    #    gutkintab['O  2 3726.03A'] + gutkintab['O  2 3728.81A']) # Add doublet...

    # Add emission-line ratios as columns. Little detour from the usual path...
    gutkintab['temp_ratio_1'] = gutkintab[param1]
    gutkintab['temp_ratio_2'] = gutkintab[param2]
    gutkintab.__class__ = table.table.Table 
    gutkintab['temp_ratio_1'].name = param1
    gutkintab['temp_ratio_2'].name = param2 # ...end detour. Now normal table.

    # Change silly names and values.
    gutkintab['lnh'].name = 'logn'

    # These are the physical parameters of the grid, e.g. Z, logU, etc.
    phypar = ('logU', 'logn', 'Z', 'xi', 'Mup', 'C')
    # Because of how the table is written, this poor programmer has to
    # transpose everything.
    phypar = ('Mup', 'C', 'xi', 'Z', 'logn', 'logU')

    if print_ranges:
        print('Grid parameters (*before* downsample keyword)')
        for pp in phypar:
            print(f'{pp: <10s} {sorted(set(gutkintab[pp]))}')

    # Here down-sample the grid according to the user's requests.
    select_table = []
    for i,(label,value) in enumerate(downsample.items()):
        if value is not None:
            value = np.atleast_1d(value)
            _selection_ = np.any([gutkintab[label]==v for v in value], axis=0)
            assert any(_selection_), f'Label {label}=={value} did not match any row!'
            select_table.append(_selection_)
    if select_table: # Check not empty.
        select_table = np.all(select_table, axis=0)
        gutkintab = gutkintab[select_table]

    if print_ranges:
        print('Grid parameters (*after* downsample keyword)')
        for pp in phypar:
            print(f'{pp: <10s} {sorted(set(gutkintab[pp]))}')


    # Because of how the table is written, this poor programmer has to
    # transpose everything.
    phypar = ('Mup', 'C', 'xi', 'Z', 'logn', 'logU')
    gutkintab.sort(keys=phypar)

    # Prepare the output grid.
    grid_shape = tuple()
    for p in phypar: # Grab how many samples per physical parameter.
        grid_shape = grid_shape + (len(set(gutkintab[p])),)
    grid = np.zeros(grid_shape + (len(phypar) + 2,)) # Add 2 for the emline ratios.

    for i,colname in enumerate(phypar + (param1, param2)):
        grid[:, :, :, :, :, :, i] = np.reshape(gutkintab[colname], grid.shape[:-1])

    return phypar, grid
