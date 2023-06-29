import os
import numpy as np
from astropy import table

__all__ = ("read_harkatz_models",)

# Ratty tables to calculate ratios on the fly.
class ratio_table(table.Table):
    """Silly class inheriting from `astropy.table.Table`. Issue is that inheriting
    from that class is a bit sticky, so I inherit, run the bit I need, then
    forego the inheritance (see `harkatztab.__class__ = table.table.Table` later on.)
    If you are reading this you may need a doctor.
    """
    def __getitem__(self, key):
        if type(key) is not str: # Remember how to slice tables
            return super(ratio_table, self).__getitem__(key)
        if key in self.colnames: # Return simple columns.
            return super(ratio_table, self).__getitem__(key)
        if '/' in key:
            return self.__getitem_div__(key)
        if '+' in key:
            return self.__getitem_plus__(key)

    def __getitem_plus__(self, key):
        try:
            keyadd1, keyadd2= key.split('+')
        except ValueError:
            message = f'Ambiguous {key}; column sums must have only one `+`'
            raise KeyError(message)
        else:
            return (
                super(ratio_table, self).__getitem__(keyadd1)
                + super(ratio_table, self).__getitem__(keyadd2)
            )

    def __getitem_div__(self, key):
        try:
            keynum, keyden = key.split('/')
        except ValueError:
            message = f'Ambiguous {key}; column ratios must have only one `/`'
            raise KeyError(message)
        else:
            return (
                super(ratio_table, self).__getitem__(keynum)
                / super(ratio_table, self).__getitem__(keyden)
            )



def read_harkatz_models(input_filename=None,
    param1='O  3 5006.84A/H  1 4861.32A', # [OIII]5007/H\beta
    param2='N  2 6583.45A/H  1 6562.80A', # [NII]6584/H\alpha
    downsample={},
    print_ranges=True,
    ):
    """
    Parameters
    ----------
    param1: str, ratio of two column names
        E.g. U/rho, but ideally two emission-line fluxes
    param2: str, ratio of two column names
        E.g. U/rho, but ideally two emission-line fluxes
    downsample: dict of {'colname': value or range,}
        E.g. downsample={'logU': (-4, -3.5, -3), 'rho': 1} will return the
        grid only for these values (removing logU>-3, rho>1).
    print_ranges: bool
        If set, print the range in all the physical parameters.

    Return
    ------

    grid : nd float array with the model values.
        The grid has shape
        (n1, n2, n3, n4, n5, 7), where
        n1 - number of unique `ssp type` values (0, 1) [0: single, 1: binary]
        n2 - number of unique `age` values (3, 5) [Myr]
        n3 - number of unique `metal` values (0.001, 0.002, ..., 0.04) [mass fraction TBC]
        n4 - number of unique `logn` values (1, 1.5, ..., 3) [dex cm^{-3}]
        n5 - number of unique `logU` values (-4, -3.5, ..., -0.5) [dex]
        7  - number of physical parameters (5) + number of emission-line ratios (2)

    Examples
    --------

    Grid with [OIII]5007/H\beta and [NII]6584/H\alpha as a function of all parameters.
    >>> grid0 = read_harkatz_models()

    For logU=-4, logn=1, Z=0.001, ssp=0, age=5:
    [OIII]5007/H\beta is grid0[0, 0, 0, 0, 0, 5]
    [NII]6584/H\alpha is grid0[0, 0, 0, 0, 0, 6]
    ssp  can be read from grid0[0, 0, 0, 0, 0, 0]
    age  can be read from grid0[0, 0, 0, 0, 0, 1]
    ...
    logU can be read from grid0[0, 0, 0, 0, 0, 4]

    To see all values of age in the grid
    >>> print(grid0[:, 0, 0, 0, 0, 0])

    To see all values of age in the grid
    >>> print(grid0[0, :, 0, 0, 0, 1])

    To see all values of metal in the grid
    >>> print(grid0[0, 0, :, 0, 0, 2])

    To see all values of logn in the grid
    >>> print(grid0[0, 0, 0, :, 0, 3])

    To see all values of logU in the grid
    >>> print(grid0[0, 0, 0, 0, :, 4])
    
    Grid with [NeIII]/[OII] and [OIII]5007/H\beta, but consider only logU>-3, age=5 and logn=2.
    >>> grid1 = read_harkatz_models(
    >>>     param1="O  3 5006.84A/H  1 4861.32A",
    >>>     param2="Ne 3 3868.76A/O  2 3727A",
    >>>     downsample={'logU': (-2.5, -2., -1.5, -1, -0.5), 'age': 5, 'logn': 2})
    
    """

    if input_filename is None:
        data_path = os.path.dirname(__file__)
        input_filename = os.path.join(data_path, 'data', 'emission_lines.csv')

    harkatztab = ratio_table.read(input_filename)
    # Need a parser to do this operation for me.
    harkatztab['O  2 3727A'] = (
        harkatztab['O  2 3726.03A'] + harkatztab['O  2 3728.81A']) # Add doublet...

    # Add emission-line ratios as columns. Little detour from the usual path...
    harkatztab['temp_ratio_1'] = harkatztab[param1]
    harkatztab['temp_ratio_2'] = harkatztab[param2]
    harkatztab.__class__ = table.table.Table 
    harkatztab['temp_ratio_1'].name = param1
    harkatztab['temp_ratio_2'].name = param2 # ...end detour. Now normal table.

    # Change silly names and values.
    harkatztab['U'].name = 'logU'
    harkatztab['rho'].name = 'logn'
    harkatztab['ssp type'] = [
        int(x=='binary') for x in harkatztab['ssp type']]

    # These are the physical parameters of the grid, e.g. Z, logU, etc.
    phypar = ('logU', 'logn', 'metal', 'ssp type', 'age')
    # Because of how the table is written, this poor programmer has to
    # transpose everything.
    phypar = ('ssp type', 'age', 'metal', 'logn', 'logU')

    if print_ranges:
        print('Grid parameters (*before* downsample keyword)')
        for pp in phypar:
            print(f'{pp: <10s} {sorted(set(harkatztab[pp]))}')

    # Here down-sample the grid according to the user's requests.
    select_table = []
    for i,(label,value) in enumerate(downsample.items()):
        if value is not None:
            value = np.atleast_1d(value)
            _selection_ = np.any([harkatztab[label]==v for v in value], axis=0)
            assert any(_selection_), f'Label {label}=={value} did not match any row!'
            select_table.append(_selection_)
    if select_table: # Check not empty.
        select_table = np.all(select_table, axis=0)
        harkatztab = harkatztab[select_table]

    if print_ranges:
        print('Grid parameters (*after* downsample keyword)')
        for pp in phypar:
            print(f'{pp: <10s} {sorted(set(harkatztab[pp]))}')


    # Because of how the table is written, this poor programmer has to
    # transpose everything.
    phypar = ('ssp type', 'age', 'metal', 'logn', 'logU')

    # Prepare the output grid.
    grid_shape = tuple()
    for p in phypar: # Grab how many samples per physical parameter.
        grid_shape = grid_shape + (len(set(harkatztab[p])),)
    grid = np.zeros(grid_shape + (len(phypar) + 2,)) # Add 2 for the emline ratios.

    for i,colname in enumerate(phypar + (param1, param2)):
        grid[:, :, :, :, :, i] = np.reshape(harkatztab[colname], grid.shape[:-1])
    
    return grid
