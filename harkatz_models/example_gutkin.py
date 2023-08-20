import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scipy.interpolate import interp1d

from .gutkin_models import read_gutkin_models
from .plotting_utils import colorline



def example_plot_ssp_metal_logu_interp_roberta(
    cmap='viridis', alpha=0.5,
    ):

    """Fix age to 5 Myr, density to 100 cm^{-3}"""

    par_names, hkt_unpadded = read_gutkin_models(
        param1='NeIII3869/OII3727', # [NeIII]3869/[OII]3727
        param2='OIII5007/Hbeta', # [OIII]5007/H\beta
        downsample={'Mup': 100, 'C': 10, 'xi': 0.3, 'logn': (1, 2, 3),
            'Z': (0.001, 0.004, 0.006, 0.014, 0.03),
            'logU': (-4., -3., -2., -1.),
            }
        )

    error_message = (
        'The model grid must have at most three dimensions, e.g. '
        '(ssp_type, logU, Z) or (logn, logU, Z), etc\n'
        'use the `downsample` keyword to reduce the grid dimensions')
    assert hkt_unpadded.squeeze().ndim<=4, error_message

    # Clad grid with padded nan's. This avoids connecting across unwanted regions.
    # Cladding is necessary around `metal` (axis 2) and `logU` (axis 4).
    hkt = hkt_unpadded.shape
    hkt = (hkt[0], hkt[1], hkt[2], hkt[3]+2, hkt[4], hkt[5]+2, hkt[6])
    hkt = np.full(hkt, np.nan)
    hkt[:, :, :, 1:-1, :, 1:-1, :] = hkt_unpadded
    hkt = hkt.transpose((4, 1, 2, 3, 0, 5, 6))

    n_ssp_types = hkt.shape[0]
    cmap = plt.get_cmap(cmap)
    norm = matplotlib.colors.Normalize(vmin=1, vmax=n_ssp_types*2)

    fig = plt.figure()    
    grid = matplotlib.gridspec.GridSpec(
        ncols=2, nrows=2, figure=fig,
        width_ratios=(1, 0.04),
        height_ratios=(1, 1))
    axes = fig.add_subplot(grid[:, 0])

    for i,subgrid in enumerate(hkt):

        logn = subgrid[0, 0, 1, 0, 1, 4] # Ones because of cladding...
        # print(i, logn)

        color = cmap(norm(float(1+i*2)))
        distance_fine = np.linspace(0, 1, 100)

        for j,(_x_, _y_) in enumerate(zip(
            subgrid[0, 0, 1:-1, 0, 1:-1, 6], subgrid[0, 0, 1:-1, 0, 1:-1, 7])):

            if np.all(np.isnan(_x_)) or np.all(np.isnan(_y_)):
                continue

            logU = subgrid[0, 0, 1, 0, 1:-1, 5]
            # print(logU)
            cmap_logU = plt.get_cmap('hot')
            norm_logU = matplotlib.colors.Normalize(vmin=logU.min(), vmax=logU.max())

            points = np.array([np.log10(_x_), np.log10(_y_), logU]).T

            # Linear length along the line:
            distance = np.cumsum(
                np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
            distance = np.insert(distance, 0, 0)/distance[-1]
            interpolator =  interp1d(distance, points, kind='linear', axis=0)
            interpolated = interpolator(distance_fine).T
            if (i==0) and (j in (0, len(logU)-1)):
                colorline(
                    10**interpolated[0], 10**interpolated[1], interpolated[2],
                    cmap=cmap_logU, norm=norm_logU, linewidth=2, linestyle='dotted',
                    )
            else:
                axes.plot( # Connect along with dashed lines.
                    10**interpolated[0], 10**interpolated[1],
                    color=color, lw=1.0, ls=':',
                    alpha=alpha, marker='none')
            axes.plot( # Connect along with dashed lines.
                    _x_, _y_,
                    color=color, lw=1.0, ls='none', alpha=alpha, marker='*', 
                    ms=7, mec='none')
            
        for j,(_x_, _y_) in enumerate(
            zip(subgrid[0, 0, 1:-1, 0, 1:-1, 6].T, subgrid[0, 0, 1:-1, 0, 1:-1, 7].T)):

            if np.all(np.isnan(_x_)) or np.all(np.isnan(_y_)):
                continue

            metal = subgrid[0, 0, 1:-1, 0, 1, 3]
            # print(metal)
            cmap_metal = plt.get_cmap('cool')
            norm_metal = matplotlib.colors.Normalize(vmin=metal.min(), vmax=metal.max())

            points = np.array([np.log10(_x_), np.log10(_y_), np.log10(metal)]).T

            # Linear length along the line:
            distance = np.cumsum(
                np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
            distance = np.insert(distance, 0, 0)/distance[-1]
            interpolator =  interp1d(distance, points, kind='linear', axis=0)
            interpolated = interpolator(distance_fine).T
            if (i==0) and (j in (0, len(metal)-1)):
                colorline(
                    10**interpolated[0], 10**interpolated[1], 10**interpolated[2],
                    cmap=cmap_metal, norm=norm_metal, linewidth=2, linestyle='solid',
                    )
            else:
                axes.plot( # Connect along with dashed lines.
                    10**interpolated[0], 10**interpolated[1],
                    color=color, lw=1.0, ls='-',
                    alpha=alpha, marker='none')
            axes.plot( # Connect along with dashed lines.
                    _x_, _y_,
                    color=color, lw=1.0, ls='none', alpha=alpha, marker='*', 
                    ms=7, mec='none')

        legend = (
            f'$\log\,n\;[\mathrm{{cm^{{-3}}}}] = {logn:2.1f}$' if i==0
            else f'${logn:2.1f}$')
        axes.lines[(-3 if i==0 else -2)].set_label(legend)

    """# Inset colorbars.
    cax_metal = inset_axes(
        axes, width="30%", height="4%", bbox_transform=axes.transAxes,
        bbox_to_anchor=(0.05, 0.14, 1, 1), loc="lower left", borderpad=0)
    cax_logU  = inset_axes(
        axes, width="30%", height="4%", bbox_transform=axes.transAxes,
        bbox_to_anchor=(0.05, 0.34, 1, 1), loc="lower left", borderpad=0)
    # Fake scatter plots.
    im_metal = axes.scatter(metal*np.nan, metal*np.nan, c=metal, cmap=cmap_metal,
        norm=norm_metal)
    im_logU  = axes.scatter(logU*np.nan, logU*np.nan, c=logU, cmap=cmap_logU,
        norm=norm_logU)
    cbar_metal = plt.colorbar(
        im_metal, cax=cax_metal, label='$Z$', #ticks=metal,
        orientation='horizontal')
    cbar_logU  = plt.colorbar(
        im_logU, cax=cax_logU,  label='$\log\,U$', #ticks=logU,
        orientation='horizontal')
    """
    cax_metal = fig.add_subplot(grid[0, 1])
    cax_logU  = fig.add_subplot(grid[1, 1])
    # Fake scatter plots.
    im_metal = axes.scatter(metal*np.nan, metal*np.nan, c=metal, cmap=cmap_metal,
        norm=norm_metal)
    im_logU  = axes.scatter(logU*np.nan, logU*np.nan, c=logU, cmap=cmap_logU,
        norm=norm_logU)
    cbar_metal = plt.colorbar(
        im_metal, cax=cax_metal, orientation='vertical')
    cbar_metal.set_label('$Z$', fontsize=16)
    cbar_logU  = plt.colorbar(
        im_logU, cax=cax_logU, orientation='vertical')
    cbar_logU.set_label('$\log\,U$', fontsize=16)

    legend = axes.legend(
        frameon=False, ncol=7, fontsize=20, scatterpoints=1, markerscale=2,
        handletextpad=0.3, labelspacing=0, handlelength=1.3, loc='lower left',
        columnspacing=0.9, borderaxespad=0.01, bbox_to_anchor=(-0., 1.))
    for l in legend.get_lines(): l.set_alpha(1); l.set_lw(4)
    axes.set_xlabel('$\mathrm{[OIII]\lambda5007/H\\beta}$', fontsize=16)
    axes.set_ylabel('$\mathrm{[NeIII]\lambda 3868/[OII]\lambda 3727}$', fontsize=16)

    for ax in fig.axes: ax.tick_params(axis='both', which='both', labelsize=14)

    axes.loglog()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


if __name__=="__main__":
    example_plot_ssp_metal_logu_interp_roberta()
