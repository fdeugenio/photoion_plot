import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from .harkatz_models import read_harkatz_models


def example_plot(
    cmap='viridis', alpha=0.5,
    ):

    """Fix age to 5 Myr, density to 100 cm^{-3}"""

    hkt_unpadded = read_harkatz_models(
        param1='N  2 6583.45A/H  1 6562.80A', # [NII]6584/H\alpha
        param2='O  3 5006.84A/H  1 4861.32A', # [OIII]5007/H\beta
        downsample={'ssp type': (0, 1), 'age': 5, 'logn': 2,
            'metal': (0.001, 0.003, 0.006, 0.014, 0.03),
            'logU': (-4., -3., -2., -1., -0.5)
            }
        )

    # Clad grid with padded nan's. This avoids connecting across unwanted regions.
    # Cladding is necessary around `metal` (axis 2) and `logU` (axis 4).
    hkt = hkt_unpadded.shape
    hkt = (hkt[0], hkt[1], hkt[2]+2, hkt[3], hkt[4]+2, hkt[5])
    hkt = np.full(hkt, np.nan)
    hkt[:, :, 1:-1, :, 1:-1, :] = hkt_unpadded

    n_ssp_types = hkt.shape[0]
    cmap = matplotlib.colormaps[cmap]
    norm = matplotlib.colors.Normalize(vmin=1, vmax=n_ssp_types*2)

    fig, axes = plt.subplots(1, 1)

    for i,subgrid in enumerate(hkt):

        ssp_type = subgrid[0, 1, 0, 1, 0] # Ones because of cladding...

        color = cmap(norm(float(1+i*2)))
        axes.plot( # Connect along with dashed lines.
            subgrid[0, :, 0, :, 5], subgrid[0, :, 0, :, 6],
            color=color, lw=1.0, ls='--', alpha=alpha, ms=7, marker='*', mec='none')
        axes.plot( # Connect across with solid lines.
            subgrid[0, :, 0, :, 5].T, subgrid[0, :, 0, :, 6].T,
            color=color, lw=1.0, ls='-', alpha=alpha, marker='none')
        legend = 'binary' if ssp_type else 'single'
        axes.lines[-1].set_label(legend)

        # Draw arrows just for the first grid.
        if i==0:
            for which_index in (1, -2):
                wi = which_index
                arrow1 = subgrid[0, 1:-1, 0,   wi, 5:7]
                arrow2 = subgrid[0,   wi, 0, 1:-1, 5:7]
                for k in range(len(arrow1)-1):
                    axes.annotate("", arrow1[k+1, :], xytext=arrow1[k, :],
                        arrowprops=dict(arrowstyle="->", facecolor='r', edgecolor='b')
                        )
                for h in range(len(arrow2)-1):
                    axes.annotate("", arrow2[h+1, :], xytext=arrow2[h, :],
                        arrowprops=dict(arrowstyle="->", facecolor='k')
                        )


    legend = axes.legend(
        frameon=False, ncol=7, fontsize=20, scatterpoints=1, markerscale=2,
        handletextpad=0.3, labelspacing=0, handlelength=1.3, loc='lower left',
        columnspacing=0.9, borderaxespad=0.01, bbox_to_anchor=(-0., 1.))
    for l in legend.get_lines(): l.set_alpha(1); l.set_lw(4)
    axes.set_xlabel('$\mathrm{[NII]\lambda 6584/H\\alpha}$', fontsize=16)
    axes.set_ylabel('$\mathrm{[OIII]\lambda5007/H\\beta}$', fontsize=16)

    axes.loglog()
    plt.show()


if __name__=="__main__":
    example_plot()
