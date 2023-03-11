import logging.config

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch
from matplotlib.tri import TriContourSet
from mpl_toolkits.axes_grid1 import make_axes_locatable

from models import ExtractedFrame, FrameGroup
from slio import load_files, save_stl

logger = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.INFO)


def main() -> None:
    # load files into frames
    frames: dict[str, list[ExtractedFrame]] = load_files('data/')

    # perform filtering, duplicate removal, normalization & flip
    group: FrameGroup = FrameGroup(frames).filter().uniquify().normalize().flip('xy')
    dim: tuple[int, int] = (2, 2)
    fig: Figure = plt.figure(figsize=(20, 20))

    # add distribution plot
    logger.info('Generating Plot 1')
    ax1: Axes = fig.add_subplot(*dim, 1, title='Depth Distribution')
    sns.kdeplot(group.zs, ax=ax1)

    # add plot representing the driven route
    logger.info('Generating Plot 2')
    ax2: Axes = fig.add_subplot(*dim, 2, title='Water Body Coverage')
    ax2.plot(group.xs, group.ys)

    # calculate the shape (including hull & holes)
    logger.info('Generating Plot 3')
    shape: FrameGroup = group.shape()
    ax3: Axes = fig.add_subplot(*dim, 3, title='Estimated Water Body')
    ax3.fill(shape.xs, shape.ys, alpha=0.2)
    ax3.plot(shape.xs, shape.ys)
    # add all patches representing a hole
    any(map(lambda p:  ax3.add_patch(PathPatch(p, color='white')), shape.hole_paths))

    logger.info('Generating Plot 4')

    cmap: LinearSegmentedColormap = LinearSegmentedColormap.from_list(
        name='Water Depth', colors=('w', 'CornflowerBlue', 'DarkBlue')
    )

    # retrieve shallowest and deepest point
    lx, ly, lz = group.shallowest.as_3d_pos()
    hx, hy, hz = group.deepest.as_3d_pos()

    # create a bathymetric map from the triangulation
    ax4: Axes = fig.add_subplot(*dim, 4, title='Bathymetric Map')
    tcs: TriContourSet = ax4.tricontourf(group.triangulate(shape), group.zs, cmap=cmap)

    # add labels for the shallowest & deepest point
    ax4.scatter(lx, ly, marker='x', c='orange')
    ax4.text(lx, ly, '{:.3f} m'.format(lz), c='orange', horizontalalignment='right')
    ax4.scatter(hx, hy, marker='x', c='orange')
    ax4.text(hx, hy, '{:.3f} m'.format(hz), c='orange', horizontalalignment='right')

    # add legend (color bar) to the side of the bathymetric map for height representation
    cax: Axes = make_axes_locatable(ax4).append_axes('right', size='3%', pad=0.1)
    cbar: Colorbar = plt.colorbar(tcs, cax=cax, cmap=cmap)
    cbar.ax.set_yticklabels(
        list(map(lambda n: round(n, 1), np.linspace(0, round(max(group.zs), 1), len(cbar.get_ticks()), endpoint=True)))
    )

    plt.show()
    logger.info('Plotting completed')

    # export and save the model
    save_stl('data/res.stl', group.as_exportable(shape, fill_holes=True, scale=30))


if __name__ == '__main__':
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'default': {
                'format': '[%(levelname)s] [%(asctime)s] : %(message)s < @%(name)s:%(lineno)s',
                'datefmt': '%d.%m.%Y %H:%M:%S'
            }
        },
        'handlers': {
            'default': {
                'level': 'DEBUG',
                'formatter': 'default',
                'class': 'logging.StreamHandler'
            }
        },
        'loggers': {
            '': {
                'handlers': ['default'],
                'level': 'DEBUG',
                'propagate': False
            }
        }
    })
    main()
