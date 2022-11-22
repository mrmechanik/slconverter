import logging.config

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from models import ExtractedFrame, FrameGroup
from slio import load_files

logger = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.INFO)


def test():
    frame_dict = load_files('data/')

    frames: [ExtractedFrame] = []

    for f in frame_dict.values():
        frames.extend(f)

    h: FrameGroup = FrameGroup(frames).filter().uniquify().normalize()

    dim: tuple[int, int] = (2, 2)
    fig: Figure = plt.figure(figsize=(20, 20))

    logger.debug('Generating Plot 1')
    ax1: Axes = fig.add_subplot(*dim, 1, title='Depth Distribution')
    sns.kdeplot(h.zs, ax=ax1)

    logger.debug('Generating Plot 2')
    ax2: Axes = fig.add_subplot(*dim, 2, title='Water Body Coverage')
    ax2.plot(h.xs, h.ys)

    logger.debug('Generating Plot 3')
    ax3: Axes3D = fig.add_subplot(*dim, 3, title='3D Visualization', projection='3d')
    ax3.plot_trisurf(h.xs, h.ys, list(map(lambda v: 0 - v, h.zs)), cmap='jet', edgecolor='none')

    logger.debug('Generating Plot 4')
    ax4: Axes = fig.add_subplot(*dim, 4, title='Bathymetric Map')
    ax4.tricontourf(h.xs, h.ys, h.zs, cmap='jet')

    plt.show()


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
            },
        }
    })
    test()
