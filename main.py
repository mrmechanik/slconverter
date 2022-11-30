import logging.config

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from models import ExtractedFrame, FrameGroup
from slio import load_files

logger = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.INFO)


def test():
    frames: dict[str, list[ExtractedFrame]] = load_files('data/')

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
    ax3: Axes = fig.add_subplot(*dim, 3, title='Normal Interpolated Bathymetric Map')
    ax3.tricontourf(h.xs, h.ys, h.zs, cmap='jet')

    logger.debug('Generating Plot 4')
    hull = h.get_hull()
    ax4: Axes = fig.add_subplot(*dim, 4, title='Convex Hull')
    ax4.plot(h.xs, h.ys, color='red')
    ax4.plot(hull.points[hull.vertices, 0], hull.points[hull.vertices, 1])

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
