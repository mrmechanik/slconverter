import logging.config

import matplotlib.pyplot as plt
import seaborn as sns

from models import ExtendedFrame, Helper
from slio import load_files

logger = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.INFO)


def test():
    frame_dict = load_files('data')

    frames: [ExtendedFrame] = []

    for f in frame_dict.values():
        frames.extend(f)

    h = Helper(frames)
    xs = h.x_vector
    ys = h.y_vector
    zs = h.z_vector

    dim = (2, 2)
    fig = plt.figure(figsize=(20, 20))

    ax1 = fig.add_subplot(*dim, 1, title='Depth Distribution')
    sns.kdeplot(zs, ax=ax1)

    ax2 = fig.add_subplot(*dim, 2, title='Water Body Coverage')
    ax2.plot(xs, ys)

    ax3 = fig.add_subplot(*dim, 3, title='3D Visualization', projection='3d')
    ax3.plot_trisurf(xs, ys, list(map(lambda v: 0 - v, zs)), cmap='jet', edgecolor='none')

    ax4 = fig.add_subplot(*dim, 4, title='Bathymetric Map')
    ax4.tricontourf(xs, ys, zs, cmap='jet')

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
