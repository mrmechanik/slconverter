import matplotlib.pyplot as plt

from models import ExtendedFrame, Helper
from io import load_files


def test():
    frame_dict = load_files('data')
    frames: [ExtendedFrame] = []

    for f in frame_dict.values():
        frames.extend(f)

    h = Helper(frames)
    h.limit = (0.6, 1.9)
    xs = h.x_vector
    ys = h.y_vector
    zs = h.z_vector

    fig = plt.figure(figsize=(40, 40))

    ax = fig.add_subplot(2, 2, 1)
    ax.plot(h.z_set)

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(xs, ys)

    ax = fig.add_subplot(2, 2, 3, projection='3d')
    ax.plot_trisurf(xs, ys, zs, cmap='jet')

    ax = fig.add_subplot(2, 2, 4)
    ax.tricontour(xs, ys, zs, cmap='jet')
    ax.tricontourf(xs, ys, zs, cmap='jet')

    plt.show()


if __name__ == '__main__':
    test()
