from sllib import Frame
from sllib.definitions import FEET_CONVERSION


class ExtendedFrame(Frame):
    def __init__(self, frame: Frame):
        super().__init__()

        for key, value in frame.to_dict().items():
            try:
                setattr(self, key, value)
            except AttributeError:
                pass

    @property
    def keel_depth_m(self) -> float:
        return self.keel_depth * FEET_CONVERSION if hasattr(self, 'keel_depth') else 0

    @property
    def actual_water_depth_m(self) -> float:
        return self.water_depth_m + self.keel_depth_m

    def __repr__(self):
        return '{:.2f}'.format(self.actual_water_depth_m)


class Helper:
    _xs = []
    _ys = []
    _zs = []
    _limits = (0, 0)

    def __init__(self, frames: [ExtendedFrame]):
        self.frames = frames

    @property
    def limit(self):
        return self._limits

    @limit.setter
    def limit(self, value: tuple[float, float]):
        self._limits = value

    @property
    def limited_frames(self):
        return self._limited_frames().copy()

    @property
    def x_vector(self) -> list[float]:
        self._build_vectors()
        return self._xs.copy()

    @property
    def x_set(self):
        return list({*self.x_vector})

    @property
    def y_vector(self) -> list[float]:
        self._build_vectors()
        return self._ys.copy()

    @property
    def y_set(self):
        return list({*self.y_vector})

    @property
    def z_vector(self) -> list[float]:
        self._build_vectors()
        return self._zs.copy()

    @property
    def z_set(self):
        return list({*self.z_vector})

    def clear(self):
        self._xs.clear()
        self._ys.clear()
        self._zs.clear()

    def _limited_frames(self) -> ['ExtendedFrame']:
        low, high = self._limits
        return list(filter(lambda f: low == high or low <= f.actual_water_depth_m <= high, self.frames))

    def _build_vectors(self):
        if self._xs:
            return

        any(map(self._split_row, self._limited_frames()))

    def _split_row(self, frame: ExtendedFrame):
        self._xs.append(frame.latitude)
        self._ys.append(frame.longitude)
        self._zs.append(frame.actual_water_depth_m)
