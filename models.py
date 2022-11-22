import logging

from sllib import Frame
from sllib.definitions import FEET_CONVERSION

logger = logging.getLogger(__name__)
default_sensitivity = 0.07


class ExtractedFrame:
    def __init__(self, frame: Frame):
        self.keel_depth_m: float = frame.keel_depth * FEET_CONVERSION if hasattr(frame, 'keel_depth') else 0
        self.actual_water_depth_m: float = frame.water_depth_m + self.keel_depth_m
        self.latitude = frame.latitude
        self.longitude = frame.longitude

    def __repr__(self):
        return '{:.2f}'.format(self.actual_water_depth_m)


class FrameGroup:
    def __init__(
            self, frames: [ExtractedFrame],
            use_highest_keel_depth_m_as_min_depth: bool = True,
            use_reasonable_depth_as_max_depth: bool | float = True
    ):
        self._low: float = 0
        self._high: float = 0

        self._frames: [ExtractedFrame] = frames

        self._xs: [float] = []
        self._ys: [float] = []
        self._zs: [float] = []

        if use_highest_keel_depth_m_as_min_depth:
            self._low = self.max_keel_m
            logger.debug(f'Detected {self._low} as lower border to invalid measurements')

        if use_reasonable_depth_as_max_depth:
            if type(use_reasonable_depth_as_max_depth) == bool:
                sensitivity = default_sensitivity
            else:
                sensitivity = use_reasonable_depth_as_max_depth

            logger.debug('{} as maximum difference'.format(f'Default of {default_sensitivity}' if type(
                use_reasonable_depth_as_max_depth
            ) == bool else use_reasonable_depth_as_max_depth))

            self._high = self.get_max_reasonable_depth(sensitivity)
            logger.debug(f'Detected {self._high} as upper border value to invalid measurements')

    @property
    def frames(self) -> [ExtractedFrame]:
        return self._frames.copy()

    @property
    def min_depth(self) -> float:
        return self._low

    @min_depth.setter
    def min_depth(self, value: float):
        self._validate_depth(value)
        self._low = value

    @property
    def max_depth(self) -> float:
        return self._high

    @max_depth.setter
    def max_depth(self, value: float):
        self._validate_depth(value)
        self._high = value

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

    @property
    def max_keel_m(self) -> float:
        return max(map(lambda f: f.keel_depth_m, self.frames))

    def get_max_reasonable_depth(self, sensitivity: float) -> float:
        ordered_depths: [float] = sorted(
            set(filter(lambda m: m > self._low, map(lambda f: f.actual_water_depth_m, self.frames)))
        )

        return next(
            (cur_d for cur_d, next_d in zip(ordered_depths, ordered_depths[1:]) if next_d - cur_d >= sensitivity),
            ordered_depths[-1]
        )

    def clear(self):
        self._xs.clear()
        self._ys.clear()
        self._zs.clear()

    @staticmethod
    def _validate_depth(value: float):
        if value < 0:
            raise ValueError('Depth cannot be negative')

    def _limited_frames(self) -> [ExtractedFrame]:
        if self._low >= self._high:
            return self.frames

        return list(filter(lambda f: self._low < f.actual_water_depth_m <= self._high, self.frames))

    def _build_vectors(self):
        if self._xs:
            return

        any(map(self._split_row, self._limited_frames()))

    def _split_row(self, frame: ExtractedFrame):
        self._xs.append(frame.latitude)
        self._ys.append(frame.longitude)
        self._zs.append(frame.actual_water_depth_m)
