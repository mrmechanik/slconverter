import logging
import time
from functools import wraps

from sllib import Frame
from sllib.definitions import FEET_CONVERSION

logger: logging.Logger = logging.getLogger(__name__)
default_sensitivity: float = 0.07
auto: str = 'auto'


def proc_time_log(msg):
    def deco(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.debug(msg)
            timer: float = time.time()
            res = func(*args, **kwargs)
            logger.debug('Took {:.2f}s'.format(time.time() - timer))
            return res

        return wrapper

    return deco


def return_new_group(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs) -> 'FrameGroup':
        res: [ExtractedFrame] = func(self, *args, **kwargs)
        return FrameGroup(res)

    return wrapper


class ExtractedFrame:
    def __init__(self, frame: Frame) -> None:
        self.keel_depth_m: float = frame.keel_depth * FEET_CONVERSION if hasattr(frame, 'keel_depth') else 0
        self.water_depth_m: float = frame.water_depth_m
        self.latitude: float = frame.latitude
        self.longitude: float = frame.longitude

    def __as_tuple(self) -> tuple[float, float, float, float]:
        return self.keel_depth_m, self.water_depth_m, self.latitude, self.longitude

    def __eq__(self, other) -> bool:
        return isinstance(other, self.__class__) and self.__as_tuple() == other.__as_tuple()

    def __hash__(self) -> int:
        return hash(self.__as_tuple())

    def __repr__(self) -> str:
        return '{{{:.6f} {:.6f} {:.2f}}}'.format(self.latitude, self.longitude, self.water_depth_m)


class FrameGroup:
    __xs: [float] = []
    __ys: [float] = []
    __zs: [float] = []

    def __init__(self, frames: [ExtractedFrame]):
        self._frames: [ExtractedFrame] = frames
        self._hash = hash(tuple(frames))

    @property
    def frames(self) -> [ExtractedFrame]:
        self.__validate_existence()
        return self._frames.copy()

    @property
    def xs(self) -> list[float]:
        self.__validate_existence()
        return self.__xs.copy()

    @property
    def ys(self) -> list[float]:
        self.__validate_existence()
        return self.__ys.copy()

    @property
    def zs(self) -> list[float]:
        self.__validate_existence()
        return self.__zs.copy()

    def get_max_keel_m(self) -> float:
        return max(map(lambda f: f.keel_depth_m, self._frames))

    @proc_time_log('Calculating maximum plausible depth...')
    def get_max_plausible_depth(self, low: float, sensitivity: float) -> float:
        depths: [float] = sorted(set(filter(lambda m: m > low, map(lambda f: f.water_depth_m, self._frames))))
        max_depth = next((cur for cur, fut in zip(depths, depths[1:]) if fut - cur >= sensitivity), depths[-1])
        logger.debug(f'Detected {max_depth} as max plausible depth for {sensitivity} as sensitivity')
        return max_depth

    def filter_min(self, min_border: float | str = auto) -> [ExtractedFrame]:
        return self.filter(min_border, 20000)

    def filter_max(self, max_border: float | str = auto, sensitivity: float = default_sensitivity) -> [ExtractedFrame]:
        return self.filter(0, max_border, sensitivity)

    @proc_time_log('Filtering frames...')
    @return_new_group
    def filter(
            self, min_border: float | str = auto, max_border: float | str = auto,
            sensitivity_for_max: float = default_sensitivity
    ) -> [ExtractedFrame]:
        self.__validate_param(min_border)
        self.__validate_param(max_border)

        low = self.get_max_keel_m() if min_border == auto else min_border
        high = self.get_max_plausible_depth(low, sensitivity_for_max) if max_border == auto else max_border

        if low >= high:
            raise ValueError(f'No outliers detectable for LOW {low} and HIGH {high}')

        logger.debug(f'Using {low} and {high} as border values')
        return list(filter(lambda f: low < f.water_depth_m <= high, self._frames))

    @proc_time_log('Removing duplicates...')
    @return_new_group
    def uniquify(self) -> [ExtractedFrame]:
        uniques: set = set()
        uniques_add = uniques.add
        ordered_uniques: [float] = [f for f in self._frames if not (f in uniques or uniques_add(f))]
        logger.debug(f'Reduced {len(self._frames)} frames to {len(ordered_uniques)} frames')
        return ordered_uniques

    @proc_time_log('Normalizing...')
    @return_new_group
    def normalize(
            self, min_x: float | str = auto, min_y: float | str = auto, scale: float | str = auto
    ) -> [ExtractedFrame]:
        self.__validate_param(min_x)
        self.__validate_param(min_y)
        self.__validate_param(scale)

        if min_x == auto:
            min_x = min(map(lambda f: f.latitude, self._frames))

        if min_y == auto:
            min_y = min(map(lambda f: f.longitude, self._frames))

        if scale == auto:
            diff_x = max(map(lambda f: f.latitude, self._frames)) - min_x
            diff_y = max(map(lambda f: f.longitude, self._frames)) - min_y

            factor = diff_x if diff_x > diff_y else diff_y
            scale = 1

            while scale * factor < 1:
                scale *= 10

        logger.debug(f'Using {min_x} and {min_y} as minimal values and {scale} as scale factor')
        return list(map(lambda f: self.__normalize_row(min_x, min_y, scale, f), self._frames))

    @proc_time_log('Filtering values...')
    @return_new_group
    def inspect(self, min_x: float, min_y: float, max_x: float, max_y: float):
        if min_x >= max_x or min_y >= max_y:
            raise ValueError('Min cannot be smaller than max')

        return list(filter(lambda f: min_x <= f.latitude < max_x and min_y <= f.longitude < max_y, self._frames))

    @staticmethod
    def __normalize_row(min_x: float, min_y: float, scale: float, frame: ExtractedFrame) -> ExtractedFrame:
        frame.latitude = (frame.latitude - min_x) * scale
        frame.longitude = (frame.longitude - min_y) * scale
        return frame

    @staticmethod
    def __validate_param(param: float | str) -> None:
        if type(param) == str and param != auto:
            raise ValueError(f'Parameters may only be floats if not set to "{auto}"')

    def __validate_existence(self) -> None:
        cur_hash = hash(tuple(self._frames))

        if cur_hash != self._hash or not self.__xs:
            self.__rebuild_vectors()
            self._hash = cur_hash

    @proc_time_log(f'Reconstructing...')
    def __rebuild_vectors(self) -> None:
        self.__clear()
        any(map(self.__split_row, self._frames))

    def __clear(self) -> None:
        self.__xs.clear()
        self.__ys.clear()
        self.__zs.clear()

    def __split_row(self, frame: ExtractedFrame) -> None:
        self.__xs.append(frame.latitude)
        self.__ys.append(frame.longitude)
        self.__zs.append(frame.water_depth_m)
