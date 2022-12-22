import logging
import time
from functools import wraps
from typing import Any, Callable, Optional

import alphashape
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from shapely.geometry import Polygon, MultiPolygon
from sllib import Frame
from sllib.definitions import FEET_CONVERSION
from triangle import triangulate

logger: logging.Logger = logging.getLogger(__name__)
auto: str = 'auto'

default_sensitivity: float = 0.07
default_alpha: float = 8
default_hole_alpha: float = 1.5
default_max_area: float = 0.04

Fs = list[float]
P = tuple[float, float]
Ps = list[P]
EFs = list['ExtractedFrame']


def proc_time_log(msg: str) -> Any:
    def deco(func: Callable[[Optional[Any]], Optional[Any]]) -> Optional[Any]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Optional[Any]:
            logger.debug(msg)
            timer: float = time.time()
            res: Optional[Any] = func(*args, **kwargs)
            logger.debug('Took {:.2f}s'.format(time.time() - timer))
            return res

        return wrapper

    return deco


def return_new_group(
        func: Callable[['FrameGroup', Optional[Any]], EFs]
) -> Callable[['FrameGroup', Optional[Any]], 'FrameGroup']:
    @wraps(func)
    def wrapper(self, *args, **kwargs) -> 'FrameGroup':
        res: EFs = func(self, *args, **kwargs)
        group: FrameGroup = FrameGroup(res)
        group._holes = self._holes
        return group

    return wrapper


class ExtractedFrame:
    def __init__(self, frame: Frame) -> None:
        self.keel_depth_m: float = frame.keel_depth * FEET_CONVERSION if hasattr(frame, 'keel_depth') else 0
        self.water_depth_m: float = frame.water_depth_m
        self.latitude: float = frame.latitude
        self.longitude: float = frame.longitude
        self.timestamp: int = frame.time1

    def __as_data_tuple(self) -> tuple[float, float, float, float]:
        return self.keel_depth_m, self.water_depth_m, self.latitude, self.longitude

    def as_pos(self) -> P:
        return self.latitude, self.longitude

    def __eq__(self, other) -> bool:
        return isinstance(other, self.__class__) and self.__as_data_tuple() == other.__as_data_tuple()

    def __hash__(self) -> int:
        return hash(self.__as_data_tuple())

    def __repr__(self) -> str:
        return '{{{:.6f} {:.6f} {:.2f}}}'.format(self.latitude, self.longitude, self.water_depth_m)


class FrameGroup:
    def __init__(self, frames: EFs | dict[str, EFs]):
        self.__xs: Fs = []
        self.__ys: Fs = []
        self.__zs: Fs = []
        self._holes: list[FrameGroup] = []
        self.__paths: list[PathPatch] = []
        self.__frames: EFs = self.__convert_frame_dict(frames) if type(frames) == dict else frames
        self.__hash: int = hash(tuple(frames))

    @property
    def frames(self) -> EFs:
        return self.__frames.copy()

    @property
    def hole_paths(self) -> list[Path]:
        if not self.__paths:
            self.__paths = list(
                map(lambda h: FrameGroup.__construct_hole_paths(h.__as_pos()), self._holes)
            )

        return self.__paths.copy()

    @property
    def xs(self) -> Fs:
        self.__validate_vectors()
        return self.__xs.copy()

    @property
    def ys(self) -> Fs:
        self.__validate_vectors()
        return self.__ys.copy()

    @property
    def zs(self) -> Fs:
        self.__validate_vectors()
        return self.__zs.copy()

    @proc_time_log('Constructing shape...')
    @return_new_group
    def shape(
            self, alpha: float = default_alpha,
            max_area: float = default_max_area,
            hole_alpha: float = default_hole_alpha
    ) -> 'FrameGroup':
        all_points: Ps = self.__as_pos()

        if alpha < 0 or hole_alpha < 0:
            raise ValueError(
                'The alpha value of an alpha shape may not be less than 0 as 0 already results in a convex hull '
                'which would be the largest alpha shape available'
            )
        elif max_area < 0:
            raise ValueError('Delaunay triangles can only be constructed if the area is valid (greater than 0)')

        hull_poly: Polygon | Any = alphashape.alphashape(all_points, float(alpha))

        if type(hull_poly) != Polygon:
            raise ValueError(
                f'The provided alpha value {alpha} causes the resulting polygon to consist of multiple polygons '
                f'or a single point or line pointing to data loss, the alpha value may be lowered to prevent this'
            )

        hull_points: Ps = list(hull_poly.exterior.coords)
        p_len: int = len(all_points)
        logger.debug(f'Hull consists of {len(hull_points)} points based on {len(all_points)} points')

        delaunay: dict = triangulate({'vertices': all_points}, opts=f'a{max_area}')
        vertices: Ps = delaunay['vertices']
        triangles: list[tuple[int, int, int]] = delaunay['triangles']
        indices: list[tuple[int, int, int]] = list(filter(
            lambda idxs: self.__is_interior(hull_points, list(map(lambda i: vertices[i], idxs))), filter(
                lambda idxs: not all(map(lambda i: i < p_len, idxs)), triangles
            )
        ))
        hole_points: Ps = list(
            map(lambda i: all_points[i], filter(lambda i: i < p_len, (idx for sub in indices for idx in sub)))
        )

        hull_frames: EFs = self.__order_and_map_points(hull_points)

        if hole_points:
            self.__construct_holes(hole_points, hole_alpha)

        logger.debug(f'Detected {len(self._holes)} holes')
        return hull_frames

    @proc_time_log('Calculating maximum plausible depth...')
    def get_max_plausible_depth(self, low: float, sensitivity: float) -> float:
        depths: Fs = sorted(set(filter(lambda m: m > low, map(lambda f: f.water_depth_m, self.__frames))))
        max_depth = next((cur for cur, fut in zip(depths, depths[1:]) if fut - cur >= sensitivity), depths[-1])
        logger.debug(f'Detected {max_depth} as max plausible depth for {sensitivity} as sensitivity')
        return max_depth

    def filter_min(self, min_border: float | str = auto) -> 'FrameGroup':
        return self.filter(min_border, 20000)

    def filter_max(self, max_border: float | str = auto, sensitivity: float = default_sensitivity) -> 'FrameGroup':
        return self.filter(0, max_border, sensitivity)

    @proc_time_log('Filtering frames...')
    @return_new_group
    def filter(
            self, min_border: float | str = auto,
            max_border: float | str = auto,
            sensitivity_for_max: float = default_sensitivity
    ) -> 'FrameGroup':
        self.__validate_param(min_border)
        self.__validate_param(max_border)

        low: float = self.get_max_keel_m() if min_border == auto else min_border
        high: float = self.get_max_plausible_depth(low, sensitivity_for_max) if max_border == auto else max_border

        if low >= high:
            raise ValueError(f'No outliers detectable for LOW {low} and HIGH {high}')

        logger.debug(f'Using {low} and {high} as border values')
        return list(filter(lambda f: low < f.water_depth_m <= high, self.__frames))

    @proc_time_log('Removing duplicates...')
    @return_new_group
    def uniquify(self) -> 'FrameGroup':
        ordered_uniques: EFs = sorted(list(set(self.__frames)), key=lambda f: f.timestamp)
        logger.debug(f'Reduced {len(self.__frames)} frames to {len(ordered_uniques)} frames')
        return ordered_uniques

    @proc_time_log('Normalizing...')
    @return_new_group
    def normalize(
            self, min_x: float | str = auto, min_y: float | str = auto, scale: float | str = auto
    ) -> 'FrameGroup':
        self.__validate_param(min_x)
        self.__validate_param(min_y)
        self.__validate_param(scale)

        if min_x == auto:
            min_x: float = min(map(lambda f: f.latitude, self.__frames))

        if min_y == auto:
            min_y: float = min(map(lambda f: f.longitude, self.__frames))

        if scale == auto:
            diff_x: float = max(map(lambda f: f.latitude, self.__frames)) - min_x
            diff_y: float = max(map(lambda f: f.longitude, self.__frames)) - min_y

            factor: float = diff_x if diff_x > diff_y else diff_y
            scale: int = 1

            while scale * factor < 1:
                scale *= 10

        logger.debug(f'Using {min_x} and {min_y} as minimal values and {scale} as scale factor')
        return list(map(lambda f: self.__normalize_row(min_x, min_y, scale, f), self.__frames))

    @proc_time_log('Filtering values...')
    @return_new_group
    def inspect(self, min_point: P, max_point: P) -> 'FrameGroup':
        min_x, min_y = min_point
        max_x, max_y = max_point

        if min_x >= max_x or min_y >= max_y:
            raise ValueError('Min cannot be smaller than max')

        return list(filter(lambda f: min_x <= f.latitude < max_x and min_y <= f.longitude < max_y, self.__frames))

    def get_max_keel_m(self) -> float:
        return max(map(lambda f: f.keel_depth_m, self.__frames))

    @staticmethod
    @proc_time_log('Reducing to list & updating timestamps...')
    def __convert_frame_dict(frames: dict[str, EFs]) -> EFs:
        internal_frames: EFs = []
        any(map(internal_frames.extend, frames.values()))

        stamps: list[int] = list(map(lambda f: f.timestamp, internal_frames))
        stamp_offset: int = max(stamps) - min(stamps)
        internal_frames: EFs = []

        for idx, frame_list in enumerate(frames.values()):
            list_offset: int = idx * stamp_offset

            for frame in frame_list:
                frame.timestamp += list_offset
                internal_frames.append(frame)

        return internal_frames

    @staticmethod
    def __construct_hole_paths(points: Ps) -> Path:
        actions: list = [Path.MOVETO] + [Path.LINETO] * (len(points) - 2) + [Path.CLOSEPOLY]
        return Path(points, actions)

    @staticmethod
    def __is_interior(hull_pts: list[P], points: list[P]) -> bool:
        return Polygon(points).within(Polygon(hull_pts))

    @staticmethod
    def __normalize_row(min_x: float, min_y: float, scale: float, frame: ExtractedFrame) -> ExtractedFrame:
        frame.latitude = (frame.latitude - min_x) * scale
        frame.longitude = (frame.longitude - min_y) * scale
        return frame

    @staticmethod
    def __validate_param(param: float | str) -> None:
        if type(param) == str and param != auto:
            raise ValueError(f'Parameters may only be numeric if not set to "{auto}"')

    @proc_time_log('Constructing holes')
    def __construct_holes(self, hole_points: Ps, hole_alpha: float) -> None:
        polys: MultiPolygon | Polygon | Any = alphashape.alphashape(hole_points, float(hole_alpha))

        if type(polys) != MultiPolygon and type(polys) != Polygon:
            raise ValueError(
                f'The hole alpha {hole_alpha} value causes the holes to be too imprecise, the alpha value may be '
                f'lowered to prevent this'
            )

        if type(polys) == Polygon:
            polys = [polys]
        elif type(polys) == MultiPolygon:
            polys = polys.geoms

        self._holes: list[FrameGroup] = list(
            map(lambda p: FrameGroup(self.__order_and_map_points(list(reversed(p.exterior.coords)))), polys)
        )

    @proc_time_log(f'Reconstructing...')
    def __rebuild_vectors(self) -> None:
        self.__clear()
        any(map(self.__split_row, self.__frames))

    def __order_and_map_points(self, ext_points: list[P]) -> EFs:
        unordered_frames: EFs = list(filter(lambda f: f.as_pos() in ext_points, self.__frames))
        return list(map(lambda pos: next((f for f in unordered_frames if f.as_pos() == pos)), ext_points))

    def __as_pos(self) -> list[P]:
        return list(map(lambda f: f.as_pos(), self.__frames))

    def __validate_vectors(self) -> None:
        cur_hash: int = hash(tuple(self.__frames))

        if cur_hash != self.__hash or not self.__xs:
            self.__rebuild_vectors()
            self.__hash: int = cur_hash

    def __clear(self) -> None:
        self.__xs.clear()
        self.__ys.clear()
        self.__zs.clear()

    def __split_row(self, frame: ExtractedFrame) -> None:
        self.__xs.append(frame.latitude)
        self.__ys.append(frame.longitude)
        self.__zs.append(frame.water_depth_m)

    def __repr__(self) -> str:
        return '[{}]'.format('\n'.join(map(repr, self.__frames)))
