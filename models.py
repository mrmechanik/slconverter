import logging
import time
from functools import wraps
from typing import Any, Callable, Optional, Type, TypeVar, Iterator

from alphashape import alphashape
from matplotlib.path import Path
from matplotlib.tri import Triangulation
from numpy import ndarray, array
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.base import BaseGeometry
from sllib import Frame
from sllib.definitions import FEET_CONVERSION
from triangle import triangulate

logger: logging.Logger = logging.getLogger(__name__)
auto: str = 'auto'

default_sensitivity: float = 0.07
default_alpha: float = 8
default_hole_alpha: float = 1.5
default_max_area: float = 0.04

T: TypeVar = TypeVar('T')
D1s: Type = list[float]
D2: Type = tuple[float, float]
D2s: Type = list[D2]
D3: Type = tuple[float, float, float]
D3s: Type = list[D3]

Is: Type = list[tuple[int, int, int]]
EFs: Type = list['ExtractedFrame']


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


def flat(iterable: Iterator[Iterator[T]]) -> Iterator[T]:
    return (elem for sub_iterable in iterable for elem in sub_iterable)


def return_new_group(
        func: Callable[['FrameGroup', Optional[Any]], EFs]
) -> Callable[['FrameGroup', Optional[Any]], 'FrameGroup']:
    @wraps(func)
    def wrapper(self, *args, **kwargs) -> 'FrameGroup':
        res: EFs = func(self, *args, **kwargs)
        group: FrameGroup = FrameGroup(res)
        group._hull = self._hull
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

    def as_2d_pos(self) -> D2:
        return self.latitude, self.longitude

    def as_3d_pos(self) -> tuple[float, float, float]:
        return self.latitude, self.longitude, self.water_depth_m

    def update_2d_pos(self, point: D2):
        self.latitude, self.longitude = point

    def __as_data_tuple(self) -> tuple[float, float, float, float]:
        return self.keel_depth_m, self.water_depth_m, self.latitude, self.longitude

    def __eq__(self, other) -> bool:
        return isinstance(other, self.__class__) and self.__as_data_tuple() == other.__as_data_tuple()

    def __hash__(self) -> int:
        return hash(self.__as_data_tuple())

    def __repr__(self) -> str:
        return '{{{:.6f} {:.6f} {:.2f}}}'.format(self.latitude, self.longitude, self.water_depth_m)


class FrameGroup:
    def __init__(self, frames: EFs | dict[str, EFs]) -> None:
        self.__xs: D1s = []
        self.__ys: D1s = []
        self.__zs: D1s = []
        self.__paths: list[Path] = []
        self.__by_depths: EFs = []

        self._hull: FrameGroup | None = None
        self._holes: list[FrameGroup] = []
        self.__frames: EFs = self.__convert_frame_dict(frames) if type(frames) == dict else frames
        self.__hash: int = hash(tuple(frames))

    @property
    def frames(self) -> EFs:
        return self.__frames.copy()

    @property
    def hole_paths(self) -> list[Path]:
        if not self.__paths:
            self.__paths: list[Path] = list(map(lambda h: self.__construct_hole_paths(h.__as_2d_pos()), self._holes))

        return self.__paths.copy()

    @property
    def xs(self) -> D1s:
        self.__validate_vectors()
        return self.__xs.copy()

    @property
    def ys(self) -> D1s:
        self.__validate_vectors()
        return self.__ys.copy()

    @property
    def zs(self) -> D1s:
        self.__validate_vectors()
        return self.__zs.copy()

    @property
    def deepest(self) -> ExtractedFrame:
        if not self.__by_depths:
            self.__order_frames()

        return self.__by_depths.copy()[0]

    @property
    def shallowest(self) -> ExtractedFrame:
        if not self.__by_depths:
            self.__order_frames()

        return self.__by_depths.copy()[-1]

    @proc_time_log('Constructing shape...')
    @return_new_group
    def shape(
            self, alpha: float = default_alpha,
            max_area: float = default_max_area,
            hole_alpha: float = default_hole_alpha
    ) -> 'FrameGroup':
        all_points: D2s = self.__as_2d_pos()

        if alpha < 0 or hole_alpha < 0:
            raise ValueError(
                'The alpha value of an alpha shape may not be less than 0 as 0 already results in a convex hull '
                'which would be the largest alpha shape available'
            )
        elif max_area < 0:
            raise ValueError('Delaunay triangles can only be constructed if the area is valid (greater than 0)')

        hull_poly: Polygon | Any = alphashape(all_points, float(alpha))

        if type(hull_poly) != Polygon:
            raise ValueError(
                f'The provided alpha value {alpha} causes the resulting polygon to consist of multiple polygons '
                f'or a single point or line pointing to data loss, the alpha value may be lowered to prevent this'
            )

        hull_points: D2s = list(hull_poly.exterior.coords)
        hull_frames: EFs = self.__order_and_map_points(hull_points)
        self._hull: FrameGroup | None = FrameGroup(hull_frames)

        p_len: int = len(all_points)
        logger.debug(f'Hull consists of {len(hull_points)} points based on {len(all_points)} points')

        delaunay: dict = triangulate({'vertices': all_points}, opts=f'a{max_area}')
        vertices: D2s = delaunay['vertices']
        triangles: Is = delaunay['triangles']
        indices: Is = list(filter(
            lambda idxs: self.is_interior(Polygon(list(map(lambda i: vertices[i], idxs)))), filter(
                lambda idxs: not all(map(lambda i: i < p_len, idxs)), triangles
            )
        ))
        hole_points: D2s = list(map(lambda i: all_points[i], filter(lambda i: i < p_len, flat(indices))))

        if hole_points:
            self.__construct_holes(hole_points, hole_alpha)

        logger.debug(f'Detected {len(self._holes)} holes')
        return hull_frames

    @proc_time_log('Calculating maximum plausible depth...')
    def get_max_plausible_depth(self, low: float, sensitivity: float) -> float:
        depths: D1s = sorted(set(filter(lambda m: m > low, map(lambda f: f.water_depth_m, self.__frames))))
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
    def inspect(self, min_point: D2, max_point: D2) -> 'FrameGroup':
        min_x, min_y = min_point
        max_x, max_y = max_point

        if min_x >= max_x or min_y >= max_y:
            raise ValueError('Min cannot be smaller than max')

        return list(filter(lambda f: min_x <= f.latitude < max_x and min_y <= f.longitude < max_y, self.__frames))

    @proc_time_log('Flipping points...')
    @return_new_group
    def flip(self, axis: str) -> 'FrameGroup':
        if axis not in 'xy':
            raise ValueError('Axis has to be x, y or xy')

        clones: EFs = self.frames.copy()

        if axis == 'xy':
            any(map(lambda f: f.update_2d_pos((reversed(f.as_2d_pos()))), clones))
            return clones

        all_points: D2s = self.__as_2d_pos()
        xs, ys = list(zip(*all_points))
        max_x: float = max(xs)
        max_y: float = max(ys)

        [f.update_2d_pos((x, max_y - y) if axis == 'x' else (max_x - x, y)) for f, (x, y) in zip(clones, all_points)]
        return clones

    @proc_time_log('Interpolating...')
    def triangulate(self, shape: Optional['FrameGroup'] = None) -> Triangulation:
        return self.__triangulate(self.__as_2d_pos(), shape if shape else self.shape())

    @proc_time_log('Converting to 3D-printable...')
    def as_exportable(self, shape: Optional['FrameGroup'] = None, fill_holes: bool = True) -> list[ndarray]:
        if not shape:
            shape: FrameGroup = self.shape()

        hull_pts: D3s = list(map(lambda f: f.as_3d_pos(), shape.__frames))
        hole_pts: list[D3s] = list(map(lambda h: list(map(lambda f: f.as_3d_pos(), h.__frames)), shape._holes))
        cur_pts: D3s = hull_pts + list(flat(hole_pts))
        other_pts: D3s = list(filter(lambda xyz: xyz not in cur_pts, map(lambda f: f.as_3d_pos(), self.__frames)))

        max_z: float = max(list(flat(map(self.__get_zs, hole_pts))) + self.__get_zs(hull_pts + other_pts))

        hull_pts: D3s = self.__mod_zs(max_z, hull_pts)
        hole_pts: list[D3s] = list(map(lambda h: self.__mod_zs(max_z, h), hole_pts))
        other_pts: D3s = self.__mod_zs(max_z, other_pts)

        if fill_holes:
            top_hole_pts: list[D3s] = list(map(lambda h: self.__set_zs(h, max_z), hole_pts))
            top_hole_vectors: list[ndarray] = list(flat(map(lambda h: self.__triangulate_vectors(h), top_hole_pts)))
            hole_vectors: list[ndarray] = top_hole_vectors + list(flat(map(self.__build_walls, top_hole_pts)))
        else:
            hole_vectors: list[ndarray] = list(flat(map(self.__build_walls, hole_pts)))

        body_vectors: list[ndarray] = self.__triangulate_vectors(hull_pts + other_pts + list(flat(hole_pts)), shape)
        return self.__build_walls(hull_pts) + body_vectors + hole_vectors

    def get_max_keel_m(self) -> float:
        return max(map(lambda f: f.keel_depth_m, self.__frames))

    def is_interior(self, other: BaseGeometry) -> bool:
        holes: list[Polygon] = list(map(lambda h: Polygon(h.__as_2d_pos()), self._holes))
        shape: Polygon = Polygon(self._hull.__as_2d_pos())
        return other.within(shape) and not any(map(lambda h: other.within(h), holes))

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
    def __construct_hole_paths(points: D2s) -> Path:
        actions: list = [Path.MOVETO] + [Path.LINETO] * (len(points) - 2) + [Path.CLOSEPOLY]
        return Path(points, actions)

    @staticmethod
    def __normalize_row(min_x: float, min_y: float, scale: float, frame: ExtractedFrame) -> ExtractedFrame:
        frame.latitude = (frame.latitude - min_x) * scale
        frame.longitude = (frame.longitude - min_y) * scale
        return frame

    @staticmethod
    def __validate_param(param: float | str) -> None:
        if type(param) == str and param != auto:
            raise ValueError(f'Parameters may only be numeric if not set to "{auto}"')

    @staticmethod
    def __get_zs(xyzs) -> D1s:
        return [z for _, _, z in xyzs]

    @staticmethod
    def __mod_zs(max_z, xyzs) -> D3s:
        return [(x, y, max_z - z) for x, y, z in xyzs]

    @staticmethod
    def __set_zs(xyzs, value: float = 0) -> D3s:
        return [(x, y, value) for x, y, _ in xyzs]

    @staticmethod
    def __square_to_triangle(points: tuple[D3, D3, D3, D3]) -> list[ndarray]:
        p1, p2, p3, p4 = points
        return [array((p1, p2, p3)), array((p2, p3, p4))]

    @staticmethod
    def __build_walls(points: D3s) -> list[ndarray]:
        bottom_points: D3s = FrameGroup.__set_zs(points)
        wall_points: list[tuple[D3, D3, D3, D3]] = list(zip(
            points, points[1:] + points[:1], bottom_points, bottom_points[1:] + bottom_points[:1]
        ))
        return list(flat(map(FrameGroup.__square_to_triangle, wall_points)))

    @staticmethod
    def __triangulate(points: D2s, shape: Optional['FrameGroup'] = None) -> Triangulation:
        delaunay: dict = triangulate({'vertices': points}, opts='')
        vertices: D2s = delaunay['vertices']
        triangles: Is = delaunay['triangles']

        if shape:
            mask: list[bool] = list(map(
                lambda idxs: not shape.is_interior(Polygon(list(map(lambda i: (vertices[i]), idxs)))), triangles
            ))
        else:
            poly: Polygon = Polygon(points)
            mask: list[bool] = list(map(
                lambda idxs: not Polygon(list(map(lambda i: (vertices[i]), idxs))).within(poly), triangles
            ))

        return Triangulation(*list(zip(*vertices)), triangles=triangles, mask=mask)

    @staticmethod
    def __triangulate_vectors(points: D3s, shape: Optional['FrameGroup'] = None) -> list[ndarray]:
        vectors: Is = FrameGroup.__triangulate([(x, y) for x, y, _ in points], shape).get_masked_triangles()
        return list(map(lambda idxs: array(list(map(lambda i: points[i], idxs))), vectors))

    @proc_time_log('Constructing holes...')
    def __construct_holes(self, hole_points: D2s, hole_alpha: float) -> None:
        polys: MultiPolygon | Polygon | Any = alphashape(hole_points, float(hole_alpha))

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

    def __rebuild_vectors(self) -> None:
        self.__clear()
        any(map(self.__split_row, self.__frames))

    def __order_frames(self):
        self.__by_depths: EFs = sorted(self.__frames, key=lambda f: f.water_depth_m)

    def __order_and_map_points(self, ext_points: list[D2]) -> EFs:
        unordered_frames: EFs = list(filter(lambda f: f.as_2d_pos() in ext_points, self.__frames))
        return list(map(lambda pos: next((f for f in unordered_frames if f.as_2d_pos() == pos)), ext_points))

    def __as_2d_pos(self) -> list[D2]:
        return list(map(lambda f: f.as_2d_pos(), self.__frames))

    def __validate_vectors(self) -> None:
        cur_hash: int = hash(tuple(self.__frames))

        if cur_hash != self.__hash or not self.__xs:
            self.__rebuild_vectors()
            self.__hash: int = cur_hash

    def __clear(self) -> None:
        self.__xs.clear()
        self.__ys.clear()
        self.__zs.clear()
        self.__paths.clear()
        self.__by_depths.clear()

    def __split_row(self, frame: ExtractedFrame) -> None:
        self.__xs.append(frame.latitude)
        self.__ys.append(frame.longitude)
        self.__zs.append(frame.water_depth_m)

    def __repr__(self) -> str:
        return '[{}]'.format('\n'.join(map(repr, self.__frames)))
