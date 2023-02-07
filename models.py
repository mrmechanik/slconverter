import logging
import math
import os
import time
from functools import wraps
from typing import Any, Callable, Optional, Type, TypeVar, Iterator

from alphashape import alphashape
from matplotlib.path import Path
from matplotlib.tri import Triangulation
from numpy import ndarray, array, zeros
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.base import BaseGeometry
from sllib import Frame
from sllib.definitions import FEET_CONVERSION
from stl import Mesh
from triangle import triangulate
from trimesh import Trimesh, load
from trimesh.repair import broken_faces
from trimesh.smoothing import filter_laplacian

logger: logging.Logger = logging.getLogger(__name__)
auto: str = 'auto'

default_sensitivity: float = 0.07
default_alpha: float = 8
default_hole_alpha: float = 1.5
default_max_area: float = 0.04
default_power: float = 0.1
default_iterations: int = 100
default_buffer: float = 0.0005
default_scale: float = 1

T: TypeVar = TypeVar('T')
D1s: Type = list[float]
D2: Type = tuple[float, float]
D2s: Type = list[D2]
D3: Type = tuple[float, float, float]
D3s: Type = list[D3]

I3s: Type = list[tuple[int, int, int]]
D3T: Type = tuple[D3, D3, D3]
Ns: Type = list[ndarray]
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

    def update_2d_pos(self, point: D2) -> None:
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

        self._hull: Optional[FrameGroup] = None
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
            logger.debug(f'Created {len(self.__paths)} paths representing the holes')

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
            self,
            alpha: float = default_alpha,
            max_area: float = default_max_area,
            hole_alpha: float = default_hole_alpha
    ) -> 'FrameGroup':
        points: D2s = self.__as_2d_pos()

        if alpha < 0 or hole_alpha < 0:
            raise ValueError(
                f'The alpha value of an alpha shape may not be less than 0 as 0 already results in a convex hull which '
                f'would be the largest alpha shape available, provided was "{alpha}" and for holes "{hole_alpha}"'
            )
        elif max_area < 0:
            raise ValueError(
                f'Delaunay triangles can only be constructed if the area is valid (greater than 0), not "{max_area}"'
            )

        hull_points, hole_points = self.__shape(points, alpha, max_area, hole_alpha)
        hull_frames: EFs = self.__order_and_map_points(hull_points)
        self._hull: FrameGroup | None = FrameGroup(hull_frames)
        self._holes: list[FrameGroup] = list(map(lambda p: FrameGroup(self.__order_and_map_points(p)), hole_points))
        logger.debug(f'Hull consists of {len(hull_points)} points based on {len(points)} points')
        logger.debug(f'Detected {len(self._holes)} holes')
        return hull_frames

    @proc_time_log('Calculating maximum plausible depth...')
    def get_max_plausible_depth(self, low: float, sensitivity: float) -> float:
        depths: D1s = sorted(set(filter(lambda m: m > low, map(lambda f: f.water_depth_m, self.__frames))))
        max_depth = next((cur for cur, fut in zip(depths, depths[1:]) if fut - cur >= sensitivity), depths[-1])
        logger.debug(f'Detected max plausible depth "{max_depth}" for sensitivity "{sensitivity}"')
        return max_depth

    def filter_min(self, min_border: float | str = auto) -> 'FrameGroup':
        return self.filter(min_border, 20000)

    def filter_max(self, max_border: float | str = auto, sensitivity: float = default_sensitivity) -> 'FrameGroup':
        return self.filter(0, max_border, sensitivity)

    @proc_time_log('Filtering frames...')
    @return_new_group
    def filter(
            self,
            min_border: float | str = auto,
            max_border: float | str = auto,
            sensitivity_for_max: float = default_sensitivity
    ) -> 'FrameGroup':
        self.__validate_param(min_border)
        self.__validate_param(max_border)

        low: float = self.get_max_keel_m() if min_border == auto else min_border
        high: float = self.get_max_plausible_depth(low, sensitivity_for_max) if max_border == auto else max_border

        if low >= high:
            raise ValueError(f'No outliers detectable for LOW "{low}" and HIGH "{high}"')

        filtered_frames: EFs = list(filter(lambda f: low < f.water_depth_m <= high, self.__frames))
        logger.debug(
            f'Using "{low}" and "{high}" as border values caused {len(self.__frames) - len(filtered_frames)} to be '
            f'removed'
        )
        return filtered_frames

    @proc_time_log('Removing duplicates...')
    @return_new_group
    def uniquify(self) -> 'FrameGroup':
        ordered_uniques: EFs = sorted(set(self.__frames), key=lambda f: f.timestamp)
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

        logger.debug(f'Using x "{min_x}" and y "{min_y}" as minimal values and "{scale}" as scale factor')
        return list(map(lambda f: self.__normalize_row(min_x, min_y, scale, f), self.__frames))

    @proc_time_log('Filtering values...')
    @return_new_group
    def inspect(self, min_point: D2, max_point: D2) -> 'FrameGroup':
        min_x, min_y = min_point
        max_x, max_y = max_point

        if min_x >= max_x or min_y >= max_y:
            raise ValueError(
                f'Min cannot be smaller than max, provided xs "{min_x}" - "{max_x}" and ys "{min_y}" - "{max_y}"'
            )

        inspected_frames: EFs = list(filter(
            lambda f: min_x <= f.latitude < max_x and min_y <= f.longitude < max_y, self.__frames
        ))
        logger.debug(f'Inspection reduced frame count from {len(self.__frames)} to {len(inspected_frames)}')
        return inspected_frames

    @proc_time_log('Flipping points...')
    @return_new_group
    def flip(self, axis: str) -> 'FrameGroup':
        if axis not in 'xy':
            raise ValueError(f'Axis has to be x, y or xy, not "{axis}"')

        clones: EFs = self.frames.copy()

        if axis == 'xy':
            any(map(lambda f: f.update_2d_pos((reversed(f.as_2d_pos()))), clones))
            return clones

        all_points: D2s = self.__as_2d_pos()
        xs, ys = list(zip(*all_points))
        max_x: float = max(xs)
        max_y: float = max(ys)

        [f.update_2d_pos((x, max_y - y) if axis == 'x' else (max_x - x, y)) for f, (x, y) in zip(clones, all_points)]
        logger.debug(f'Flipped {len(clones)} frames on the "{axis}" axis')
        return clones

    @proc_time_log('Triangulating...')
    def triangulate(self, shape: Optional['FrameGroup'] = None) -> Triangulation:
        triang: Triangulation = self.__triangulate(self.__as_2d_pos(), shape if shape else self.shape())
        logger.debug(f'Triangulation resulted in {len(triang.triangles)} triangles')
        return triang

    @proc_time_log('Converting to 3D-printable...')
    def as_exportable(
            self,
            shape: Optional['FrameGroup'] = None,
            scale: float | D3 = default_scale,
            z_buffer: float = default_buffer,
            smooth_power: float = default_power,
            smooth_iterations: int = default_iterations,
            fill_holes: bool = False,
            keep_tmp: bool = False
    ) -> Trimesh:
        self.__validate_param(smooth_power)
        self.__validate_param(smooth_iterations)

        if not 0 <= smooth_power <= 1:
            raise ValueError(f'The smoothing power has to be between 0 and 1 inclusive, not "{smooth_power}"')

        if smooth_iterations < 0:
            raise ValueError(f'The iterations have to be positive, not "{smooth_iterations}"')

        if not shape:
            shape: FrameGroup = self.shape()

        points: D3s = self.__as_3d_pos()
        max_z: float = max(self.__get_zs(points))
        hole_vectors: Ns = []

        if fill_holes:
            hole_vectors: Ns = list(flat(map(
                lambda h: self.__triangulate_vectors(self.__set_zs(h.__as_3d_pos(), max_z)), self._holes
            )))
            shape._holes = []

        lid_vectors: Ns = self.__triangulate_vectors(self.__mod_zs(points, max_z), shape)

        temps: list[str] = [f'data/temp{num}.stl' for num in range(1, 3)]
        temp1, temp2 = temps
        self.__mesh_from_vectors(lid_vectors).save(temp1)

        lid_mesh: Trimesh = load(temp1)
        filter_laplacian(lid_mesh, lamb=smooth_power, iterations=smooth_iterations)
        logger.debug(f'Smoothing reduced vector count from {len(lid_vectors)} to {len(lid_mesh.vertices)}')

        min_z: float = min((z for _, _, z in lid_mesh.vertices)) - z_buffer
        vectors: Ns = list(flat(map(
            lambda tri: self.__build_prism_from_tri(tri, - min_z), list(lid_mesh.triangles) + list(hole_vectors)
        )))

        if type(scale) == float or type(scale) == int:
            scale = (scale, scale, scale)

        if scale != (1, 1, 1):
            logger.debug(f'Scaling {len(vectors)} vectors by {scale}')
            vectors: Ns = list(map(lambda t: array(list(map(lambda v: v * scale, t))), vectors))

        self.__mesh_from_vectors(vectors).save(temp2)
        mesh: Trimesh = load(temp2)

        if not keep_tmp:
            any(map(lambda t: os.remove(t), temps))

        old_broken: int = len(broken_faces(mesh))
        mesh.fill_holes()
        logger.debug('Repaired {} broken faces, mesh is {}watertight'.format(
            old_broken - len(broken_faces(mesh)), '' if mesh.is_watertight else 'not '
        ))
        logger.debug(f'Final mesh consists of {len(mesh.vertices)} vertices')
        return mesh

    def get_max_keel_m(self) -> float:
        return max(map(lambda f: f.keel_depth_m, self.__frames))

    def is_interior(self, other: BaseGeometry) -> bool:
        holes: list[Polygon] = list(map(lambda h: Polygon(h.__as_2d_pos()), self._holes))
        shape: Polygon = Polygon(self._hull.__as_2d_pos())
        return other.within(shape) and not any(map(lambda h: other.within(h), holes))

    @staticmethod
    @proc_time_log('Reducing to list & updating timestamps...')
    def __convert_frame_dict(frames: dict[str, EFs]) -> EFs:
        frames_stack: list[EFs] = list(frames.values())
        stamps: list[int] = list(map(lambda f: f.timestamp, flat(frames_stack)))
        offset: int = max(stamps) - min(stamps)
        return [FrameGroup.__update_and_return(f, idx * offset) for idx, fs in enumerate(frames_stack) for f in fs]

    @staticmethod
    def __construct_hole_paths(points: D2s) -> Path:
        return Path(points, [Path.MOVETO] + [Path.LINETO] * (len(points) - 2) + [Path.CLOSEPOLY])

    @staticmethod
    def __normalize_row(min_x: float, min_y: float, scale: float, frame: ExtractedFrame) -> ExtractedFrame:
        frame.latitude = (frame.latitude - min_x) * scale
        frame.longitude = (frame.longitude - min_y) * scale
        return frame

    @staticmethod
    def __update_and_return(frame: ExtractedFrame, offset: int):
        frame.timestamp += offset
        return frame

    @staticmethod
    def __shape(
            points: D2s,
            alpha: float = default_alpha,
            max_area: float = default_max_area,
            hole_alpha: float = default_hole_alpha
    ) -> tuple[D2s, list[D2s]]:
        hull_poly: Polygon | Any = alphashape(points, float(alpha))

        if type(hull_poly) != Polygon:
            raise ValueError(
                f'The provided alpha value "{alpha}" causes the resulting polygon to consist of multiple polygons '
                f'or a single point or line pointing to data loss, the alpha value may be lowered to prevent this'
            )

        hull_points: D2s = list(hull_poly.exterior.coords)
        p_len: int = len(points)

        delaunay: dict = triangulate({'vertices': points}, opts=f'a{max_area}')
        vertices: D2s = delaunay['vertices']
        triangles: I3s = delaunay['triangles']
        indices: I3s = list(filter(
            lambda idxs: hull_poly.contains(Polygon(list(map(lambda i: vertices[i], idxs)))), filter(
                lambda idxs: not all(map(lambda i: i < p_len, idxs)), triangles
            )
        ))
        stacked_hole_points: D2s = list(map(lambda i: points[i], filter(lambda i: i < p_len, flat(indices))))
        return hull_points, FrameGroup.__get_holes(stacked_hole_points, hole_alpha) if stacked_hole_points else []

    @staticmethod
    def __validate_param(param: float | str) -> None:
        if type(param) == str and param != auto:
            raise ValueError(f'Parameters may only be numeric if not set to "{auto}", provided "{param}"')

    @staticmethod
    def __get_zs(xyzs) -> D1s:
        return [z for _, _, z in xyzs]

    @staticmethod
    def __mod_zs(xyzs, mod) -> D3s:
        return [(x, y, mod - z) for x, y, z in xyzs]

    @staticmethod
    def __set_zs(xyzs, new_z) -> D3s:
        return [(x, y, new_z) for x, y, z in xyzs]

    @staticmethod
    def __build_prism_from_tri(tri: ndarray, z_mod: float = 0) -> Ns:
        p1, p2, p3 = [(x, y, z + z_mod) for x, y, z in tri]
        floor_tri = [(x, y, 0) for x, y, _ in tri]
        p1f, p2f, p3f = floor_tri
        v12: Ns = FrameGroup.__square_to_triangle((p1, p2, p1f, p2f))
        v13: Ns = FrameGroup.__square_to_triangle((p1, p3, p1f, p3f))
        v23: Ns = FrameGroup.__square_to_triangle((p2, p3, p2f, p3f))
        return [array((p1, p2, p3))] + [array(floor_tri)] + v12 + v13 + v23

    @staticmethod
    def __mesh_from_vectors(vectors: Ns) -> Mesh:
        data: ndarray = zeros(len(vectors), dtype=Mesh.dtype)
        data['vectors']: list[ndarray] = vectors
        return Mesh(data, remove_empty_areas=True)

    @staticmethod
    def __square_to_triangle(points: tuple[D3, D3, D3, D3]) -> Ns:
        p1, p2, p3, p4 = points
        z1, z2, z3, z4 = FrameGroup.__get_zs(points)
        is_smaller: bool = math.fabs(z1 - z4) < math.fabs(z2 - z3)
        return [array((p1, p2, p3)), array((p2, p3, p4))] if is_smaller else [array((p1, p3, p4)), array((p1, p2, p4))]

    @staticmethod
    def __triangulate(points: D2s, shape: Optional['FrameGroup'] = None) -> Triangulation:
        delaunay: dict = triangulate({'vertices': points}, opts='')
        vertices: D2s = delaunay['vertices']
        triangles: I3s = delaunay['triangles']
        mask: Optional[list[bool]] = None

        if shape:
            mask: list[bool] = list(map(
                lambda idxs: not shape.is_interior(Polygon(list(map(lambda i: (vertices[i]), idxs)))), triangles
            ))

        return Triangulation(*list(zip(*vertices)), triangles=triangles, mask=mask)

    @staticmethod
    def __triangulate_vectors(points: D3s, shape: Optional['FrameGroup'] = None) -> Ns:
        vectors: I3s = FrameGroup.__triangulate([(x, y) for x, y, _ in points], shape).get_masked_triangles()
        return list(map(lambda idxs: array(list(map(lambda i: points[i], idxs))), vectors))

    @staticmethod
    def __get_holes(hole_points: D2s, hole_alpha: float) -> list[D2s]:
        polys: MultiPolygon | Polygon | Any = alphashape(hole_points, float(hole_alpha))

        if type(polys) != MultiPolygon and type(polys) != Polygon:
            raise ValueError(
                f'The hole alpha "{hole_alpha}" value causes the holes to be too imprecise, the alpha value may be '
                'lowered to prevent this'
            )

        if type(polys) == Polygon:
            polys = [polys]
        elif type(polys) == MultiPolygon:
            polys = polys.geoms

        return list(map(lambda p: list(reversed(p.exterior.coords)), polys))

    def __rebuild_vectors(self) -> None:
        self.__clear()
        any(map(self.__split_row, self.__frames))

    def __order_frames(self):
        self.__by_depths: EFs = sorted(self.__frames, key=lambda f: f.water_depth_m)

    def __order_and_map_points(self, ext_points: D2s) -> EFs:
        unordered_frames: EFs = list(filter(lambda f: f.as_2d_pos() in ext_points, self.__frames))
        return list(map(lambda pos: next((f for f in unordered_frames if f.as_2d_pos() == pos)), ext_points))

    def __as_2d_pos(self) -> D2s:
        return list(map(lambda f: f.as_2d_pos(), self.__frames))

    def __as_3d_pos(self) -> D3s:
        return list(map(lambda f: f.as_3d_pos(), self.__frames))

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
