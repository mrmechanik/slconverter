import logging
import math
import os
import time
from collections import Counter
from functools import wraps
from typing import Any, Callable, Optional, Type, TypeVar, Iterator

from alphashape import alphashape
from matplotlib.path import Path
from matplotlib.tri import Triangulation
from numpy import ndarray, array, zeros, cross
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, MultiLineString, LinearRing
from shapely.geometry.base import BaseGeometry
from sllib import Frame
from sllib.definitions import FEET_CONVERSION
from stl import Mesh
from triangle import triangulate
from trimesh import Trimesh, load
from trimesh.smoothing import filter_laplacian

logger: logging.Logger = logging.getLogger(__name__)
auto: str = 'auto'

# define default values
default_sensitivity: float = 0.07
default_alpha: float = 8
default_hole_alpha: float = 1.5
default_max_area: float = 0.04
default_power: float = 0.1
default_iterations: int = 100
default_buffer: float = 0.0005
default_scale: float = 1

# define custom types
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
    """
    measure and log the execution time for the decorated function

    :param msg: the initial message to display
    :return: the decorated function
    """

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
    """
    flattens any iterable to a lower dimension

    :param iterable: the n-dimensional iterable
    :return: the (n - 1)-dimensional iterable
    """
    return (elem for sub_iterable in iterable for elem in sub_iterable)


def return_new_group(
        func: Callable[['FrameGroup', Optional[Any]], EFs]
) -> Callable[['FrameGroup', Optional[Any]], 'FrameGroup']:
    """
    constructs a new frame group from the result of the decorated function

    :param func: the function that returns frames to be converted
    :return: the decorated function returning a frame group of the resulting frames
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs) -> 'FrameGroup':
        res: EFs = func(self, *args, **kwargs)
        group: FrameGroup = FrameGroup(res)
        # copy the hull and the holes over as a compensation for method chaining
        group._hull = self._hull
        group._holes = self._holes
        return group

    return wrapper


class ExtractedFrame:
    """
    a simplified frame consisting of:
    - keel depth (in meters)
    - water depth (in meters)
    - latitude
    - longitude
    - timestamp
    """

    def __init__(self, frame: Frame) -> None:
        self.keel_depth_m: float = frame.keel_depth * FEET_CONVERSION if hasattr(frame, 'keel_depth') else 0
        self.water_depth_m: float = frame.water_depth_m
        self.latitude: float = frame.latitude
        self.longitude: float = frame.longitude
        self.timestamp: int = frame.time1

    def as_2d_pos(self) -> D2:
        """
        a quick access function to get only the 2d representation of the frame

        :return: a tuple with the latitude and the longitude of the frame
        """
        return self.latitude, self.longitude

    def as_3d_pos(self) -> tuple[float, float, float]:
        """
        a quick access function to get only the 3d representation of the frame

        :return: a tuple with the latitude, the longitude and the water depth of the frame
        """
        return self.latitude, self.longitude, self.water_depth_m

    def update_2d_pos(self, point: D2) -> None:
        """
        a quick access function to update the 2d representation

        :param point: the new latitude and longitude as a tuple to update the old values
        """
        self.latitude, self.longitude = point

    def __as_data_tuple(self) -> tuple[float, float, float, float]:
        """
        a method to retrieve all relevant information of the frame (excluding the timestamp)

        :return: a tuple with the keel depth, the water depth, the latitude and the longitude
        """
        return self.keel_depth_m, self.water_depth_m, self.latitude, self.longitude

    def __eq__(self, other) -> bool:
        return isinstance(other, self.__class__) and self.__as_data_tuple() == other.__as_data_tuple()

    def __hash__(self) -> int:
        return hash(self.__as_data_tuple())

    def __repr__(self) -> str:
        return '{{{:.6f} {:.6f} {:.2f}}}'.format(self.latitude, self.longitude, self.water_depth_m)


class FrameGroup:
    """
    a collection of frames and optionally the hull and holes of the frames
    """

    def __init__(self, frames: EFs | dict[str, EFs]) -> None:
        self.__xs: D1s = []
        self.__ys: D1s = []
        self.__zs: D1s = []
        self.__paths: list[Path] = []
        self.__by_depths: EFs = []

        self._hull: Optional[FrameGroup] = None
        self._holes: list[FrameGroup] = []
        # handle both cases to construct a frame group
        self.__frames: EFs = self.__convert_frame_dict(frames) if type(frames) == dict else frames
        self.__hash: int = hash(tuple(frames))

    @property
    def frames(self) -> EFs:
        """
        a property returning all frames of the frame group

        :return: a list of copied frames
        """
        return self.__frames.copy()

    @property
    def hole_paths(self) -> list[Path]:
        """
        a property converting all holes to closed paths

        :return: a list of paths representing the holes of the frame group
        """
        if not self.__paths:
            self.__paths: list[Path] = list(map(lambda h: self.__construct_hole_paths(h.__as_2d_pos()), self._holes))
            logger.debug(f'Created {len(self.__paths)} paths representing the holes')

        return self.__paths.copy()

    @property
    def xs(self) -> D1s:
        """
        a property to access a copy of all x (latitude) values of all frames in the frame group

        :return: a list of all x (latitude) values
        """
        self.__validate_vectors()
        return self.__xs.copy()

    @property
    def ys(self) -> D1s:
        """
        a property to access a copy of all y (longitude) values of all frames in the frame group

        :return: a list of all y (longitude) values
        """
        self.__validate_vectors()
        return self.__ys.copy()

    @property
    def zs(self) -> D1s:
        """
        a property to access a copy of all z (water depth) values of all frames in the frame group

        :return: a list of all z (water depth) values
        """
        self.__validate_vectors()
        return self.__zs.copy()

    @property
    def deepest(self) -> ExtractedFrame:
        """
        a property to evaluate the deepest point in the frames of the frame group

        :return: a copy of the frame with the deepest water depth
        """
        if not self.__by_depths:
            self.__order_frames()

        return self.__by_depths.copy()[0]

    @property
    def shallowest(self) -> ExtractedFrame:
        """
        a property to evaluate the shallowest point in the frames of the frame group

        :return: a copy of the frame with the shallowest water depth
        """
        if not self.__by_depths:
            self.__order_frames()

        return self.__by_depths.copy()[- 1]

    @proc_time_log('Constructing shape...')
    @return_new_group
    def shape(
            self,
            alpha: float = default_alpha,
            max_area: float = default_max_area,
            hole_alpha: float = default_hole_alpha
    ) -> 'FrameGroup':
        """
        calculates the shape including holes from the contained frames in this frame group

        :param alpha: the value for the alpha shapes constructing the hull of the frame group
        :param max_area: the maximum area of a delaunay triangulated triangle
        :param hole_alpha: the value for the alpha shapes constructing the holes based on the delaunay triangles
        :raises: ValueError if the alpha values are less than 0 or if the max area is smaller or equal to 0
        :return: a new frame group consisting of the hull (and the hull and holes set)
        """
        points: D2s = self.__as_2d_pos()

        if alpha < 0 or hole_alpha < 0:
            raise ValueError(
                f'The alpha value of an alpha shape may not be less than 0 as 0 already results in a convex hull which '
                f'would be the largest alpha shape available, provided was "{alpha}" and for holes "{hole_alpha}"'
            )
        elif max_area <= 0:
            raise ValueError(
                f'Delaunay triangles can only be constructed if the area is valid (greater than 0), not "{max_area}"'
            )

        # extract the hull and hole points
        hull_points, hole_points = self.__shape(points, alpha, max_area, hole_alpha)
        # reorder the frames based on the order of the found points of the hull and the holes
        hull_frames: EFs = self.__order_and_map_points(hull_points)
        self._hull: FrameGroup | None = FrameGroup(hull_frames)
        self._holes: list[FrameGroup] = list(map(lambda p: FrameGroup(self.__order_and_map_points(p)), hole_points))
        logger.debug(f'Hull consists of {len(hull_points)} points based on {len(points)} points')
        logger.debug(f'Detected {len(self._holes)} holes')
        return hull_frames

    @proc_time_log('Calculating maximum plausible depth...')
    def get_max_plausible_depth(self, low: float, sensitivity: float) -> float:
        """
        detects a plausible depth based on the sensitivity

        :param low: the lower boundary which should be ignored for the detection
        :param sensitivity: the maximum value between two depths
        :return: a plausible depth
        """
        # extract only depths above the lower boundary, order them from low to high
        depths: D1s = sorted(set(filter(lambda d: d > low, map(lambda f: f.water_depth_m, self.__frames))))
        # find the two depths that exceed the sensitivity value
        max_depth = next((cur for cur, fut in zip(depths, depths[1:]) if fut - cur >= sensitivity), depths[- 1])
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
        """
        filters the frame group for invalid values

        :param min_border: the lower boundary for the frame values to be filtered
        :param max_border: the upper boundary for the frame values to be filtered
        :param sensitivity_for_max: the sensitivity for the upper boundary
        :raises: ValueError if the parameters are of type string but not 'auto' or if the lower exceeds higher bounds
        :return: the filtered frame group
        """
        self.__validate_param(min_border)
        self.__validate_param(max_border)

        # use default values if auto is set
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
        """
        removes all duplicates from the frame group and sorts the remains by the timestamp

        :return: the duplication free frame group
        """
        ordered_uniques: EFs = sorted(set(self.__frames), key=lambda f: f.timestamp)
        logger.debug(f'Reduced {len(self.__frames)} frames to {len(ordered_uniques)} frames')
        return ordered_uniques

    @proc_time_log('Normalizing...')
    @return_new_group
    def normalize(
            self, min_x: float | str = auto, min_y: float | str = auto, scale: float | str = auto
    ) -> 'FrameGroup':
        """
        normalizes the frame group

        :param min_x: the minimal x (latitude) value
        :param min_y: the minimal y (longitude) value
        :param scale: the scale factor the values should be normalized to
        :raises: ValueError if the parameters are of type string and not 'auto'
        :return: the normalized frame group
        """
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

            # use the maximum difference to not stretch the data points
            factor: float = max(diff_x, diff_y)
            scale: int = 1

            # multiply by 10 as long as the factor is less than 1
            while scale * factor < 1:
                scale *= 10

        logger.debug(f'Using x "{min_x}" and y "{min_y}" as minimal values and "{scale}" as scale factor')
        return list(map(lambda f: self.__normalize_row(min_x, min_y, scale, f), self.__frames))

    @proc_time_log('Filtering values...')
    @return_new_group
    def inspect(self, min_point: D2, max_point: D2) -> 'FrameGroup':
        """
        inspects a part of the frame group

        :param min_point: the 'bottom left' point consisting of the lowest x and y value
        :param max_point: the 'top right' point consisting of the highest x any y value
        :return: the frame group delimited to the min and max point
        """
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
        """
        flips the frame group based on the provided axis

        :param axis: a string representing the axis
        :return: the flipped frame group
        """
        if axis not in 'xy':
            raise ValueError(f'Axis has to be x, y or xy, not "{axis}"')

        clones: EFs = self.frames.copy()

        if axis == 'xy':
            any(map(lambda f: f.update_2d_pos(reversed(f.as_2d_pos())), clones))
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
        """
        triangulate the frame group

        :param shape: the shape with the hull and holes used for the triangulation
        :return: a triangulation respecting the shape of the frame group
        """
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
        """
        constructs a 3d model based on the shape and the frames of the frame group

        :param shape: the shape of the frame group including the hull and the holes
        :param scale: the scaling factor to increase or decrease the model by
        :param z_buffer: the buffer to have between the lowest point and the ground
        :param smooth_power: the smoothing power of the laplacian smoothing
        :param smooth_iterations: the amount of iterations to perform the laplacian smoothing
        :param fill_holes: whether the holes of the frame group should be filled or not
        :param keep_tmp: whether the temporary processing files should be kept
        :raises: ValueError if the smoothing power is not between 0 and 1 or if the iterations are negative
        :return: a trimesh representing the 3d model
        """
        if not 0 <= smooth_power <= 1:
            raise ValueError(f'The smoothing power has to be between 0 and 1 inclusive, not "{smooth_power}"')

        if smooth_iterations < 0:
            raise ValueError(f'The iterations have to be positive, not "{smooth_iterations}"')

        if not shape:
            shape: FrameGroup = self.shape()

        points: D3s = self.__as_3d_pos()
        max_z: float = max(self.__get_zs(points))

        shape._holes = []
        # flip the points upside down to represent the actual model heights, then construct the mesh
        lid_vectors: Ns = self.__triangulate_vectors(self.__mod_zs(points, max_z), shape)
        shape._holes = self._holes

        temps: list[str] = [f'data/temp{num}.stl' for num in range(1, 3)]
        temp1, temp2 = temps
        # save the mesh into the first temporary file
        self.__mesh_from_vectors(lid_vectors).save(temp1)

        lid_mesh: Trimesh = load(temp1)
        filter_laplacian(lid_mesh, lamb=smooth_power, iterations=smooth_iterations)
        logger.debug(f'Smoothing reduced vector count from {len(lid_vectors)} to {len(lid_mesh.vertices)}')

        # calculate the minimal z value (including the buffer) to not lose the lowest point
        min_z: float = min((z for _, _, z in lid_mesh.vertices)) - z_buffer
        dirty_ceil_tris: Ns = list(map(lambda tri: array(self.__mod_zs(tri, - min_z, True)), lid_mesh.triangles))
        dirty_ceil_poly: Polygon = self.__unite(list(map(lambda tri: tri[:, :2], dirty_ceil_tris)))
        # re-triangulate the smoothed surfaces to exclude errors from overlaying triangles
        part_ceil_tris: Ns = list(filter(
            lambda tri: dirty_ceil_poly.covers(Polygon(tri[:, :2])),
            self.__triangulate_vectors(list(flat(map(lambda tri: tuple(map(tuple, tri)), dirty_ceil_tris))))
        ))
        # only extract triangles that are not within any hole
        ceil_tris: Ns = list(filter(
            lambda tri: not any(map(lambda h: Polygon(tri[:, :2]).within(Polygon(h.__as_2d_pos())), shape._holes)),
            part_ceil_tris
        ))

        # extract the triangles intersecting any hole
        inter_tris: list[Ns] = list(map(lambda h: list(filter(
            lambda tri: Polygon(tri[:, :2]).intersects(Polygon(h.__as_2d_pos())), ceil_tris
        )), shape._holes))
        # calculate intersection points from the intersecting triangles and the hole
        hole_points: list[list[D3s]] = [
            list(map(lambda tri: self.__calc_hole_points(tri, hole.__as_2d_pos()), tris)) for tris, hole in
            zip(inter_tris, shape._holes)
        ]
        # join the intersecting triangles with the intersection points to create triangles touching the respective hole
        fixed_hole_tris: list[Ns] = [list(flat((
            self.__fix_tri(tri, fixed, Polygon(hole.__as_2d_pos())) for tri, fixed in zip(tris, fixes))
        )) for tris, fixes, hole in zip(inter_tris, hole_points, shape._holes)]

        # take all triangles that do not touch or intersect with any hole
        ceil_mesh: Ns = list(filter(
            lambda tri: not any(map(lambda h: Polygon(tri[:, :2]).intersects(Polygon(h.__as_2d_pos())), shape._holes)),
            part_ceil_tris
        )) + list(flat(fixed_hole_tris))
        floor_mesh: Ns = list(map(self.__build_floor_tri, part_ceil_tris if fill_holes else ceil_mesh))
        wall_mesh: Ns = self.__tris_to_walls(part_ceil_tris if fill_holes else ceil_mesh)

        ceil_poly: Polygon = self.__unite(ceil_mesh)
        floor_poly: Polygon = self.__unite(list(map(lambda tri: tri[:, :2], floor_mesh)))
        # join the fixed triangles with the intersecting triangles, triangulate them & keep the touching triangles
        # inside the smoothed shape
        hole_tris: list[Ns] = [list(filter(
            lambda tri: floor_poly.covers(Polygon(tri[:, :2])) and not ceil_poly.covers(Polygon(tri[:, :2])),
            self.__triangulate_vectors(list(flat(tris)) + list(flat(part_ceil_tris)))
        )) for tris, hole in zip(fixed_hole_tris, shape._holes)]
        hole_wall_mesh: Ns = list(flat(map(lambda h_tris: self.__tris_to_walls(h_tris, max_z - min_z), hole_tris)))
        hole_ceil_mesh: Ns = list(flat(map(
            lambda h_tris: array(list(map(lambda tri: self.__set_zs(tri, max_z - min_z), h_tris))), hole_tris
        )))

        # join all meshes
        final_hole_mesh: Ns = list(hole_ceil_mesh) + hole_wall_mesh
        final_mesh: Ns = list(ceil_mesh) + wall_mesh + floor_mesh + (final_hole_mesh if fill_holes else [])

        if type(scale) == float or type(scale) == int:
            scale: D3 = (scale, scale, scale)

        if scale != (1, 1, 1):
            logger.debug(f'Scaling {len(final_mesh)} vectors by {scale}')
            final_mesh: Ns = list(map(lambda t: array(list(map(lambda v: v * scale, t))), final_mesh))

        self.__mesh_from_vectors(final_mesh).save(temp2)
        mesh: Trimesh = load(temp2)

        if not keep_tmp:
            any(map(lambda t: os.remove(t), temps))

        # try fixing the holes
        mesh.fill_holes()

        return mesh

    def get_max_keel_m(self) -> float:
        return max(map(lambda f: f.keel_depth_m, self.__frames))

    def is_interior(self, other: BaseGeometry) -> bool:
        """
        checks if a given geometry lies within the interior of the hull but outside any hole

        :param other: the other geometry to be tested
        :return: true if it is inside the hull but outside any hole, false otherwise
        """
        holes: list[Polygon] = list(map(lambda h: Polygon(h.__as_2d_pos()), self._holes))
        shape: Polygon = Polygon(self._hull.__as_2d_pos())
        return other.within(shape) and not any(map(lambda h: other.within(h), holes))

    @staticmethod
    @proc_time_log('Reducing to list & updating timestamps...')
    def __convert_frame_dict(frames: dict[str, EFs]) -> EFs:
        """
        converts separate frames to a list of frames with corrected timestamps

        :param frames: a dictionary of frames split into their origin file
        :return: a list of frames with fixed timestamps
        """
        frames_stack: list[EFs] = list(frames.values())
        stamps: list[int] = list(map(lambda f: f.timestamp, flat(frames_stack)))
        offset: int = max(stamps) - min(stamps)
        return [FrameGroup.__update_and_return(f, idx * offset) for idx, fs in enumerate(frames_stack) for f in fs]

    @staticmethod
    def __tri_to_lines(tri: ndarray) -> list[tuple[D3, D3]]:
        """
        converts a 2d triangle to its lines

        :param tri: the 2d triangle to convert
        :return: a list of tuples with points that form a line of the triangle
        """
        p1, p2, p3 = sorted(map(tuple, tri))
        return [(p1, p2), (p1, p3), (p2, p3)]

    @staticmethod
    def __construct_hole_paths(points: D2s) -> Path:
        """
        constructs a closed path based on the provided points

        :param points: the 2d points
        :return: a closed polygon path
        """
        return Path(points, [Path.MOVETO] + [Path.LINETO] * (len(points) - 2) + [Path.CLOSEPOLY])

    @staticmethod
    def __normalize_row(min_x: float, min_y: float, scale: float, frame: ExtractedFrame) -> ExtractedFrame:
        """
        normalizes a singular row

        :param min_x: the minimal x (latitude) value
        :param min_y: the minimal y (longitude) value
        :param scale: the precision scaling factor
        :param frame: the frame to be modified
        :return: the normalized frame
        """
        frame.latitude = (frame.latitude - min_x) * scale
        frame.longitude = (frame.longitude - min_y) * scale
        return frame

    @staticmethod
    def __update_and_return(frame: ExtractedFrame, offset: int):
        """
        fixes the timestamp of a frame

        :param frame: the frame to be updated
        :param offset: the offset that should be added to the timestamp of the provided frame
        :return: the frame with the updated timestamp
        """
        frame.timestamp += offset
        return frame

    @staticmethod
    def __shape(
            points: D2s,
            alpha: float = default_alpha,
            max_area: float = default_max_area,
            hole_alpha: float = default_hole_alpha
    ) -> tuple[D2s, list[D2s]]:
        """
        constructs the shape with the hull and holes

        :param points: the points to create the hull and holes from
        :param alpha: the value for the alpha shapes constructing the hull of the frame group
        :param max_area: the maximum area of a delaunay triangulated triangle
        :param hole_alpha: the value for the alpha shapes constructing the holes based on the delaunay triangles
        :raises: ValueError if the alpha shape forms anything other than a polygon
        :return: a tuple of points representing the hull and a list containing lists of all hole points
        """
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
        # check the vectors that have been created due to the maximum area limitation
        # if said new vector is inside the hull and the other two vectors are not both new as well, then it must belong
        # to a hole
        indices: I3s = list(filter(
            lambda idxs: hull_poly.contains(Polygon(list(map(lambda i: vertices[i], idxs)))), filter(
                lambda idxs: not all(map(lambda i: i < p_len, idxs)), triangles
            )
        ))
        # join all hole points
        stacked_hole_points: D2s = list(map(lambda i: points[i], filter(lambda i: i < p_len, flat(indices))))
        return hull_points, FrameGroup.__get_holes(stacked_hole_points, hole_alpha) if stacked_hole_points else []

    @staticmethod
    def __validate_param(param: float | str) -> None:
        """
        validates a parameter to be numerical or to be 'auto'

        :param param: the parameter to be checked
        :raises: ValueError if the parameter is of type string but not set to 'auto'
        """
        if type(param) == str and param != auto:
            raise ValueError(f'Parameters may only be numeric if not set to "{auto}", provided "{param}"')

    @staticmethod
    def __get_zs(xyzs: D3s) -> D1s:
        """
        extracts only the depth information from a list with latitude, longitude and

        :param xyzs: a list of 3d points
        :return: a list of depth values
        """
        return [z for _, _, z in xyzs]

    @staticmethod
    def __mod_zs(xyzs: D3s, mod: float, add: bool = False) -> D3s:
        """
        modifies all z (depth) values of a provided list

        :param xyzs: a list of 3d points
        :param mod: the value to be added to the z value or the value that z will be subtracted from
        :param add: if true, simply adds the value to the original z value, subtracts z from the mod value otherwise
        :return: the modified list
        """
        return [(x, y, mod + z if add else mod - z) for x, y, z in xyzs]

    @staticmethod
    def __set_zs(xyzs: D3s, new_z: float) -> D3s:
        """
        replaces all z (depth) values of a provided list

        :param xyzs: a list of 3d points
        :param new_z: the new z value that will replace the old one
        :return: the list with the fixed z values
        """
        return [(x, y, new_z) for x, y, z in xyzs]

    @staticmethod
    def __build_floor_tri(tri: ndarray) -> ndarray:
        """
        creates a triangle on the xy0-plane

        :param tri: the triangle that should be placed on the xy0-plane
        :return: a triangle on the xy0-plane
        """
        return array([(x, y, 0) for x, y, _ in tri])

    @staticmethod
    def __build_wall(points: tuple[D3, D3], z: float = 0) -> Ns:
        """
        creates a wall segment for the provided points

        :param points: the points forming a line
        :param z: the value where the xy-plane should be constructing the other line
        :return: two triangles representing the wall
        """
        pms: D3s = [(x, y, z) for x, y, _ in points]
        return FrameGroup.__square_to_triangle((*points, *pms))

    @staticmethod
    def __mesh_from_vectors(vectors: Ns) -> Mesh:
        """
        converts vectors into an stl-mesh

        :param vectors: the vectors forming triangles representing the 3d model
        :return: a mesh object
        """
        data: ndarray = zeros(len(vectors), dtype=Mesh.dtype)
        data['vectors']: Ns = vectors
        return Mesh(data, remove_empty_areas=True)

    @staticmethod
    def __tris_to_walls(vectors: Ns, z: float = 0) -> Ns:
        """
        constructs walls from the provided list of triangles

        :param vectors: a list of triangles
        :param z: the z value of the bottom or top piece of the wall
        :return: a list of triangles representing the walls for the provided triangles
        """
        wall_lines: list[tuple[D3, D3]] = list(flat(map(lambda tri: FrameGroup.__tri_to_lines(tri), vectors)))
        unique_wall_lines: list[tuple[D3, D3]] = [k for k, v in Counter(wall_lines).items() if v == 1]
        return list(flat(map(lambda l: FrameGroup.__build_wall(l, z), unique_wall_lines)))

    @staticmethod
    def __unite(tris: Ns) -> Polygon:
        """
        joins triangles to a single polygon

        :param tris: the triangles to be joined into a polygon
        :return: the polygon containing all provided triangles
        """
        poly: Polygon = Polygon()

        for tri in tris:
            poly: Polygon = poly.union(Polygon(tri))

        return poly

    @staticmethod
    def __fix_tri(tri: ndarray, fixed_points: D3s, hole_poly: Polygon) -> Ns:
        """
        fixes the overlapping part of a triangle if it intersects with a hole

        :param tri: the triangle to be fixed
        :param fixed_points: the points that should be joined to fix the triangle
        :param hole_poly: the polygon of a hole
        :return: a list of triangles that now only touch the provided hole instead of intersecting into it
        """
        exterior_points: D3s = list(map(tuple, filter(lambda p: not Point(p[:2]).intersects(hole_poly), tri)))
        return list(filter(
            lambda t_tri: not hole_poly.covers(Polygon(t_tri[:, :2])),
            FrameGroup.__triangulate_vectors(list(exterior_points) + fixed_points)
        ))

    @staticmethod
    def __calc_hole_points(tri: ndarray, hole: D2s) -> D3s:
        """
        calculates the depth values of intersecting points using a plane (hessian normal form)

        :param tri: the triangle that intersects the provided hole
        :param hole: a hole that the triangle intersects into
        :return: a list of points with the original points outside the hole and all intersection points with the hole
        """
        inters: LineString | MultiLineString = LinearRing(hole).intersection(Polygon(tri[:, :2]))
        inter_points: D2s = list(inters.coords) if type(inters) == LineString else list(flat(map(
            lambda geom: list(geom.coords), inters.geoms
        )))
        p1, p2, p3 = tri
        normal: ndarray = cross(p2 - p1, p3 - p1)
        a, b, c = normal
        d: float = normal.dot(p1)
        return [(x, y, (d - a * x - b * y) / c) for x, y in inter_points]

    @staticmethod
    def __square_to_triangle(points: tuple[D3, D3, D3, D3]) -> Ns:
        """
        converts 4 points to 2 triangles

        :param points: the points forming any square
        :return: a list of triangles (of size 2)
        """
        p1, p2, p3, p4 = points
        z1, z2, z3, z4 = FrameGroup.__get_zs(points)
        # check the points inside the triangles to prevent a straight plane and add better contours
        is_smaller: bool = math.fabs(z1 - z4) < math.fabs(z2 - z3)
        return [array((p1, p2, p3)), array((p2, p3, p4))] if is_smaller else [array((p1, p3, p4)), array((p1, p2, p4))]

    @staticmethod
    def __triangulate(points: D2s, shape: Optional['FrameGroup'] = None) -> Triangulation:
        """
        triangulates a given list of points with respect to the shape

        :param points: the points to be triangulated
        :param shape: the shape to limit the triangulation to (hull and holes)
        :return: a triangulation with a mask
        """
        delaunay: dict = triangulate({'vertices': points}, opts='')
        vertices: D2s = delaunay['vertices']
        triangles: I3s = delaunay['triangles']
        # create a mask based on the location of the vertices
        mask: Optional[list[bool]] = None if not shape else list(map(
            lambda idxs: not shape.is_interior(Polygon(list(map(lambda i: (vertices[i]), idxs)))), triangles
        ))

        return Triangulation(*list(zip(*vertices)), triangles=triangles, mask=mask)

    @staticmethod
    def __triangulate_vectors(points: D3s, shape: Optional['FrameGroup'] = None) -> Ns:
        """
        removes one dimension of the provided points, triangulates them and re-adds the dimension

        :param points: the 3d points to be triangulated
        :param shape: the shape to be used when triangulating (hull & holes)
        :return: a list of triangles forming the triangulated 3d area
        """
        vectors: I3s = FrameGroup.__triangulate([(x, y) for x, y, _ in points], shape).get_masked_triangles()
        return list(map(lambda idxs: array(list(map(lambda i: points[i], idxs))), vectors))

    @staticmethod
    def __get_holes(hole_points: D2s, hole_alpha: float) -> list[D2s]:
        """
        constructs holes from provided points

        :param hole_points: the points considered to be part of a hole
        :param hole_alpha: the value for the precision of the alpha shapes
        :raises: ValueError if the resulting alpha shapes form anything other than a polygon or a multi-polygon
        :return:
        """
        polys: MultiPolygon | Polygon | Any = alphashape(hole_points, float(hole_alpha))

        if type(polys) != MultiPolygon and type(polys) != Polygon:
            raise ValueError(
                f'The hole alpha "{hole_alpha}" value causes the holes to be too imprecise, the alpha value may be '
                'lowered to prevent this'
            )

        # reduce the complexity to a list of polygons
        if type(polys) == Polygon:
            polys = [polys]
        elif type(polys) == MultiPolygon:
            polys = polys.geoms

        return list(map(lambda p: list(reversed(p.exterior.coords)), polys))

    def __rebuild_vectors(self) -> None:
        """
        reconstructs most properties after changing values and splits rows
        """
        self.__clear()
        any(map(self.__split_row, self.__frames))

    def __order_frames(self):
        self.__by_depths: EFs = sorted(self.__frames, key=lambda f: f.water_depth_m)

    def __order_and_map_points(self, ext_points: D2s) -> EFs:
        """
        orders the internal frames by the order of provided points

        :param ext_points: the 2d points forming any kind of order
        :return: a list of frames ordered by the order of the provided points
        """
        unordered_frames: EFs = list(filter(lambda f: f.as_2d_pos() in ext_points, self.__frames))
        return list(map(lambda pos: next((f for f in unordered_frames if f.as_2d_pos() == pos)), ext_points))

    def __as_2d_pos(self) -> D2s:
        return list(map(lambda f: f.as_2d_pos(), self.__frames))

    def __as_3d_pos(self) -> D3s:
        return list(map(lambda f: f.as_3d_pos(), self.__frames))

    def __validate_vectors(self) -> None:
        """
        validates most properties
        """
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
