from typing import List, Tuple
from pathlib import Path
from shapely.geometry import (
    Point,
    LineString,
    MultiPoint,
    Polygon,
    MultiLineString,
    GeometryCollection,
)
from shapely import get_coordinates
from shapely.ops import unary_union
from shapely.geometry.polygon import orient
from geolib.models.dstability import DStabilityModel


def case_insensitive_glob(filepath: str, fileextension: str) -> List[Path]:
    """Find files in given path with given file extension (case insensitive)

    Arguments:
        filepath (str): path to files
        fileextension (str): file extension to use as a filter (example .gef or .csv)

    Returns:
        List(str): list of files
    """
    p = Path(filepath)
    result = []
    for filename in p.glob("**/*"):
        if str(filename.suffix).lower() == fileextension.lower():
            result.append(filename.absolute())
    return result


def polyline_polyline_intersections(
    points_line1: List[Tuple[float, float]],
    points_line2: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    result = []
    ls1 = LineString(points_line1)
    ls2 = LineString(points_line2)
    intersections = ls1.intersection(ls2)

    if intersections.is_empty:
        final_result = []
    elif type(intersections) == MultiPoint:
        result = [(g.x, g.y) for g in intersections.geoms]
    elif type(intersections) == Point:
        x, y = intersections.coords.xy
        result.append((x[0], y[0]))
    elif intersections.is_empty:
        return []
    else:
        raise ValueError(f"Unimplemented intersection type '{type(intersections)}'")

    # do not include points that are on line1 or line2
    final_result = [p for p in result if not p in points_line1 or p in points_line2]

    if len(final_result) == 0:
        return []

    return sorted(final_result, key=lambda x: x[0])


def line_polygon_intersections(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    polygon_points: List[Tuple[float, float]],
):
    line = LineString([p1, p2])
    pg = Polygon(polygon_points)
    intersections = pg.intersection(line)

    if intersections.is_empty:
        result = []
    elif type(intersections) == MultiPoint:
        result = [(g.x, g.y) for g in intersections.geoms]
    elif type(intersections) == LineString:
        result = get_coordinates(intersections).tolist()
    elif type(intersections) == Point:
        x, y = intersections.coords.xy
        result.append((x[0], y[0]))
    elif type(intersections) == MultiLineString:
        result = get_coordinates(intersections).tolist()
    elif type(intersections) == GeometryCollection:
        geoms = [g for g in intersections.geoms if type(g) != Point]
        result = get_coordinates(geoms).tolist()
    else:
        raise ValueError(f"Unimplemented intersection type '{type(intersections)}'")

    if len(result) == 0:
        return []

    return sorted(result, key=lambda x: x[1], reverse=True)


def surface_of_polygon_collection(
    polygons: List[List[Tuple[float, float]]]
) -> List[Tuple[float, float]]:
    polygons = [Polygon(polygon) for polygon in polygons]
    boundary = orient(unary_union(polygons), sign=-1)
    boundary = [
        (round(p[0], 3), round(p[1], 3))
        for p in list(zip(*boundary.exterior.coords.xy))[:-1]
    ]
    left = min([p[0] for p in boundary])
    topleft_point = sorted([p for p in boundary if p[0] == left], key=lambda x: x[1])[
        -1
    ]

    # get the rightmost points
    right = max([p[0] for p in boundary])
    rightmost_point = sorted(
        [p for p in boundary if p[0] == right], key=lambda x: x[1]
    )[-1]

    # get the index of leftmost point
    idx_left = boundary.index(topleft_point)
    surface = boundary[idx_left:] + boundary[:idx_left]

    # get the index of the rightmost point
    idx_right = surface.index(rightmost_point)
    surface = surface[: idx_right + 1]
    return surface


def fix_surface_points(dm: DStabilityModel):
    pass
