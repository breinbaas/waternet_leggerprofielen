from typing import List, Tuple, Dict
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


def get_soils(dm: DStabilityModel) -> List[Dict]:
    return {
        s.Id: {
            "code": s.Code,
            "ys": s.VolumetricWeightBelowPhreaticLevel,
            "cohesion": s.MohrCoulombClassicShearStrengthModel.Cohesion,
        }
        for s in dm.datastructure.soils.Soils
    }


def get_soillayers(dm: DStabilityModel) -> List[Dict]:
    geometry = dm._get_geometry(scenario_index=0, stage_index=0)
    soils = get_soils(dm)

    # grondlaag connecties met grondsoorten
    layer_soil_dict = {
        l.LayerId: l.SoilId
        for l in dm._get_soil_layers(scenario_index=0, stage_index=0).SoilLayers
    }
    return [
        {
            "id": layer.Id,
            "soil": soils[layer_soil_dict[layer.Id]],
            "points": [(p.X, p.Z) for p in layer.Points],
        }
        for layer in geometry.Layers
    ]


def get_natural_slopes_line(
    dm: DStabilityModel, x_uittredepunt: float
) -> List[Tuple[float, float]]:
    result = [(x_uittredepunt, dm.z_at(x_uittredepunt))]
    x = x_uittredepunt
    geometry = dm._get_geometry(scenario_index=0, stage_index=0)
    soillayers = get_soillayers(dm)
    all_points = []
    for soillayer in soillayers:
        all_points += soillayer["points"]
    bottom = min([p[1] for p in all_points])
    top = max([p[1] for p in all_points])
    soils = get_soils(dm)
    layer_soil_dict = {
        l.LayerId: l.SoilId
        for l in dm._get_soil_layers(scenario_index=0, stage_index=0).SoilLayers
    }
    soillayers = [
        {
            "id": layer.Id,
            "soil": soils[layer_soil_dict[layer.Id]],
            "points": [(p.X, p.Z) for p in layer.Points],
        }
        for layer in geometry.Layers
    ]
    # create a slopes dictionary
    slopes = {}
    for sl in soillayers:
        soilcode = sl["soil"]["code"]
        ys = sl["soil"]["ys"]
        c = sl["soil"]["cohesion"]

        if soilcode not in slopes.keys():
            if ys < 12:
                slopes[soilcode] = 6
            elif ys > 18:
                slopes[soilcode] = 4
            elif c >= 3.0:
                slopes[soilcode] = 3
            else:
                slopes[soilcode] = 4

    all_intersections = []
    for sl in soillayers:
        intersections = line_polygon_intersections(
            (x, top + 1.0), (x, bottom - 1.0), sl["points"]
        )

        if len(intersections) > 0 and len(intersections) % 2 == 0:
            for i in range(int(len(intersections) / 2)):
                all_intersections.append(
                    {
                        "top": intersections[i * 2][1],
                        "bottom": intersections[i * 2 + 1][1],
                        "slope": slopes[sl["soil"]["code"]],
                    }
                )

    # sort all intersections
    all_intersections = sorted(all_intersections, key=lambda x: x["top"], reverse=True)

    # voeg nu de punten toe obv de helling van de grondsoorten
    # px, pz = x, max([p["top"] for p in all_intersections])
    px = x
    for intersection in all_intersections[:-1]:
        dz = intersection["top"] - intersection["bottom"]
        px += dz * intersection["slope"]
        result.append((px, intersection["bottom"]))

    return result
