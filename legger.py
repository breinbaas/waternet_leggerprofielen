import logging
from typing import List, Tuple, Dict
from pathlib import Path
from geolib.models.dstability import DStabilityModel
from geolib.models.dstability.analysis import (
    DStabilityBishopBruteForceAnalysisMethod,
    DStabilitySearchGrid,
    DStabilitySlipPlaneConstraints,
)
from geolib.soils import Soil, ShearStrengthModelTypePhreaticLevel
from geolib.geometry.one import Point
from geolib.models.dstability.internal import AnalysisTypeEnum, BishopBruteForceResult
from math import isnan
from helpers import (
    case_insensitive_glob,
    surface_of_polygon_collection,
    line_polygon_intersections,
    fix_surface_points,
)
from shapely import Polygon, get_coordinates
import matplotlib.pyplot as plt

### VEREISTEN
# de karakteristieke punten moeten ingevuld zijn
# de punten aan de rechterzijde van de geometrie moeten allemaal op dezelfde x coordinaat liggen
#


LOG_FILE_LEGGER = r"Z:\Documents\Klanten\Output\Waternet\Legger\legger.log"
PATH_STIX_FILES = r"Z:\Documents\Klanten\OneDrive\Waternet\Legger\input\berekeningen"
CALCULATIONS_PATH = r"Z:\Documents\Klanten\Output\Waternet\Legger\calculations"
PLOT_PATH = r"Z:\Documents\Klanten\Output\Waternet\Legger"

MIN_SLIP_PLANE_LENGTH = 3.0
MIN_SLIP_PLANE_DEPTH = 2.0
OFFSET_B = 0.3
PL_SURFACE_OFFSET = 0.1
MAX_ITERATIONS = 10

logging.basicConfig(
    filename=LOG_FILE_LEGGER,
    filemode="w",
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

logging.info(f"Minimale lengte glijvlak is ingesteld op {MIN_SLIP_PLANE_LENGTH} meter")
logging.info(f"Minimale diepte glijvlak is ingesteld op {MIN_SLIP_PLANE_DEPTH} meter")
logging.info(
    f"De afstand tussen de freatische lijn en het maaiveld is ingesteld op {PL_SURFACE_OFFSET} meter"
)
logging.info(
    f"NB. Er wordt GEEN rekening gehouden met het verloop van de stijghoogte naar de freatische lijn."
)

# get the stix files
stix_files = case_insensitive_glob(PATH_STIX_FILES, ".stix")

# TODO
# dth, kruinbreedte, polderpeil bepalen obv data
DTH = 0.1
CREST_WIDTH = 1.5
REQUIRED_SF = 1.2
FMIN_MARGIN = 0.1
POLDERPEIL = -2.0
EXCAVATION_DEPTH = 2.0


def get_soils(dm: DStabilityModel) -> List[Dict]:
    return {
        s.Id: {
            "code": s.Code,
            "ys": s.VolumetricWeightBelowPhreaticLevel,
            "cohesion": s.MohrCoulombClassicShearStrengthModel.Cohesion,
        }
        for s in dm.datastructure.soils.Soils
    }


def get_uittredepunt(dm: DStabilityModel) -> Tuple[float, float]:
    result = dm.get_result(0, 0)
    if type(result) == BishopBruteForceResult:
        return result.Points[-1].X, result.Points[-1].Z
    else:
        raise ValueError(f"Unhandled resulttype '{type(result)}'")


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


def get_used_soils(dm: DStabilityModel) -> List[Soil]:
    used_soilcodes = [d["soil"]["code"] for d in get_soillayers(dm)]
    return [s for s in dm.datastructure.soils.Soils if s.Code in used_soilcodes]


def cut(
    dm: DStabilityModel,
    slope: float,
    dth: float,
    mhw: float,
    polderpeil: float,
    polderpeil_offset: float = 0.1,  # afstand tussen maaiveld polder en polderpeil, standaard 0.1m
    crest_width: float = 1.5,  # kruinbreedte
    pl_offset_binnenkruin: float = 0.3,  # afstand tussen mhw en binnenkruin pl lijn (standaard 0.3m lager)
) -> DStabilityModel:

    # get the current soillayers
    soillayers = get_soillayers(dm)

    # geometrie limieten
    all_points = []
    for soillayer in soillayers:
        all_points += soillayer["points"]
    left = min([p[0] for p in all_points])
    right = max([p[0] for p in all_points])
    top = max([p[1] for p in all_points])

    # de eerste drie punten (links, reflijn en minimale kruinbreedte)
    cut_line = [(left, dth), (0.0, dth), (crest_width, dth)]

    # dan helling met slope tot polderpeil
    dz = dth - polderpeil + polderpeil_offset
    cut_line += [
        (crest_width + dz * slope, polderpeil + polderpeil_offset),
        (right, polderpeil + polderpeil_offset),
    ]

    # create a polygon from the line
    cut_line += [
        (right + 1.0, cut_line[-1][1]),
        (right + 1.0, top + 1.0),
        (left - 1.0, top + 1.0),
        (left - 1.0, cut_line[0][1]),
    ]
    pg_extract = Polygon(cut_line)

    pl_points = [
        (left, mhw),
        (0.0, mhw),
        (crest_width, mhw - pl_offset_binnenkruin),
        (crest_width + dz * slope, polderpeil),
        (right, polderpeil),
    ]

    # get all soillayers and extract the polygon
    # create a list of all remaining polygons with their soiltype
    new_soillayers = []
    for sl in soillayers:
        pg = Polygon(sl["points"])

        extraction_result = pg.difference(pg_extract)

        if type(extraction_result) == Polygon:
            if not extraction_result.is_empty:
                xx, yy = extraction_result.centroid.xy
                new_soillayers.append(
                    {
                        "soilcode": sl["soil"]["code"],
                        "points": get_coordinates(extraction_result).tolist(),
                        "cog": (xx.tolist()[0], yy.tolist()[0]),
                    }
                )
        else:
            print(f"Unhandled type; {type(extraction_result)}")

    sorted_soillayers = sorted(new_soillayers, key=lambda s: (s["cog"][1], s["cog"][0]))

    # build new model
    dm_new = DStabilityModel()

    # copy all soils that are used
    # TODO get the used soil materials
    used_soils = get_used_soils(dm)
    added_soilcodes = [s.Code for s in dm_new.soils.Soils]
    for psoil in used_soils:
        if psoil.Code in added_soilcodes:
            continue

        soil = Soil()
        soil.name = psoil.Name
        soil.code = psoil.Code
        soil.soil_weight_parameters.saturated_weight.mean = (
            psoil.VolumetricWeightAbovePhreaticLevel
        )
        soil.soil_weight_parameters.unsaturated_weight.mean = (
            psoil.VolumetricWeightBelowPhreaticLevel
        )
        soil.mohr_coulomb_parameters.cohesion.mean = (
            psoil.MohrCoulombAdvancedShearStrengthModel.Cohesion
        )
        soil.mohr_coulomb_parameters.friction_angle.mean = (
            psoil.MohrCoulombAdvancedShearStrengthModel.FrictionAngle
        )
        soil.shear_strength_model_above_phreatic_level = (
            ShearStrengthModelTypePhreaticLevel.MOHR_COULOMB
        )
        soil.shear_strength_model_below_phreatic_level = (
            ShearStrengthModelTypePhreaticLevel.MOHR_COULOMB
        )
        dm_new.add_soil(soil)
        added_soilcodes.append(psoil.Code)

    # create the layers
    for soillayer in sorted_soillayers:
        points = [Point(x=p[0], z=p[1]) for p in soillayer["points"]]
        dm_new.add_layer(points, soillayer["soilcode"])

    # create a new phreatic line
    pl_id = dm_new.add_head_line(
        points=[Point(x=p[0], z=p[1]) for p in pl_points],
        is_phreatic_line=True,
        label="Freatische lijn",
    )
    dm_new.add_reference_line(
        points=[Point(x=p[0], z=p[1]) for p in dm_new.surface],
        top_head_line_id=pl_id,
        bottom_headline_id=pl_id,
    )

    # copy the calculation settings
    calculation_settings = dm._get_calculation_settings(0, 0)
    if calculation_settings.AnalysisType == AnalysisTypeEnum.BISHOP_BRUTE_FORCE:
        dm_new.set_model(
            DStabilityBishopBruteForceAnalysisMethod(
                search_grid=DStabilitySearchGrid(
                    bottom_left=Point(
                        x=calculation_settings.BishopBruteForce.SearchGrid.BottomLeft.X,
                        z=calculation_settings.BishopBruteForce.SearchGrid.BottomLeft.Z,
                    ),
                    number_of_points_in_x=calculation_settings.BishopBruteForce.SearchGrid.NumberOfPointsInX,
                    number_of_points_in_z=calculation_settings.BishopBruteForce.SearchGrid.NumberOfPointsInZ,
                    space=calculation_settings.BishopBruteForce.SearchGrid.Space,
                ),
                bottom_tangent_line_z=calculation_settings.BishopBruteForce.TangentLines.BottomTangentLineZ,
                number_of_tangent_lines=calculation_settings.BishopBruteForce.TangentLines.NumberOfTangentLines,
                space_tangent_lines=calculation_settings.BishopBruteForce.TangentLines.Space,
                slip_plane_constraints=DStabilitySlipPlaneConstraints(
                    is_size_constraints_enabled=True,
                    minimum_slip_plane_depth=MIN_SLIP_PLANE_DEPTH,
                    minimum_slip_plane_length=MIN_SLIP_PLANE_LENGTH,
                ),
            )
        )
    else:
        logging.error(f"Unhandled analysis type '{calculation_settings.AnalysisType}'")
        return None

    return dm_new


def get_natural_slopes_line(
    dm: DStabilityModel, uittredepunt: Tuple[float, float]
) -> List[Tuple[float, float]]:
    result = [uittredepunt]
    x = uittredepunt[0]
    geometry = dm._get_geometry(scenario_index=0, stage_index=0)
    soillayers = get_soillayers(dm)
    all_points = []
    for soillayer in soillayers:
        all_points += soillayer["points"]
    bottom = min([p[1] for p in all_points])
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
    px, pz = x, max([p["top"] for p in all_intersections])
    for intersection in all_intersections[:-1]:
        dz = pz - intersection["bottom"]
        px += dz * intersection["slope"]
        result.append((px, intersection["bottom"]))

    return result


for stix_file in stix_files[:10]:
    # check of er al een solution bestand is
    solution_file = Path(CALCULATIONS_PATH) / f"{stix_file.stem}.solution.stix"

    if solution_file.exists():
        logging.info(
            f"Skipping {stix_file.stem} because a solution has already been found."
        )
        continue

    logging.info(f"Handling {stix_file}")
    dm = DStabilityModel()
    dm.parse(Path(stix_file))

    soillayers = get_soillayers(dm)

    all_points = []
    for soillayer in soillayers:
        all_points += soillayer["points"]

    # geometrie limieten
    left = min([p[0] for p in all_points])
    right = max([p[0] for p in all_points])
    top = max([p[1] for p in all_points])

    # TODO check if there is only 1 scenario and 1 stage
    if len(dm.scenarios) > 1:
        logging.error(
            "Dit bestand heeft meer dan 1 scenario, beperk de invoer tot 1 scenario."
        )
        continue
    if len(dm.scenarios[0].Stages) > 1:
        logging.error(
            "Dit bestand heeft in het eerste scenario meer dan 1 stage, beperk de invoer tot 1 scenario met 1 stage."
        )
        continue

    try:
        dm.execute()
        sf = round(dm.get_result(0, 0).FactorOfSafety, 2)
    except Exception as e:
        logging.error(f"Fout bij het berekenen; '{e}'")
        continue

    # RAPPORT paragraaf 4.2
    # Stap 1
    # Indien Fmin aan de norm voldoet dan is de uitkomst gelijk aan de huidige geometrie
    if sf >= REQUIRED_SF and sf <= (REQUIRED_SF + FMIN_MARGIN):
        logging.info(
            f"De huidige geometrie heeft een veiligheidsfactor van {sf:.2f} en valt daarmee binnen de marge van {REQUIRED_SF:.2f} tot {(REQUIRED_SF + FMIN_MARGIN):.2f} waar de huidige geometrie de input voor de legger is."
        )
        continue

    if sf < REQUIRED_SF:
        logging.info(
            f"De huidige veilgheidsfactor {sf:.2f} is lager dan de vereiste veiligheidsfactor ({REQUIRED_SF:.2f})"
        )
        continue

    if sf > REQUIRED_SF + FMIN_MARGIN:
        logging.info(
            f"De huidige veiligheidsfactor ({sf:.2f}) is groter dan de vereiste veiligheids factor + marge ({(REQUIRED_SF + FMIN_MARGIN):.2f}) wat aangeeft dat de dijk overgedimensioneerd is."
        )
        logging.info("Proces om de dijk af te minimaliseren is begonnen...")

        slope = 1.0
        iterations = 0
        while (
            sf < REQUIRED_SF
            or sf > REQUIRED_SF + FMIN_MARGIN
            and iterations < MAX_ITERATIONS
        ):
            dm_cut = cut(
                dm,
                slope=slope,
                mhw=DTH - 0.1,
                dth=DTH,
                polderpeil=POLDERPEIL,
                crest_width=CREST_WIDTH,
            )

            if dm_cut is None:
                logging.error(
                    "Fout in het genereren van de berekening voor het minimale profiel. Check de log."
                )
                break

            try:
                dm_cut.serialize(
                    Path(CALCULATIONS_PATH) / f"{stix_file.stem}_slope_{slope:.2f}.stix"
                )
                dm_cut.execute()
                sf = round(dm_cut.get_result(0, 0).FactorOfSafety, 2)
                logging.info(
                    f"Bij een helling van 1:{slope:.2f} is de veiligheidsfactor {sf:.3f}."
                )
            except Exception as e:
                logging.error(
                    f"Fout bij het berekenen van de berm met helling 1:{slope:.2f}; '{e}'"
                )
                continue

            if sf < REQUIRED_SF or sf > REQUIRED_SF + FMIN_MARGIN:
                slope *= (REQUIRED_SF / sf) * 1.1

            iterations += 1

        if iterations == MAX_ITERATIONS:
            logging.info(
                "Na het maximale aantal iteraties is er nog geen oplossing gekomen waarbij voldaan wordt aan de vereiste veiligheidsfactor met marge."
            )
            continue

        if dm_cut is None:
            logging.error(
                "Er is een fout aangetroffen waardoor dit bestand niet berekend kan worden."
            )
            continue

        logging.info(f"Er is een oplossing gevonden met helling 1:{slope:.2f}")
        dm_cut.serialize(solution_file)

        fig, ax = plt.subplots(figsize=(15, 5))
        # create the line of the surface
        ax.plot([p[0] for p in dm_cut.surface], [p[1] for p in dm_cut.surface], "k")

        uittredepunt = get_uittredepunt(dm=dm_cut)
        # maak hellingen conform grondopbouw op uittredepunt
        slopes_line = get_natural_slopes_line(dm_cut, uittredepunt)
        ax.plot([p[0] for p in slopes_line], [p[1] for p in slopes_line], "k--")

        excavation_level = POLDERPEIL - EXCAVATION_DEPTH
        start_excavation = left
        for i in range(1, len(slopes_line)):
            x1, z1 = slopes_line[i - 1]
            x2, z2 = slopes_line[i]

            if z1 >= excavation_level and excavation_level >= z2:
                start_excavation = x1 + (z1 - excavation_level) / (z1 - z2) * (x2 - x1)
                ax.plot(
                    [start_excavation, start_excavation],
                    [POLDERPEIL, excavation_level],
                    "k--",
                )
                logging.info(
                    f"Het begin van de ontgravingsbak van {EXCAVATION_DEPTH:.2f}m ligt op x={start_excavation:.2f},z={excavation_level:.2f}"
                )
                break

        if start_excavation == left:
            logging.info(
                f"Geen snijpunt gevonden met de lijn van de grondsoorten en de ontgravingsbak."
            )
            continue

        fig.savefig(Path(PLOT_PATH) / f"{stix_file.stem}.png")

        # buitenbeschermings zone = raaklijn voorgaande helling met 2m ontgraving tov maaiveld

        # opdrijven berekenen
        # en veiligheid SF > SF;eis
        # of opschuiven bak
        # opdrijfveiligheid < 1.1 geen buitenbeschermingszone
        # maar beschermingszone
        # wel opdrijfveiligheid >= 1.1 dan beschermingszone vanaf zijkant bak

    # # get the characteristic points
    # # we need the waternet creator
    # if dm.datastructure.scenarios[0].Stages[0].WaternetCreatorSettingsId is None:
    #     logging.error("No waternet creator settings id found.")
    #     continue

    # wnetcreator_settings = None
    # for wn in dm.datastructure.waternetcreatorsettings:
    #     if wn.Id == dm.datastructure.scenarios[0].Stages[0].WaternetCreatorSettingsId:
    #         wnetcreator_settings = wn
    #         break

    # if wnetcreator_settings is None:
    #     logging.error("No waternet creator settings found.")
    #     continue

    # # grondsoorten
    # soils = {
    #     s.Id: {
    #         "code": s.Code,
    #         "ys": s.VolumetricWeightBelowPhreaticLevel,
    #         "cohesion": s.MohrCoulombClassicShearStrengthModel.Cohesion,
    #     }
    #     for s in dm.datastructure.soils.Soils
    # }

    # # grondlaag connecties met grondsoorten
    # layer_soil_dict = {
    #     l.LayerId: l.SoilId
    #     for l in dm._get_soil_layers(scenario_index=0, stage_index=0).SoilLayers
    # }

    # # geometrie
    # geometry = dm._get_geometry(scenario_index=0, stage_index=0)
    # layers = [layer for layer in geometry.Layers]
    # soillayers = [
    #     {
    #         "id": layer.Id,
    #         "soil": soils[layer_soil_dict[layer.Id]],
    #         "points": [(p.X, p.Z) for p in layer.Points],
    #     }
    #     for layer in geometry.Layers
    # ]
    # # create a slopes dictionary
    # slopes = {}
    # for sl in soillayers:
    #     soilcode = sl["soil"]["code"]
    #     ys = sl["soil"]["ys"]
    #     c = sl["soil"]["cohesion"]

    #     if soilcode not in slopes.keys():
    #         if ys < 12:
    #             slopes[soilcode] = 6
    #         elif ys > 18:
    #             slopes[soilcode] = 4
    #         elif c >= 3.0:
    #             slopes[soilcode] = 3
    #         else:
    #             slopes[soilcode] = 4

    # # rechterzijde geometrie
    # all_points = []
    # for soillayer in soillayers:
    #     all_points += soillayer["points"]

    # # geometrie limieten
    # left = min([p[0] for p in all_points])
    # right = max([p[0] for p in all_points])
    # top = max([p[1] for p in all_points])

    # if DTH >= top:
    #     logging.error(
    #         f"Bij deze berekening ligt DTH={DTH:.2f} hoger dan of gelijk aan het hoogste punt op de geometrie."
    #     )
    #     continue

    # bottom = min([p[1] for p in all_points])

    # # dwarsprofiel
    # surface = surface_of_polygon_collection(
    #     [soillayer["points"] for soillayer in soillayers]
    # )

    # # buitenteen
    # Ax = wnetcreator_settings.EmbankmentCharacteristics.EmbankmentToeWaterSide
    # # buitenkruin VERPLICHT
    # Bx = wnetcreator_settings.EmbankmentCharacteristics.EmbankmentTopWaterSide
    # if isnan(Bx):
    #     logging.error(f"Deze berekening heeft geen x coordinaat voor de buitenteen.")
    #     continue
    # # binnenkruin
    # Cx = wnetcreator_settings.EmbankmentCharacteristics.EmbankmentTopLandSide
    # # berm top
    # Dx = wnetcreator_settings.EmbankmentCharacteristics.ShoulderBaseLandSide
    # # binnenteen
    # Ex = wnetcreator_settings.EmbankmentCharacteristics.EmbankmentToeLandSide
    # # sloot bovenzijde dijkzijde
    # Fx = wnetcreator_settings.DitchCharacteristics.DitchEmbankmentSide
    # # sloot onderzijde dijkzijde
    # Gx = wnetcreator_settings.DitchCharacteristics.DitchBottomEmbankmentSide
    # # sloot onderzijde polderzijde
    # Hx = wnetcreator_settings.DitchCharacteristics.DitchBottomLandSide
    # # sloot bovenzijde polderzijde
    # Ix = wnetcreator_settings.DitchCharacteristics.DitchLandSide

    # # MINIMAAL PROFIEL
    # # p1 = punt aan linkerzijde van de geometrie, dijktafelhoogte met wat marge
    # min_profile_points = []
    # min_profile_points.append((left - 1.0, DTH))
    # min_profile_points.append((left, DTH))
    # # p2 = binnenkruinlijn + minimale kruinbreedte, dijktafelhoogte
    # min_profile_points.append((CREST_WIDTH, DTH))  # 0 = reference line

    # # Maak meteen de freatische lijn
    # # Begint op MHW tot onder binnenkruin en volgt dan (uitgesneden) maaiveld tot polderpeil
    # z_pl = dm.phreatic_line.Points[0].Z
    # polderpeil = dm.phreatic_line.Points[-1].Z
    # pl_points = [(left, z_pl), (0.0, z_pl), (CREST_WIDTH, z_pl - OFFSET_B)]

    # # voeg de punten toe op basis van de helling van de grondsoorten
    # # we maken geen onderscheid tussen naast de dijk of onder de dijk
    # # maar we pakken de grondsoorten direct onder p2
    # all_intersections = []
    # for sl in soillayers:
    #     intersections = line_polygon_intersections(
    #         (Bx, top + 1.0), (Bx, bottom - 1.0), sl["points"]
    #     )
    #     if len(intersections) == 2:
    #         all_intersections.append(
    #             {
    #                 "top": intersections[0][1],
    #                 "bottom": intersections[1][1],
    #                 "slope": slopes[sl["soil"]["code"]],
    #             }
    #         )

    # # sort all intersections
    # all_intersections = sorted(all_intersections, key=lambda x: x["top"], reverse=True)

    # # remove all intersections with the bottom of the layer above the DTH
    # all_intersections = [i for i in all_intersections if i["bottom"] < DTH]

    # # voeg nu de punten toe obv de helling van de grondsoorten
    # x, z = Bx, DTH
    # for intersection in all_intersections[:-1]:
    #     dz = z - intersection["bottom"]
    #     x += dz * intersection["slope"]

    #     if x > right:
    #         dx = right - x
    #         dz = z - dx / intersection["slope"]
    #         min_profile_points.append((right, intersection["bottom"]))
    #         break

    #     min_profile_points.append((x, intersection["bottom"]))

    #     # pl lijn
    #     if intersection["bottom"] < polderpeil:
    #         dz = intersection["top"] - polderpeil
    #         pl_points.append(
    #             (x + dz * intersection["slope"], polderpeil), (right, polderpeil)
    #         )
    #     else:
    #         pl_points.append((x, intersection["bottom"] - PL_SURFACE_OFFSET))

    # # voeg het laatste punt toe (meest rechterpunt van het dwarsprofiel met wat marge)
    # min_profile_points.append((right + 1.0, min_profile_points[-1][1]))

    # # en maak een gesloten polygon
    # min_profile_points.append((right + 1.0, top + 1.0))
    # min_profile_points.append((left - 1.0, top + 1.0))

    # # create the extraction polygon
    # pg_extract = Polygon(min_profile_points)

    # # get all soillayers and extract the polygon
    # # create a list of all remaining polygons with their soiltype
    # new_soillayers = []
    # for sl in soillayers:
    #     pg = Polygon(sl["points"])

    #     extraction_result = pg.difference(pg_extract)

    #     if type(extraction_result) == Polygon:
    #         if not extraction_result.is_empty:
    #             coords = get_coordinates(extraction_result).tolist()
    #             xx, yy = extraction_result.centroid.xy
    #             new_soillayers.append(
    #                 {
    #                     "soilcode": sl["soil"]["code"],
    #                     "points": get_coordinates(extraction_result).tolist(),
    #                     "cog": (xx.tolist()[0], yy.tolist()[0]),
    #                 }
    #             )
    #     else:
    #         print(f"Unhandled type; {type(extraction_result)}")

    # sorted_soillayers = sorted(new_soillayers, key=lambda s: (s["cog"][1], s["cog"][0]))

    # # build new model
    # dm_new = DStabilityModel()

    # # copy all soils
    # added_soilcodes = []
    # for psoil in dm.soils.Soils:
    #     if psoil.Code in added_soilcodes:
    #         continue

    #     soil = Soil()
    #     soil.name = psoil.Name
    #     soil.code = psoil.Code
    #     soil.soil_weight_parameters.saturated_weight.mean = (
    #         psoil.VolumetricWeightAbovePhreaticLevel
    #     )
    #     soil.soil_weight_parameters.unsaturated_weight.mean = (
    #         psoil.VolumetricWeightBelowPhreaticLevel
    #     )
    #     soil.mohr_coulomb_parameters.cohesion.mean = (
    #         psoil.MohrCoulombAdvancedShearStrengthModel.Cohesion
    #     )
    #     soil.mohr_coulomb_parameters.friction_angle.mean = (
    #         psoil.MohrCoulombAdvancedShearStrengthModel.FrictionAngle
    #     )
    #     soil.shear_strength_model_above_phreatic_level = (
    #         ShearStrengthModelTypePhreaticLevel.MOHR_COULOMB
    #     )
    #     soil.shear_strength_model_below_phreatic_level = (
    #         ShearStrengthModelTypePhreaticLevel.MOHR_COULOMB
    #     )
    #     soil_id = dm_new.add_soil(soil)
    #     added_soilcodes.append(psoil.Code)

    # # create the layers
    # for soillayer in sorted_soillayers:
    #     points = [Point(x=p[0], z=p[1]) for p in soillayer["points"]]
    #     dm_new.add_layer(points, soillayer["soilcode"])

    # # create a new phreatic line
    # pl_id = dm_new.add_head_line(
    #     points=[Point(x=p[0], z=p[1]) for p in pl_points],
    #     is_phreatic_line=True,
    #     label="Freatische lijn",
    # )
    # dm_new.add_reference_line(
    #     points=[Point(x=p[0], z=p[1]) for p in min_profile_points[1:-3]],
    #     top_head_line_id=pl_id,
    #     bottom_headline_id=pl_id,
    # )

    # # copy the calculation settings
    # calculation_settings = dm._get_calculation_settings(0, 0)
    # if calculation_settings.AnalysisType == AnalysisTypeEnum.BISHOP_BRUTE_FORCE:
    #     dm_new.set_model(
    #         DStabilityBishopBruteForceAnalysisMethod(
    #             search_grid=DStabilitySearchGrid(
    #                 bottom_left=Point(
    #                     x=calculation_settings.BishopBruteForce.SearchGrid.BottomLeft.X,
    #                     z=calculation_settings.BishopBruteForce.SearchGrid.BottomLeft.Z,
    #                 ),
    #                 number_of_points_in_x=calculation_settings.BishopBruteForce.SearchGrid.NumberOfPointsInX,
    #                 number_of_points_in_z=calculation_settings.BishopBruteForce.SearchGrid.NumberOfPointsInZ,
    #                 space=calculation_settings.BishopBruteForce.SearchGrid.Space,
    #             ),
    #             bottom_tangent_line_z=calculation_settings.BishopBruteForce.TangentLines.BottomTangentLineZ,
    #             number_of_tangent_lines=calculation_settings.BishopBruteForce.TangentLines.NumberOfTangentLines,
    #             space_tangent_lines=calculation_settings.BishopBruteForce.TangentLines.Space,
    #             slip_plane_constraints=DStabilitySlipPlaneConstraints(
    #                 is_size_constraints_enabled=True,
    #                 minimum_slip_plane_depth=MIN_SLIP_PLANE_DEPTH,
    #                 minimum_slip_plane_length=MIN_SLIP_PLANE_LENGTH,
    #             ),
    #         )
    #     )
    # else:
    #     logging.error(f"Unhandled analysis type '{calculation_settings.AnalysisType}'")

    # dm_new.serialize(Path(CALCULATIONS_PATH) / "test.stix")
