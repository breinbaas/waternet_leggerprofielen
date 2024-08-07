import logging
import os
import glob
from pathlib import Path
from geolib.models.dstability import DStabilityModel
from geolib.models.dstability.dstability_model import CharacteristicPointEnum
from geolib.models.dstability.internal import (
    AnalysisTypeEnum,
    ShearStrengthModelTypePhreaticLevelInternal,
)
from geolib.soils.soil import Soil
import matplotlib.pyplot as plt
from copy import deepcopy

from helpers import case_insensitive_glob, get_natural_slopes_line
from dstability_model_modifier import DStabilityModelModifier


# set to true if the script / method has changed
FORCE_RECALCULATION = False

# input paths and files
PATH_STIX_FILES = (
    r"Z:\Documents\Klanten\OneDrive\Waternet\Legger\input\berekeningen\A535"
)

# output paths and files
LOG_FILE_LEGGER = r"Z:\Documents\Klanten\Output\Waternet\Legger\Log\legger.log"
CALCULATIONS_PATH = r"Z:\Documents\Klanten\Output\Waternet\Legger\calculations"
CALCULATION_ERRORS_PATH = (
    r"Z:\Documents\Klanten\Output\Waternet\Legger\calculations\errors"
)
PLOT_PATH = r"Z:\Documents\Klanten\Output\Waternet\Legger\Plots"
CSV_PATH = r"Z:\Documents\Klanten\Output\Waternet\Legger\csv"
SOLUTIONS_PATH = r"Z:\Documents\Klanten\Output\Waternet\Legger\solutions"

# settings
MIN_SLIP_PLANE_LENGTH = 3.0
MIN_SLIP_PLANE_DEPTH = 2.0
# OFFSET_B = 0.3
PL_SURFACE_OFFSET = 0.1
INITIAL_SLOPE = 1.0
MAX_ITERATIONS = 10
SF_MARGIN = 0.1

# TODO next info should come from shapefiles
DTH = -1.8
RIVER_LEVEL = -2.0
CREST_WIDTH = 3.0
REQUIRED_SF = 1.0
FMIN_MARGIN = 0.1
POLDER_LEVEL = -5.0
EXCAVATION_DEPTH = 2.0


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
# logging.info(
#     f"De afstand tussen MHW en de waterstand onder de buitenkruinlijn is ingesteld op {OFFSET_B} meter"
# )
logging.info(
    f"NB. Er wordt GEEN rekening gehouden met het verloop van de stijghoogte naar de freatische lijn."
)

stix_files = case_insensitive_glob(PATH_STIX_FILES, ".stix")

# remove all the generated files
# TODO > maybe not remove the solutions and check this to avoid calculating where
# we already have a solution
files = glob.glob(f"{CALCULATION_ERRORS_PATH}/*.stix")
files += glob.glob(f"{CALCULATIONS_PATH}/*.stix")
files += glob.glob(f"{PLOT_PATH}/*.png")
files += glob.glob(f"{CSV_PATH}/*.csv")
files += glob.glob(f"{SOLUTIONS_PATH}/*.stix")
for f in files:
    os.remove(f)

for stix_file in stix_files:
    solution_file = Path(CALCULATIONS_PATH) / f"{stix_file.stem}.solution.stix"
    if solution_file.exists() and not FORCE_RECALCULATION:
        logging.info(
            f"Skipping {stix_file.stem} because a solution has already been found."
        )
        continue

    logging.info(f"Handling {stix_file}")

    #################
    # READ THE FILE #
    #################
    dm = DStabilityModel()
    try:
        dm.parse(Path(stix_file))
    except Exception as e:
        logging.error(f"Cannot open file '{stix_file}', got error '{e}'")
        continue

    ############################################
    # CHECK THE NUMBER OF SCENARIOS AND STAGES #
    ############################################
    if len(dm.scenarios) > 1:
        logging.warning(
            "Dit bestand heeft meer dan 1 scenario, beperk de invoer tot 1 scenario. We rekenen wel door maar check de uitkomsten!"
        )
    if len(dm.scenarios[0].Stages) > 1:
        logging.error(
            "Dit bestand heeft in het eerste scenario meer dan 1 stage, beperk de invoer tot 1 scenario met 1 stage. We rekenen wel door maar check de uitkomsten!"
        )

    ####################
    # CHECK CURRENT SF #
    ####################
    try:
        dm.execute()
        sf = round(dm.get_result(0, 0).FactorOfSafety, 2)
        logging.info(f"De huidige veiligheidsfactor bedraagt {sf:.2f}")
    except Exception as e:
        logging.error(
            f"Fout bij het berekenen van de originele veiligheidsfactor; '{e}'"
        )
        dm_iteration.serialize(Path(CALCULATION_ERRORS_PATH) / f"{stix_file.stem}.stix")
        continue

    ########################################
    # CHECK THE CURRENT CALCULATION METHOD #
    ########################################
    calc_settings = dm._get_calculation_settings(scenario_index=0, calculation_index=0)

    if calc_settings.AnalysisType == AnalysisTypeEnum.SPENCER_GENETIC:
        logging.warning(
            f"Het originele rekenmodel gebruikt Spencer Genetic, deze instellingen kunnen nog niet automatisch gegenereerd worden waardoor de leggerprofielen met Bishop Brute Force worden bepaald."
        )
    elif calc_settings.AnalysisType == AnalysisTypeEnum.UPLIFT_VAN_PARTICLE_SWARM:
        logging.warning(
            f"Het originele rekenmodel gebruikt Uplift Van Particle Swarm, deze instellingen kunnen nog niet automatisch gegenereerd worden waardoor de leggerprofielen met Bishop Brute Force worden bepaald."
        )
    elif calc_settings.AnalysisType != AnalysisTypeEnum.BISHOP_BRUTE_FORCE:
        logging.error(
            "Niet ondersteund type berekening (geen bishof brute force, spencer genetic of uplift van particle swarm)"
        )
        continue

    ##########################################
    # TODO -> GET THE CALCULATION PARAMETERS #
    ##########################################
    dth = DTH
    crest_width = CREST_WIDTH
    river_level = RIVER_LEVEL
    polder_level = POLDER_LEVEL
    required_sf = REQUIRED_SF

    # make sure to add the ophoogmateriaal to the model
    # TODO dit is een setting
    soil = Soil()
    soil.code = "Ophoogmateriaal"
    soil.name = "Ophoogmateriaal"
    soil.shear_strength_model_above_phreatic_level = "Mohr_Coulomb"
    soil.shear_strength_model_below_phreatic_level = "Mohr_Coulomb"
    soil.soil_weight_parameters.saturated_weight.mean = 15.0
    soil.soil_weight_parameters.unsaturated_weight.mean = 15.0
    soil.mohr_coulomb_parameters.cohesion = 2.0
    soil.mohr_coulomb_parameters.friction_angle = 22.0
    soil.mohr_coulomb_parameters.dilatancy_angle = 22.0
    dm.add_soil(soil)

    ##########################
    # ITERATE OVER SOLUTIONS #
    ##########################
    slope = INITIAL_SLOPE
    iteration = 1
    solution = None
    done = False
    while not done:
        logging.info(
            f"Generating and calculating iteration {iteration} with slope {slope:.2f}"
        )

        dm_copy = deepcopy(dm)
        x1 = dm_copy.xmin
        z1 = dth
        x2 = CREST_WIDTH  # we expect the reference line to be on x=0.0, TODO > check?
        z2 = dth
        x4 = dm_copy.xmax
        z4 = dm_copy.surface[-1][
            1
        ]  # we assume the polder level is the same as the last point of the surface
        z3 = z4
        x3 = x2 + (z2 - z3) * slope

        profile_line = [(x1, z1), (x2, z2), (x3, z3), (x4, z4)]

        # if we have a ditch add it to the new profile
        if len(dm_copy.ditch_points) == 4:
            new_ditch_points = []
            dp = dm_copy.ditch_points

            # check if the ditch bottom is above z3, if so skip the ditch
            ditch_zmin = min([p[1] for p in dp])
            if ditch_zmin >= z3:
                break  # no need to add the ditch because the original bottom is above the new polderlevel

            # get the original slopes
            slope_left = (dp[1][0] - dp[0][0]) / (dp[0][1] - dp[1][1])
            slope_right = (dp[-1][0] - dp[-2][0]) / (dp[-1][1] - dp[-2][1])
            # and the original width
            ditch_width = dp[2][0] - dp[1][0]

            # if we pass the original ditch we need to move the ditch
            if x3 > dp[0][0] - 0.1:
                xs = x3 + 0.1
            else:
                xs = dp[0][0]

            xd1 = xs
            zd1 = z3
            zd2 = dp[1][1]
            xd2 = xd1 + slope_left * (zd1 - zd2)
            xd3 = xd2 + ditch_width
            zd3 = dp[2][1]
            zd4 = z4
            xd4 = xd3 + slope_right * (zd4 - zd3)

            # check if we did not pass x4, if so adjust x4
            if xd4 >= x4:
                x4 = xd4 + 0.1

            profile_line = profile_line[:3] + [
                (xd1, zd1),
                (xd2, zd2),
                (xd3, zd3),
                (xd4, zd4),
                (x4, z4),
            ]

        # create a plot for debugging purposes
        # fig, ax = plt.subplots(figsize=(15, 5))
        # ax.plot([p[0] for p in dm.surface], [p[1] for p in dm.surface], "k")
        # ax.plot([p[0] for p in profile_line], [p[1] for p in profile_line], "r")
        # ax.set_aspect("equal", adjustable="box")
        # fig.savefig(Path(PLOT_PATH) / f"{stix_file.stem}.profile_line_{slope:.2f}.png")

        # create the model
        dmm = DStabilityModelModifier(
            dm=dm_copy,
        )
        dmm.initialize()
        dmm.cut(profile_line)
        dmm.fill(
            line=profile_line,
            soil_code="Ophoogmateriaal",
        )

        # generate the phreatic line
        plline_points = [
            (x1, river_level),
            (0, river_level),
            (1.0, river_level - 1.0),
            (x2, river_level - 1.2),
            (x3, polder_level),
            (x4, polder_level),
        ]
        dmm.set_phreatic_line(plline_points)

        # dmm.set_phreatic_line()
        dm_iteration = dmm.to_dstability_model_with_autogenerated_settings(
            point_ref=(0, dth),
            point_crest_land=(x2, z2),
            point_toe=(x3, z3),
            ditch_points=dm_copy.ditch_points,
        )

        dm_iteration.serialize(
            Path(CALCULATIONS_PATH)
            / f"{stix_file.stem}_iteration_{iteration}_slope_{slope:.2f}.stix"
        )

        try:
            dm_iteration.execute()
            sf = round(dm_iteration.get_result(0, 0).FactorOfSafety, 2)
        except Exception as e:
            logging.error(f"Could not calculate slope {slope:.2f}, got error {e}")
            dm_iteration.serialize(
                Path(CALCULATION_ERRORS_PATH)
                / f"{stix_file.stem}_iteration_{iteration}_slope_{slope:.2f}.stix"
            )

        if sf >= required_sf and sf <= required_sf + SF_MARGIN:
            logging.info(
                f"Found a solution after {iteration} iteration(s) with slope=1:{slope:.2f}"
            )
            solution = dm_iteration
            done = True
        elif sf < required_sf:
            slope *= 1.2
        else:
            slope /= 1.1

        iteration += 1

        if not done and iteration > MAX_ITERATIONS:
            logging.error(
                f"After {MAX_ITERATIONS} iterations we still have no solution, skipping this levee"
            )
            done = True
            break

    if solution is None:
        continue

    # get uittrede punt
    x_uittredepunt = solution.datastructure.bishop_bruteforce_results[0].Points[-1].X
    # get the remaining profile based on the slopes of the soils at x_uittredepunt
    # NOTE that we only use the soils directly under x_uittredepunt for the slopes
    # any changes in soil layers to the right of x_uittredepunt are ignored
    # TODO > can be optimized
    natural_slopes_line = get_natural_slopes_line(solution, x_uittredepunt)
    natural_slopes_line_left = [
        (-1 * p[0], p[1]) for p in get_natural_slopes_line(solution, 0)
    ]

    # create the final line
    final_line = natural_slopes_line_left[::-1]
    final_line += [p for p in solution.surface if p[0] > 0.0 and p[0] < x_uittredepunt]
    final_line += natural_slopes_line

    # plot the solution
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot([p[0] for p in dm.surface], [p[1] for p in dm.surface], "k")
    ax.plot([p[0] for p in final_line], [p[1] for p in final_line], "k--")
    fig.savefig(Path(PLOT_PATH) / f"{stix_file.stem}_solution.png")

    # write a csv file
    with open(Path(CSV_PATH) / f"{stix_file.stem}_solution.csv", "w") as f:
        f.write("x,z\n")
        for p in final_line:
            f.write("{p[0]:.2f},{p[1]:.2f}\n")

    solution.serialize(Path(SOLUTIONS_PATH) / f"{stix_file.stem}_solution.stix")
