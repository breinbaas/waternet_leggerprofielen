# general imports
import logging
from geolib.models.dstability import DStabilityModel
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt

# local imports
from dstability_model_modifier import DStabilityModelModifier
from helpers import case_insensitive_glob

# set to True if the calculation method has changed or else
# existing solutions will NOT be recalculated
FORCE_RECALCULATION = True

# file locations for input and output
LOG_FILE_LEGGER = r"D:\Documents\Klanten\Waternet\LeggerProfielen\Output\Log\legger.log"
PATH_STIX_FILES = r"D:\Documents\Klanten\Waternet\LeggerProfielen\StixFiles"
CALCULATIONS_PATH = r"D:\Documents\Klanten\Waternet\LeggerProfielen\Output\Calculations"
PLOT_PATH = r"D:\Documents\Klanten\Waternet\LeggerProfielen\Output\Plots"

# general calculation settings
MIN_SLIP_PLANE_LENGTH = 3.0
MIN_SLIP_PLANE_DEPTH = 2.0
# OFFSET_B = 0.3
PL_SURFACE_OFFSET = 0.1
# MAX_ITERATIONS = 10

# TODO next info should come from shapefiles
DTH = 0.1
CREST_WIDTH = 3.0
REQUIRED_SF = 1.0
FMIN_MARGIN = 0.1
POLDERPEIL = -2.0
EXCAVATION_DEPTH = 2.0

# how many iterations do we allow to find a levee that comes close
# to the required SF? If we do not set this value the process could
# go on forever
MAX_ITERATIONS = 10


# define the function to maximize or minimize a levee
def maximize_levee():
    pass


def minimize_levee(
    dm: DStabilityModel,
    sf: float,
    dth: float,
    crest_width: float,
    polder_level: float,
    polder_level_offset: float = 0.1,
) -> Optional[DStabilityModel]:
    """Minimaliseren van het dijkprofiel

    Dit gebeurt door een lijn te trekken vanaf het referentiepunt, 3m naar de polder (instelbaar via CREST_WIDTH)
    (hoogteschermen (waarbij 1.5m moet worden toegepast) worden genegeerd omdat we dit
    niet kunnen achterhalen) en een aangenomen helling te volgen tot de volgende diepte;

    Is er een sloot aanwezig?
    Ja -> afsnoepen tot een niveau van sloot landzijde
    Nee -> afsnoepen tot een niveau van het laatste punt van het maaiveld

    Als de stabiliteitsfactor te hoog is wordt de helling steiler gemaakt en vice
    versa. Dit gaat door tot een oplossing is gevonden of tot een maximum van 10
    iteraties niet tot een oplossing heeft geleid.

    Args:
        dm (DStabilityModel): _description_
        sf (float): _description_

    Returns:
        bool: _description_
    """
    slope = 1.0
    iterations = 0

    # gebruik het laatste punt van de surface als polderpeil
    z_polderpeil = dm.surface[-1][1]
    # als we een sloot hebben gebruikt dan het hoge punt aan de landzijde als polderniveau
    if dm.datastructure.scenarios[0].Stages[0].WaternetCreatorSettingsId is not None:
        for wn in dm.datastructure.waternetcreatorsettings:
            if (
                wn.Id
                == dm.datastructure.scenarios[0].Stages[0].WaternetCreatorSettingsId
            ):
                wnetcreator_settings = wn
                break

        try:
            Ix = wnetcreator_settings.DitchCharacteristics.DitchLandSide
            z_polderpeil - dm.z_at(Ix)
        except Exception as e:
            logging.info("No ditch land side point found.")

    logging.info(
        f"We gebruiken bij het minimaliseren van het profiel {z_polderpeil:.2f} als ontgravingspeil voor deze berekening"
    )

    while (
        sf < REQUIRED_SF
        or sf > REQUIRED_SF + FMIN_MARGIN
        and iterations < MAX_ITERATIONS
    ):
        dmm = DStabilityModelModifier(
            dm=dm,
            phreatic_line_offset=PL_SURFACE_OFFSET,
            polder_level=POLDERPEIL,
        )
        dmm.initialize()

        #
        cut_line = [(dm.xmin, dth), (0.0, dth), (crest_width, dth)]
        dz = dth - polder_level + polder_level_offset
        cut_line += [
            (crest_width + dz * slope, polder_level + polder_level_offset),
            (dm.xmax, polder_level + polder_level_offset),
        ]

        dmm.cut(cut_line)
        dm_cut = dmm.to_dstability_model()
        if dm_cut is None:
            logging.error(
                "Fout in het genereren van de berekening voor het minimale profiel. Check de log."
            )
            return None

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
        return None

    if dm_cut is None:
        logging.error(
            "Er is een fout aangetroffen waardoor dit bestand niet berekend kan worden."
        )
        return None

    logging.info(f"Er is een oplossing gevonden met helling 1:{slope:.2f}")
    dm_cut.serialize(solution_file)

    return dm_cut


# setup logging
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

# get the stix files
stix_files = case_insensitive_glob(PATH_STIX_FILES, ".stix")

# there we go
for stix_file in stix_files[:10]:
    # check if we already have a solution
    solution_file = Path(CALCULATIONS_PATH) / f"{stix_file.stem}.solution.stix"

    if solution_file.exists() and not FORCE_RECALCULATION:
        logging.info(
            f"Skipping {stix_file.stem} because a solution has already been found."
        )
        continue

    # read the file
    logging.info(f"Handling {stix_file}")
    dm = DStabilityModel()
    try:
        dm.parse(Path(stix_file))
    except Exception as e:
        logging.error(f"Cannot open file '{stix_file}', got error '{e}'")
        continue

    # check if there is only 1 scenario and 1 stage
    if len(dm.scenarios) > 1:
        logging.warning(
            "Dit bestand heeft meer dan 1 scenario, beperk de invoer tot 1 scenario. We rekenen wel door maar check de uitkomsten!"
        )
    if len(dm.scenarios[0].Stages) > 1:
        logging.error(
            "Dit bestand heeft in het eerste scenario meer dan 1 stage, beperk de invoer tot 1 scenario met 1 stage. We rekenen wel door maar check de uitkomsten!"
        )

    # bepaal de huidige veiligheidsfactor
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
        dm = maximize_levee(dm, sf)
        if dm is None:
            logging.info(
                "Het is niet gelukt om de dijk te laten voldoen aan de vereiste veiligheidsfactor, zie bovenstaande log entries."
            )
            continue
    elif sf > REQUIRED_SF + FMIN_MARGIN:
        logging.info(
            f"De huidige veiligheidsfactor ({sf:.2f}) is groter dan de vereiste veiligheids factor + marge ({(REQUIRED_SF + FMIN_MARGIN):.2f}) wat aangeeft dat de dijk overgedimensioneerd is."
        )
        logging.info("Proces om de dijk af te minimaliseren is begonnen...")

        # TODO constants should be from shape file
        dm = minimize_levee(dm, sf, DTH, CREST_WIDTH, POLDERPEIL)

        if dm is None:
            logging.info(
                "Het is niet gelukt om de dijk te minimaliseren, zie bovenstaande log entries."
            )
            continue

    fig, ax = plt.subplots(figsize=(15, 5))
    # create the line of the surface
    ax.plot([p[0] for p in dm.surface], [p[1] for p in dm.surface], "k")

    # uittredepunt = get_uittredepunt(dm=dm)
    # # maak hellingen conform grondopbouw op uittredepunt
    # slopes_line = get_natural_slopes_line(dm, uittredepunt)
    # ax.plot([p[0] for p in slopes_line], [p[1] for p in slopes_line], "k--")

    # excavation_level = POLDERPEIL - EXCAVATION_DEPTH
    # start_excavation = left
    # for i in range(1, len(slopes_line)):
    #     x1, z1 = slopes_line[i - 1]
    #     x2, z2 = slopes_line[i]

    #     if z1 >= excavation_level and excavation_level >= z2:
    #         start_excavation = x1 + (z1 - excavation_level) / (z1 - z2) * (x2 - x1)
    #         ax.plot(
    #             [start_excavation, start_excavation],
    #             [POLDERPEIL, excavation_level],
    #             "k--",
    #         )
    #         logging.info(
    #             f"Het begin van de ontgravingsbak van {EXCAVATION_DEPTH:.2f}m ligt op x={start_excavation:.2f},z={excavation_level:.2f}"
    #         )
    #         break

    # if start_excavation == left:
    #     logging.info(
    #         f"Geen snijpunt gevonden met de lijn van de grondsoorten en de ontgravingsbak."
    #     )
    #     continue

    fig.savefig(Path(PLOT_PATH) / f"{stix_file.stem}.png")
