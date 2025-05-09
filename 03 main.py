###
# TODO rare rechte lijn bij A124_0200
# TODO test met BBF
##

# Python packages die nodig zijn voor het script
import logging
import os
import glob
from pathlib import Path
import shutil
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from geolib.soils.soil import Soil

# Dit is een aangepaste leveelogic module NIET de module die op pypi staat
# maar meegeleverd wordt met het script
from leveelogic.helpers import case_insensitive_glob
from leveelogic.geolib.dstability_model_helper import DStabilityModelHelper

# We maken gebruik van versie 2.6.0 van geolib
# Let op dat daar nog steeds een bug in zit, zie https://gist.github.com/breinbaas/881d836d9d691768dd232ee31c76b9a5
from geolib.geometry.one import Point

# Lokale packages
from objects.dijktrajecten import Dijktrajecten
from objects.iposearch import IPOSearch
from objects.uitgangspunten import Uitgangspunten
from settings import *
from helpers import (
    stix_has_solution,
    get_natural_slopes_line,
    uplift_at,
    xs_at,
    z_at,
    move_to_error_directory,
    line_above,
)


# indien de volgende waarde True is dan worden alle berekeningen opnieuw gemaakt, ook als er al een oplossing is
FORCE_RECALCULATION = True

# kies uit bbf - liftvan
MODELS_TO_RUN = "liftvan"

# set de backend voor matplotlib op "agg" zodat er geen schermen worden geopend
plt.switch_backend("agg")


#############
# BASISDATA #
#############

# lees de dijktrajecten en de IPO informatie in
# TODO moet via ArcGIS API
try:
    dijktrajecten = Dijktrajecten.from_csv(CSV_FILE_DTH, CSV_FILE_ONDERHOUDSDIEPTE)
    ipo_search = IPOSearch.from_csv(CSV_FILE_IPO)
except Exception as e:
    raise ValueError(
        "Kan de invoer niet lezen, zijn alle bestanden in settings.py gedefinieerd?"
    )


# maak het pad met de tijdelijke berekeningen leeg
files = glob.glob(f"{PATH_TEMP_CALCULATIONS}/*.stix")
for f in files:
    os.remove(f)

# maak het pad met de foute berekeningen leeg
files = glob.glob(f"{PATH_ERRORS}/*.stix")
for f in files:
    os.remove(f)

# zoek alle stix bestanden in de directory met de berekeningen
stix_files = case_insensitive_glob(Path(PATH_ALL_STIX_FILES) / MODELS_TO_RUN, ".stix")

# itereer over alle stix bestanden
for stix_file in stix_files:
    # maak een aparte debug directory per berekening
    debug_path = Path(PATH_DEBUG) / stix_file.stem
    debug_path.mkdir(parents=True, exist_ok=True)

    files = glob.glob(f"{debug_path}/*")
    for f in files:
        os.remove(f)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=str(debug_path / f"00_{stix_file.stem}.log"),
        filemode="w",
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,  # INFO in production
    )

    # is er al een oplossing?
    # zo ja skip deze berekening tenzij FORCE_RECALCULATION == True
    if not FORCE_RECALCULATION and stix_has_solution(stix_file):
        logging.warning(
            "Er is al een oplossing gevonden, deze berekening wordt niet opnieuw gedaan."
        )
        move_to_error_directory(
            stix_file,
            "Er is al een oplossing gevonden, deze berekening wordt niet opnieuw gedaan.",
        )
        continue

    #################
    # READ THE FILE #
    #################

    # de DStabilityModelHelper heeft functies om extra informatie uit de bestanden te halen
    # en om acties op de stix bestanden uit te voeren
    try:
        dsh = DStabilityModelHelper.from_stix(stix_file)
    except Exception as e:
        logging.error(f"Cannot open file '{stix_file.stem}.stix' got error; {e}")
        move_to_error_directory(
            stix_file, f"Cannot open file '{stix_file.stem}.stix' got error; {e}"
        )
        continue

    ##############################
    # EXTRACT INFO FROM FILENAME #
    ##############################
    args = stix_file.stem.split("_")
    try:
        dtcode = args[0]
        chainage = float(args[1])
    except Exception as e:
        logging.error(
            f"Kan de naam van het dijktraject en/of de metrering niet uit de bestandsnaam '{stix_file.stem}' bepalen'",
        )
        move_to_error_directory(
            stix_file,
            f"Kan de naam van het dijktraject en/of de metrering niet uit de bestandsnaam '{stix_file.stem}' bepalen'",
        )
        continue

    # copy the original calculation into the base path
    shutil.copy(stix_file, debug_path / f"01_original.stix")

    ##################################
    # GET THE CALCULATION PARAMETERS #
    ##################################
    try:
        dth = dijktrajecten.get_by_code(dtcode).dth_2024_at(chainage)
        river_level = dijktrajecten.get_by_code(dtcode).mhw_2024_at(chainage)
        onderhoudsdiepte = dijktrajecten.get_by_code(dtcode).onderhoudsdiepte_at(
            chainage
        )
    except Exception as e:
        logging.error(
            f"Fout bij het zoeken naar informatie voor dijktraject '{dtcode}' bij metrering {chainage}; {e}",
        )
        move_to_error_directory(
            stix_file,
            f"Fout bij het zoeken naar informatie voor dijktraject '{dtcode}' bij metrering {chainage}; {e}",
        )
        continue

    if dth is None:
        logging.error(
            f"Geen DTH kunnen bepalen voor dijktraject '{dtcode}' bij metrering {chainage}.",
        )
        move_to_error_directory(
            stix_file,
            f"Geen DTH kunnen bepalen voor dijktraject '{dtcode}' bij metrering {chainage}.",
        )
        continue

    if river_level is None:
        logging.error(
            f"Geen MHW kunnen bepalen voor dijktraject '{dtcode}' bij metrering {chainage}.",
        )
        move_to_error_directory(
            stix_file,
            f"Geen MHW kunnen bepalen voor dijktraject '{dtcode}' bij metrering {chainage}.",
        )
        continue

    if onderhoudsdiepte is None:
        logging.error(
            f"Geen onderhoudsdiepte kunnen bepalen voor dijktraject '{dtcode}' bij metrering {chainage}.",
        )
        move_to_error_directory(
            stix_file,
            f"Geen onderhoudsdiepte kunnen bepalen voor dijktraject '{dtcode}' bij metrering {chainage}.",
        )
        continue

    ipo = ipo_search.get_ipo(dtcode, chainage)
    if ipo is None:
        logging.error(
            f"Geen IPO informatie gevonden voor dijktraject '{dtcode}' bij metrering {chainage}.",
        )
        move_to_error_directory(
            stix_file,
            f"Geen IPO informatie gevonden voor dijktraject '{dtcode}' bij metrering {chainage}.",
        )
        continue

    try:
        required_sf = IPO_DICT[ipo]
    except Exception as e:
        logging.error(
            f"Ongeldige IPO informatie '{ipo}' gevonden voor dijktraject '{dtcode}' bij metrering {chainage}.",
        )
        move_to_error_directory(
            stix_file,
            f"Ongeldige IPO informatie '{ipo}' gevonden voor dijktraject '{dtcode}' bij metrering {chainage}.",
        )
        continue

    # de volgende stap is eigenlijk niet nodig omdat script 2 de schifting uitvoert maar
    # voor de zekerheid toch maar even een controle
    analysis_type = dsh.analysis_type()
    try:
        model_factor = MODELFACTOR[analysis_type]
    except Exception as e:
        logging.error(
            f"Ongeldig analyse type '{analysis_type}' gevonden voor dijktraject '{dtcode}' bij metrering {chainage}.",
        )
        move_to_error_directory(
            stix_file,
            f"Ongeldig analyse type '{analysis_type}' gevonden voor dijktraject '{dtcode}' bij metrering {chainage}.",
        )
        continue

    # vertaal naar een object
    uitgangspunten = Uitgangspunten(
        river_level=river_level,
        dth=dth,
        onderhoudsdiepte=onderhoudsdiepte,
        ipo=ipo,
        required_sf=required_sf,
        kruinbreedte=CREST_WIDTH,
        pl_surface_offset=PL_SURFACE_OFFSET,
        traffic_load_width=TRAFFIC_LOAD_WIDTH,
        traffic_load_magnitude=TRAFFIC_LOAD_MAGNITUDE,
        schematiseringsfactor=SCHEMATISERINGSFACTOR,
        modelfactor=MODELFACTOR[analysis_type],
    )

    # schrijf informatie naar de debug log
    logging.info(
        f"Aangehouden uitgangspunten voor dijktraject '{dtcode}' metrering {chainage}:"
    )
    logging.info(f"\tIPO klasse: {uitgangspunten.ipo}")
    logging.info(f"\tVereiste veiligheidsfactor: {uitgangspunten.required_sf}")
    logging.info(f"\tDijktafelhoogte: {uitgangspunten.dth}")
    logging.info(f"\tMaatgevend hoogwater: {uitgangspunten.river_level}")
    logging.info(f"\tOnderhoudsdiepte: {uitgangspunten.onderhoudsdiepte}")
    logging.info(f"\tKruinbreedte voor minimaal profiel: {uitgangspunten.kruinbreedte}")
    logging.info(
        "\tPolderniveau voor minimaal profiel: gelijk aan laagste punt op maaiveld dat geen onderdeel is van de sloot"
    )
    logging.info(
        "\tDe oude verkeersbelasting wordt gehanteerd en elke laag met een cohesie die groter is dan 0 wordt op 50% consolidatie gezet"
    )
    logging.info(
        f"Minimale lengte glijvlak is ingesteld op {MIN_SLIP_PLANE_LENGTH} meter"
    )
    logging.info(
        f"Minimale diepte glijvlak is ingesteld op {MIN_SLIP_PLANE_DEPTH} meter"
    )
    logging.info(
        f"De afstand tussen de freatische lijn en het maaiveld is ingesteld op {PL_SURFACE_OFFSET} meter"
    )
    logging.info(
        f"NB. Er wordt GEEN rekening gehouden met het verloop van de stijghoogte naar de freatische lijn."
    )
    logging.info(
        f"Gebruikt ophoogmateriaal yd={OPH_YD}, ys={OPH_YS}, c={OPH_C}, phi={OPH_PHI}"
    )

    ####################
    # CHECK CURRENT SF #
    ####################
    try:
        # bereken de huidige veiligheidsfactor en rapporteer deze inclusief de modelfactor
        dsh.model.execute()
        org_sf = dsh.model.get_result(0, 0).FactorOfSafety
        org_sf = org_sf / SCHEMATISERINGSFACTOR / MODELFACTOR[dsh.analysis_type()]
        logging.info(
            f"De huidige veiligheidsfactor (inclusief een modelfactor van {MODELFACTOR[dsh.analysis_type()]}) uit de originele berekening bedraagt {org_sf:.2f}"
        )

        # check of er een verkeersbelasting is
        loads = dsh.model._get_loads(scenario_index=0, stage_index=0)
        if len(loads.UniformLoads) == 0:
            logging.warning("Er is geen verkeerslast gevonden in deze berekening!")
    except Exception as e:
        logging.error(
            f"Fout bij het berekenen van de originele veiligheidsfactor; '{e}'",
        )
        move_to_error_directory(
            stix_file,
            f"Fout bij het berekenen van de originele veiligheidsfactor; '{e}'",
        )
        continue

    # voeg de grondsoort ophoogmateriaal toe aan het model
    soil = Soil()
    soil.name = "Ophoogmateriaal"
    soil.code = "Ophoogmateriaal"
    soil.shear_strength_model_below_phreatic_level = "Mohr_Coulomb"
    soil.soil_weight_parameters.saturated_weight.mean = OPH_YS
    soil.soil_weight_parameters.unsaturated_weight.mean = OPH_YD
    soil.mohr_coulomb_parameters.cohesion.mean = OPH_C
    soil.mohr_coulomb_parameters.friction_angle.mean = OPH_PHI
    soil.mohr_coulomb_parameters.dilatancy_angle.mean = OPH_PHI

    try:
        dsh.model.add_soil(soil)
    except Exception as e:
        logging.error(
            f"Fout bij het toevoegen van de grondsoort '{soil.name}' aan het model; '{e}'",
        )
        move_to_error_directory(
            stix_file,
            f"Fout bij het toevoegen van de grondsoort '{soil.name}' aan het model; '{e}'",
        )
        continue

    # Als we met bishop brute force werken dan voegen we constraints toe om de glijvlakken realistischer te maken
    if dsh.analysis_type() == AnalysisTypeEnum.BISHOP_BRUTE_FORCE:
        dsh.add_bbf_slipplane_constraints(
            min_slipplane_depth=MIN_SLIP_PLANE_DEPTH,
            min_slipplane_length=MIN_SLIP_PLANE_LENGTH,
        )

    #############################
    # ITEREER TOT EEN OPLOSSING #
    #############################
    slope_factor = INITIAL_SLOPE_FACTOR
    iteration = 1
    counter = 2  # voor de opeenvolgende bestanden in de debug directory
    solution = None
    done = False
    while not done:
        try:
            logging.info(
                f"Generating and calculating iteration {iteration} with slope factor {slope_factor:.2f}"
            )
            # create a copy of the base levee
            dsh_copy = deepcopy(dsh)
            # dsh_copy.change_limits(right=dsh_copy.right() + 20.0)

            x0, z0 = None, None
            x1 = 0.0
            z1 = uitgangspunten.dth
            x2 = uitgangspunten.kruinbreedte
            z2 = uitgangspunten.dth
            x4 = dsh_copy.right()

            dth_above_surface = z1 > dsh_copy.z_at(x1)

            # als z1 lager is dan de dijktafelhoogte dan moeten we op zoek naar een snijpunt links van 0 met helling 1:2
            if dth_above_surface:
                zbot = dsh_copy.bottom()
                line_left = [(x1, z1), (x1 - (z1 - zbot) * 2, zbot)]
                intersections = dsh_copy.get_surface_intersections(line_left)
                if len(intersections) == 0:
                    logging.error(
                        f"Geen snijpunt gevonden met de lijn van de dijktafelhoogte naar beneden met helling 1:2"
                    )
                    # move_to_error_directory(
                    #     stix_file,
                    #     "Geen snijpunt gevonden met de lijn van de dijktafelhoogte naar beneden met helling 1:2",
                    # )
                    continue
                x0, z0 = intersections[-1]
            else:
                line_left = [(x1, z1), (dsh_copy.left(), z1)]
                intersections = dsh_copy.get_surface_intersections(line_left)
                intersections = [i for i in intersections if i[0] <= 0]
                if len(intersections) == 0:
                    logging.error(
                        f"Geen snijpunt gevonden met de lijn van de dijktafelhoogte naar links"
                    )
                    # move_to_error_directory(
                    #     stix_file,
                    #     "Geen snijpunt gevonden met de lijn van de dijktafelhoogte naar links",
                    # )
                    continue
                x0, z0 = intersections[-1]

            # we zetten het maaiveld in de polder gelijk aan het laagste punt op
            # het maaiveld dat geen onderdeel is van een eventuele aanwezige sloot
            wnet_creator_settings = dsh_copy.get_waternet_creator_settings()

            if wnet_creator_settings is None:
                logging.error(
                    "Geen waternet creator settings gevonden, dit moet ingevuld zijn"
                )
                # move_to_error_directory(
                #     stix_file,
                #     "Geen waternet creator settings gevonden, dit moet ingevuld zijn",
                # )
                continue

            if (
                wnet_creator_settings.EmbankmentCharacteristics.EmbankmentToeLandSide
                == "NaN"
            ):
                logging.error(
                    "De binnenteen is niet gedefinieerd in de waternet settings!"
                )
                # move_to_error_directory(
                #     stix_file,
                #     "De binnenteen is niet gedefinieerd in de waternet settings!",
                # )
                raise ValueError(
                    "De binnenteen is niet gedefinieerd in de waternet settings!"
                )
            else:
                x_binnenteen = (
                    wnet_creator_settings.EmbankmentCharacteristics.EmbankmentToeLandSide
                )

            if x_binnenteen < CREST_WIDTH:
                logging.error(
                    "De binnenteen ligt voor het binnenkruin punt, dit is niet toegestaan!"
                )
                # move_to_error_directory(
                #     stix_file,
                #     "De binnenteen ligt voor het binnenkruin punt, dit is niet toegestaan!",
                # )
                raise ValueError(
                    "De binnenteen ligt voor het binnenkruin punt, dit is niet toegestaan!"
                )

            x_sloot_bodem_dijkzijde = None
            x_sloot_insteek_dijkzijde = None
            x_sloot_insteek_polderzijde = None
            if wnet_creator_settings.DitchCharacteristics is not None:
                if (
                    wnet_creator_settings.DitchCharacteristics.DitchBottomEmbankmentSide
                    != "NaN"
                ):
                    x_sloot_bodem_dijkzijde = (
                        wnet_creator_settings.DitchCharacteristics.DitchBottomEmbankmentSide
                    )
                if (
                    wnet_creator_settings.DitchCharacteristics.DitchEmbankmentSide
                    != "NaN"
                ):
                    x_sloot_insteek_dijkzijde = (
                        wnet_creator_settings.DitchCharacteristics.DitchEmbankmentSide
                    )
                if wnet_creator_settings.DitchCharacteristics.DitchLandSide != "NaN":
                    x_sloot_insteek_polderzijde = (
                        wnet_creator_settings.DitchCharacteristics.DitchLandSide
                    )

            # we hebben het laagste punt op het maaiveld nodig maar als er een sloot is dan willen
            # we die punten niet meenemen
            if (
                x_sloot_insteek_dijkzijde is not None
                and x_sloot_insteek_polderzijde is not None
            ):  # we hebben een sloot dus punten tussen de insteek zijdes verwijderen
                surface_points = [
                    p
                    for p in dsh_copy.surface()
                    if p[0] <= x_sloot_insteek_dijkzijde
                    and p[0] >= x_sloot_insteek_polderzijde
                ]
            else:
                surface_points = dsh_copy.surface()
            z4 = min([p[1] for p in dsh_copy.surface() if p[0] > x_binnenteen])

            # bepaal de natuurlijke hellingen vanaf de binnenkruin
            natural_slopes = get_natural_slopes_line(
                dsh=dsh_copy,
                z_top=uitgangspunten.dth,
                x=x2,
                top_slope=OPH_SLOPE,
                slope_factor=slope_factor,
            )
            slope_points = [(x2, z2)]
            for i in range(1, len(natural_slopes)):
                p1x, p1z = natural_slopes[i - 1]
                p2x, p2z = natural_slopes[i]
                if p1z >= z4 and z4 >= p2z:
                    z = z4
                    x = p1x + (p1z - z) / (p1z - p2z) * (p2x - p1x)
                    slope_points.append((x, z))
                    break

                slope_points.append(natural_slopes[i])

            profile_line = [(x0, z0), (x1, z1)] + slope_points  # + [(x4, z4)]
            if profile_line[-1][0] > dsh_copy.right():
                dsh_copy.change_limits(right=profile_line[-1][0] + 1.0)

            p_right = dsh_copy.surface()[-1]
            profile_line.append((p_right[0] - 0.01, profile_line[-1][1]))
            profile_line.append(p_right)

            x3, z3 = slope_points[-1]

            logging.debug(f"Points on profile line, {profile_line}")

            # find the intersections with the profile line starting from the left of the profile line
            intersections = [
                p
                for p in dsh_copy.get_surface_intersections(profile_line)
                if p[0] > profile_line[0][0]
            ]

            # create a plot for debugging purposes
            fig, ax = plt.subplots(figsize=(15, 5))
            ax.plot(
                [p[0] for p in dsh_copy.surface()],
                [p[1] for p in dsh_copy.surface()],
                "k",
            )
            x_intersections = [p[0] for p in intersections]
            z_intersections = [p[1] for p in intersections]
            ax.plot(x_intersections, z_intersections, "bo")
            ax.plot([p[0] for p in profile_line], [p[1] for p in profile_line], "r")
            ax.grid(which="both", linestyle="--", linewidth=0.5)
            ax.set_aspect("equal", adjustable="box")
            fig.savefig(
                Path(debug_path)
                / f"{counter:02d}_{stix_file.stem}.profile_line_{slope_factor:.2f}.png"
            )
            counter += 1
            plt.clf()

            if dth_above_surface:
                mode = "fill"
            else:
                mode = "cut"

            # add intersections to the profile line
            profile_line = sorted(
                list(set(profile_line + intersections)), key=lambda x: x[0]
            )

            start_point = profile_line[0]

            for i, intersection in enumerate(intersections):

                # TODO, dit werk niet, bepaal de punten
                # check of de punten boven of onder het maaiveld liggen
                # en bepaal daarmee of het een ontgraving of ophoging is

                fill_or_excavate_points = [
                    (round(p[0], 3), round(p[1], 3))
                    for p in profile_line
                    if p[0] >= start_point[0] and p[0] <= intersection[0]
                ]

                check_points = (
                    [fill_or_excavate_points[0]]
                    + [
                        (round(p[0], 3), round(p[1], 3))
                        for p in dsh_copy.surface()
                        if p[0] > fill_or_excavate_points[0][0]
                        and p[0] < fill_or_excavate_points[-1][0]
                    ]
                    + [fill_or_excavate_points[-1]]
                )

                if fill_or_excavate_points == check_points:
                    continue

                fill = line_above(fill_or_excavate_points, check_points)

                if fill:
                    surface_points = [
                        (round(p[0], 3), round(p[1], 3))
                        for p in dsh_copy.surface()
                        if p[0] > fill_or_excavate_points[0][0]
                        and p[0] < fill_or_excavate_points[-1][0]
                    ]
                    add_points = fill_or_excavate_points + surface_points[::-1]

                    if len(add_points) < 3:
                        logging.warning(
                            f"Er is een ophoging gevonden met minder dan 3 punten, te weten, {add_points}"
                        )
                    else:
                        try:
                            logging.info(f"Toevoegen laag met punten {add_points}")
                            dsh_copy.model.add_layer(
                                points=[Point(x=p[0], z=p[1]) for p in add_points],
                                soil_code="Ophoogmateriaal",
                                label="ophoging",
                            )
                        except Exception as e:
                            logging.error(
                                f"Toevoegen van ophoogmateriaal leidt tot een ongeldige geometrie, '{e}'"
                            )
                            logging.error(f"Punten:", add_points)
                            raise e
                else:
                    try:
                        logging.info(
                            f"Verwijderen laag met punten {fill_or_excavate_points}"
                        )
                        dsh_copy.model.add_excavation(
                            [Point(x=p[0], z=p[1]) for p in fill_or_excavate_points],
                            label="ontgraving",
                        )
                    except Exception as e:
                        logging.error(
                            f"Verwijderen van laag leidt tot een ongeldige geometrie, '{e}'"
                        )
                        logging.error(f"Punten:", fill_or_excavate_points)
                        raise e

                start_point = intersection

            # Slootpeil gelijk aan dat in de sloot of maaiveld minus 0.15m
            # bepaal het polderpeil
            if x_sloot_bodem_dijkzijde is not None:
                polder_level = dsh_copy.phreatic_level_at(
                    x_sloot_bodem_dijkzijde, return_last_point_if_no_result=True
                )

                if polder_level > z4 - 0.15:
                    logging.info(
                        f"Het peil in de sloot is gebaseerd op het originele slootpeil ({polder_level:.2f}) gecorrigeerd voor een lager liggend maaiveld ({z4}) en bedraagt {(z4 - 0.15):.2f}"
                    )
                    polder_level = z4 - 0.15
                else:
                    logging.info(
                        f"Het peil in de sloot is gebaseerd op het originele slootpeil en bedraagt {polder_level:.2f}"
                    )
            else:
                polder_level = z4 - 0.15
                logging.info(
                    f"Het peil in de sloot is gebaseerd op het laagste punt in het maaiveld niet horend bij de sloot en bedraagt {polder_level:.2f}"
                )

            # generate the phreatic line
            # Updated 6-8-2024
            # point 2 = intersection with surface
            pl_p1 = (dsh_copy.left(), uitgangspunten.river_level)
            try:
                intersections = dsh_copy.get_surface_intersections(
                    [(dsh_copy.left(), river_level), (dsh_copy.right(), river_level)]
                )
                pl_p2 = (intersections[0][0], river_level)
            except Exception as e:
                logging.error(
                    f"Kan geen snijpunt vinden met het nieuwe maaiveld en de rivier waterstand van {river_level}"
                )
                raise e

            pl_p3 = (0, uitgangspunten.river_level - 0.2)
            pl_p4 = (uitgangspunten.kruinbreedte, uitgangspunten.river_level - 0.6)

            pl_points_right_from_pl_p4 = [p for p in profile_line if p[0] > pl_p4[0]]
            plline_points = [pl_p1, pl_p2, pl_p3, pl_p4]
            plline_points += [
                (p[0], p[1] - PL_SURFACE_OFFSET) for p in pl_points_right_from_pl_p4
            ]

            # check that points going to the right are below the previous point
            final_pl_points = [plline_points[0]]
            for p in plline_points[1:]:
                y_prev = final_pl_points[-1][1]
                if p[1] > y_prev:
                    final_pl_points.append((p[0], y_prev))
                else:
                    final_pl_points.append(p)

            dsh_copy.set_phreatic_line(final_pl_points)
            dsh_copy.move_traffic_load(CREST_WIDTH - TRAFFIC_LOAD_WIDTH)

            calculation_name = (
                f"{stix_file.stem}_iteration_{iteration}_slope_{slope_factor:.2f}.stix"
            )
            dsh_copy.serialize(
                Path(PATH_TEMP_CALCULATIONS) / f"{calculation_name}.stix"
            )
            dsh_copy.serialize(debug_path / f"{counter:02d}_{calculation_name}")
            counter += 1

            try:
                dsh_copy.model.execute()
                dsh_result = dsh_copy.model.get_result(0, 0)
                sf = round(dsh_result.FactorOfSafety, 3)
                logging.info(f"De berekende veiligheidsfactor = {sf:.3f}")
                sf = sf / SCHEMATISERINGSFACTOR / MODELFACTOR[dsh_copy.analysis_type()]
                logging.info(
                    f"Met modelfactor {MODELFACTOR[dsh_copy.analysis_type()]} en schematisatie factor {SCHEMATISERINGSFACTOR} wordt de veiligheidsfactor {sf:.3f}"
                )
            except Exception as e:
                logging.error(
                    f"Could not calculate slope {slope_factor:.2f}, got error {e}"
                )
                raise e

                # update, we start with the natural slopes
                # if this leads to SF >= SF_REQUIRED we are done and have the solution
            if sf >= required_sf:  # and sf <= required_sf + SF_MARGIN:
                logging.info(
                    f"Found a solution after {iteration} iteration(s) with slope factor={slope_factor:.2f}"
                )

                try:
                    x_uittredepunt = dsh_result.Points[-1].X
                    z_uittredepunt = dsh_result.Points[-1].Z
                    z_intredepunt = dsh_result.Points[0].Z
                    logging.info(f"Het intredepunt ligt op {x_uittredepunt:.2f}")

                    if z_intredepunt < z_uittredepunt:
                        logging.error(
                            "Het intredepunt ligt lager dan het uittredepunt, wellicht is dit een omgedraaide geometrie!"
                        )
                        raise ValueError(
                            "Het intredepunt ligt lager dan het uittredepunt, wellicht is dit een omgedraaide geometrie!"
                        )

                    solution = dsh_copy

                except Exception as e:
                    logging.error(f"kan uittredepunt niet bepalen, foutmelding: {e}")
                    raise e

                done = True
            elif sf < required_sf:
                slope_factor *= 1.2
            else:
                slope_factor /= 1.1

            iteration += 1

            if not done and iteration > MAX_ITERATIONS:
                logging.error(
                    f"After {MAX_ITERATIONS} iterations we still have no solution, skipping this levee"
                )
                done = True
                break
        except Exception as e:
            logging.error(f"Onverwachte fout opgetreden '{e}'")
            try:
                dsh_copy.serialize(
                    Path(debug_path)
                    / f"{counter:02d}_{stix_file.stem}_iteration_{iteration}_with_error.stix"
                )
                counter += 1
            except:
                logging.debug(
                    "Could not save error file, probably because it is impossible to generate the model"
                )

            iteration += 1
            slope_factor *= 1.2  # maybe solved with a new slope
            done = iteration > MAX_ITERATIONS

    if solution is None:
        logging.error("Geen oplossing gevonden, controleer de bovenstaande log.")
        move_to_error_directory(
            stix_file,
            "Geen oplossing gevonden",
        )
        continue

    # nu we de methode gebruiken via excavations is het maaiveld niet meer gelijk
    # aan het maaiveld met ophogingen en excavations dus moeten we de profile line
    # gebruiken om de juiste start z coordinaat te vinden!
    z_refline = z_at(profile_line, 0)
    z_mv_uittredepunt = z_at(profile_line, x_uittredepunt)
    if z_mv_uittredepunt is None:
        z_mv_uittredepunt = profile_line[-1][1]

    # bepaal het natuurlijke hellingen bij het uittredepunt
    natural_slopes_line_right = get_natural_slopes_line(
        dsh=solution,
        z_top=z_mv_uittredepunt,
        x=x_uittredepunt,
        top_slope=OPH_SLOPE,
    )
    natural_slopes_line_left = [
        (-1 * p[0], p[1])
        for p in get_natural_slopes_line(
            dsh=solution,
            z_top=z_refline,
            x=0.0,
            top_slope=OPH_SLOPE,
        )
    ][::-1]

    # create the final line
    final_line = natural_slopes_line_left
    final_line += [p for p in profile_line if p[0] > 0.0 and p[0] < x_uittredepunt]
    final_line += natural_slopes_line_right

    # het kan voorkomen dat het punt dth-1.5 niet op de final line ligt omdat het pleistoceen hoger ligt dan
    # DTH-1.5m bijvoorbeeld bij de zuidelijke dijken van Waternet
    # we moeten daarvoor een lijn pleist_to_pleist_line maken die aan de linkerzijde doorgetrokken wordt tot de *onderkant* van het pleistoceen
    fp1x, fp1z = final_line[0]
    fp2x, fp2z = final_line[1]
    pleist_to_pleist_line = [p for p in final_line]
    slope = (fp2x - fp1x) / (fp2z - fp1z)
    x3 = fp1x + (dsh_copy.bottom() - fp1z) * slope
    pleist_to_pleist_line.insert(0, (x3, dsh_copy.bottom()))

    # we moeten de lijn nog aan kunnen passen dus we moeten van eventuele tuple punten af
    final_line = [[p[0], p[1]] for p in final_line]

    # bewaar de leggerprofiel punten in de volgende lijst
    points_to_plot = []

    ####################################################
    # P1 = snijpunt natuurlijke helling vanaf de       #
    # buitenteen met de bovenzijde van het pleistoceen #
    # Dit is de start van het profiel                  #
    # Het is start BBZ als P2.x > P1.x anders is het   #
    # de start van de BZ                               #
    # P2 = snijpunt natuurlijke helling vanaf de       #
    ####################################################
    p1 = natural_slopes_line_left[0]
    points_to_plot.append({"code": 25, "label": "Start BZZ", "point": p1})

    # hebben we te maken met opdrijven?
    has_uplift = uplift_at(
        dsh=dsh_copy, x=x_binnenteen, hydraulic_head=uitgangspunten.river_level
    )

    ########################################################
    # P2 = snijpunt natuurlijke helling van de buitenteen  #
    # met de bovenzijde van het pleistoceen en de z waarde #
    # voor de onderhoudsdiepte minus 2.0m                  #
    # Let op dat dit punt voor P1 kan liggen als de        #
    # z waarde lager ligt dan het pleistoceen              #
    # Als P2.x > P1.x dan is dit begin BZ en anders wordt  #
    # dit punt genegeerd en is P1 begin BZ ipv begin BBZ   #
    ########################################################
    z2 = onderhoudsdiepte - 2.0
    try:
        x2 = xs_at(pleist_to_pleist_line, z2)[0]
    except Exception as e:
        logging.error(
            f"De x coordinaat voor het punt onderhoudsdiepte - 2.0m voor overgang BZ naar BBZ aan waterzijde kan niet gevonden worden, melding: {e}"
        )
        move_to_error_directory(
            stix_file,
            f"De x coordinaat voor het punt onderhoudsdiepte - 2.0m voor overgang BZ naar BBZ aan waterzijde kan niet gevonden worden, melding: {e}",
        )
        continue

    x2_before_x1 = x2 <= p1[0]
    if x2_before_x1:
        # x2 ligt voor of op x1 dus we bewaren x2 niet maar beschouwen punt 1 als begin BZ ipv begin BBZ
        points_to_plot[0]["label"] = "Start BZ"
    else:
        p2 = (x2, z2)
        points_to_plot.append({"code": 25, "label": "BZ", "point": p2})

    #######################################################
    # P3 = snijpunt natuurlijke helling van de buitenteen #
    # met de DTH minus 1.5m, situaties waar de bovenzijde #
    # van het pleistoceen hoger ligt dan DTH-1.5m worden  #
    # als fouten behandeld                                #
    #######################################################
    try:
        x3 = xs_at([p for p in pleist_to_pleist_line if p[0] <= 0.0], dth - 1.5)[0]
    except Exception as e:
        logging.error(
            f"De x coordinaat voor het punt DHT-1.5m voor overgang KZ naar BZ aan waterzijde kan niet gevonden worden, melding: {e}"
        )
        move_to_error_directory(
            stix_file,
            f"De x coordinaat voor het punt DHT-1.5m voor overgang KZ naar BZ aan waterzijde kan niet gevonden worden, melding: {e}",
        )
        continue
    p3 = (x3, dth - 1.5)
    points_to_plot.append({"code": 25, "label": "KZ", "point": p3})

    ###########################
    # P4 = het referentiepunt #
    ###########################
    p4 = (0.0, dth)
    points_to_plot.append({"code": 90, "label": "Referentielijn", "point": p4})

    #################################################
    # P5 = uittredepunt van het gevonden glijvlak   #
    # Dit is het einde van de KZ en begin van de BZ #
    #################################################
    z5 = z_at(final_line, x_uittredepunt)
    if z5 is None:
        logging.error("De z coordinaat op het uittredepunt kan niet gevonden worden")
        continue
    p5 = [x_uittredepunt, z5]
    points_to_plot.append({"code": 25, "label": "BZ", "point": p5})

    ###########################################################
    # Als er sprake is van opdrijven dan worden punten 6 en 7 #
    # gebaseerd op de profiel lijn waarbij de punten voorbij  #
    # punt 5 met 10m naar rechts zijn verschoven              #
    ###########################################################
    if has_uplift:
        final_line_start = [p for p in final_line if p[0] <= x_uittredepunt]
        offset_point = [final_line_start[-1][0] + 10.0, final_line_start[-1][1]]
        final_line_end = [
            [p[0] + UPLIFT_OFFSET, p[1]] for p in final_line if p[0] > x_uittredepunt
        ]

        # create a plot for debugging purposes
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(
            [p[0] for p in dsh_copy.surface()],
            [p[1] for p in dsh_copy.surface()],
            "k",
        )
        # oude lijn
        ax.plot([p[0] for p in final_line], [p[1] for p in final_line], "k--")

        final_line = final_line_start + [offset_point] + final_line_end

        # nieuwe lijn
        ax.plot([p[0] for p in final_line], [p[1] for p in final_line], "r--")
        ax.grid(which="both", linestyle="--", linewidth=0.5)

        ax.set_aspect("equal", adjustable="box")
        fig.savefig(
            Path(debug_path)
            / f"{counter:02d}_{stix_file.stem}.profile_line_after_uplift{slope_factor:.2f}.png"
        )
        counter += 1
        plt.clf()

    ##################################################################
    # P6 = snijpunt van de natuurlijke hellingen aan de polderzijde  #
    # op het polderniveau minus 2.0m                                 #
    # Het kan voorkomen dat dit punt niet bestaat omdat              #
    # dit lager ligt dan de bovenzijde van het pleistoceen           #
    # in dat geval wordt dit punt genegeerd                          #
    # Bestaat P6 dan is dat begin BBZ                                #
    ##################################################################
    try:
        x6 = xs_at(final_line, polder_level - 2.0)[-1]
        p6 = [x6, polder_level - 2.0]
        points_to_plot.append({"code": 25, "label": "BBZ", "point": p6})
        p7 = [final_line[-1][0], final_line[-1][1]]  # BBZ eind landzijde
        points_to_plot.append({"code": 25, "label": "Eind", "point": p7})
        has_excavation_intersection = True
    except Exception as e:
        # bakje snijdt met pleistoceen -> geen BBZ en einde ligt op 10m voorbij einde BZ
        # als we de ligging van het laatste punt nog niet hebben verschoven vanwege uplift
        # dan doen we dat nu vanwege het feit dat de onderkant van het bakje het pleistoceen snijdt
        if not has_uplift:
            final_line_start = [p for p in final_line if p[0] <= x_uittredepunt]
            offset_point = [final_line_start[-1][0] + 10.0, final_line_start[-1][1]]
            final_line_end = [
                [p[0] + UPLIFT_OFFSET, p[1]]
                for p in final_line
                if p[0] > x_uittredepunt
            ]
            final_line = final_line_start + [offset_point] + final_line_end

            # verplaats uittredepunt naar rechts
            points_to_plot[-1]["point"][0] += UPLIFT_OFFSET

        p7 = [final_line[-1][0], final_line[-1][1]]

        points_to_plot.append({"code": 25, "label": "Eind", "point": p7})
        logging.warning(
            f"De x coordinaat voor het punt polderpeil-2.0m voor overgang BZ naar BBZ aan landzijde kan niet gevonden worden, melding: {e}"
        )
        logging.warning(
            "Aanname dat dit komt omdat er geen snijpunt gevonden kan worden omdat onderzijde bakje lager ligt dan de Pleistocene zandlaag."
        )
        logging.warning(
            f"We verplaatsen het leggerprofiel tot {UPLIFT_OFFSET}m voorbij het uitredepunt om dit te ondervangen."
        )
        has_excavation_intersection = False

    # plot the solution
    fig, ax = plt.subplots(figsize=(15, 5))

    for _, layer in dsh.layer_dict().items():
        soil = layer["soil"]
        p = Polygon(layer["points"], facecolor="#fff", edgecolor="k")
        ax.add_patch(p)

    ax.plot([p[0] for p in dsh.surface()], [p[1] for p in dsh.surface()], "k")
    ax.plot([p[0] for p in final_line], [p[1] for p in final_line], "r--")
    ax.plot(
        [p[0] for p in dsh.phreatic_line()],
        [p[1] for p in dsh.phreatic_line()],
        "b--",
    )

    # add the zones
    zmin = min([p[1] for p in final_line]) - 1.0
    zmax = max([p[1] for p in final_line]) + 1.0
    ax.plot([p1[0], p1[0]], [zmin, zmax], "k--")
    ax.scatter(
        [p["point"][0] for p in points_to_plot],
        [p["point"][1] for p in points_to_plot],
    )

    # plot de glijcirkel
    solution_result = dsh_copy.model.get_result(0, 0)
    ax.plot(
        [p.X for p in solution_result.Points],
        [p.Z for p in solution_result.Points],
        "g",
    )

    ax.grid(which="both", linestyle="--", linewidth=0.5)

    for p in points_to_plot:
        ax.plot([p["point"][0], p["point"][0]], [zmin, zmax], "k--")
        ax.text(p["point"][0], p["point"][1], f"{p['code']} {p['label']}")

    if has_uplift:
        ax.text(x_uittredepunt, zmax, "Opdrijven geconstateerd")
    if not has_excavation_intersection:
        ax.text(x_uittredepunt, zmax - 1.0, "2m ontgraving snijdt pleistoceen")

    fig.savefig(Path(PATH_SOLUTIONS_PLOTS) / f"{stix_file.stem}_solution.png")
    # for debugging
    fig.savefig(debug_path / f"{counter:02d}_{stix_file.stem}_solution.png")
    counter += 1

    # create the points with codes
    csv_points = []

    for i in range(1, len(points_to_plot)):
        d1 = points_to_plot[i - 1]
        d2 = points_to_plot[i]

        if i == 1:
            csv_points.append(d1)

        csv_points += [
            {"point": p, "code": "99", "label": ""}
            for p in final_line
            if p[0] > d1["point"][0] and p[0] < d2["point"][0]
        ]
        csv_points.append(d2)

    # csv debug plot
    fig, ax = plt.subplots(figsize=(15, 5))
    xs = [p["point"][0] for p in csv_points]
    zs = [p["point"][1] for p in csv_points]
    ax.plot(xs, zs, "k")
    ax.scatter(xs, zs)
    ax.grid(which="both", linestyle="--", linewidth=0.5)
    fig.savefig(
        Path(debug_path) / f"{counter:02d}_{stix_file.stem}.csv_line_solution.png"
    )
    counter += 1
    plt.clf()

    # write a csv file
    lines = ["code,x,z\n"]
    for p in csv_points:
        lines.append(f"{p['code']},{p['point'][0]:.2f},{p['point'][1]:.2f}\n")

    with open(Path(PATH_SOLUTIONS_CSV) / f"{stix_file.stem}_solution.csv", "w") as f:
        for l in lines:
            f.write(l)

    # for debugging
    with open(debug_path / f"{counter:02d}_{stix_file.stem}_solution.csv", "w") as f:
        for l in lines:
            f.write(l)
        counter += 1

    solution.serialize(Path(PATH_SOLUTIONS) / f"{stix_file.stem}_solution.stix")
    # for debugging
    solution.serialize(debug_path / f"{counter:02d}_{stix_file.stem}_solution.stix")

    # move the input file
    stix_file.replace(Path(PATH_SOLUTIONS) / stix_file.name)
    plt.close("all")
