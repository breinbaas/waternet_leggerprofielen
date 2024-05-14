from leveelogic.helpers import case_insensitive_glob
from leveelogic.deltares.dstability import DStability
from tqdm import tqdm
import logging

from settings import STIX_FILES_PATH, LOG_FILE_GRONDSOORTEN

# Dit script leest alle grondsoorten uit een directory met stix bestanden
# en maakt een csv bestand van de grondsoorten en de helling. Dit is nodig
# voor de leggerprofielen omdat we moeten weten welke grondsoort naam
# bij welke helling hoort
logging.basicConfig(
    filename=LOG_FILE_GRONDSOORTEN,
    filemode="w",
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)


# Haal alle stix bestanden op (recursief)
stix_files = case_insensitive_glob(STIX_FILES_PATH, ".stix")

# Maak een lijst voor alle unieke grondsoorten (geen dubbelingen in deze lijst)
unique_soils = []

# Itereer over de stix bestanden
for stix_file in tqdm(stix_files):
    # Lees het bestand in
    try:
        ds = DStability.from_stix(stix_file)
    except Exception as e:
        logging.error(f"Cannot handle file '{stix_file}', got error '{e}'")
        continue
    # ds.soils is een dictionary met alle belangrijke eigenschappen
    for d in ds.soils:
        if (
            not d in unique_soils
        ):  # Voeg enkel toe als de grondsoort nog niet in de lijst staat
            unique_soils.append(d)

# Sorteer op code (makkelijker hanteerbare uitvoer)
unique_soils = sorted(unique_soils, key=lambda d: d["code"])

# Schrijf naar csv bestand
with open("./grondsoorten.csv", "w") as f:
    # Header
    f.write("code,yd,yd,c,phi,helling\n")
    # Itereer over de grondsoorten en bepaal de helling
    for soil in unique_soils:
        if (
            soil["ys"] < 12
        ):  # onder de 12kNm3 is aanname veen of venige klei dus helling 6
            helling = 6
        elif soil["ys"] > 18:  # boven de 18kNm3 is aanname zand dus helling 4
            helling = 4
        elif (
            soil["cohesion"] > 3.0
        ):  # tussen de 12 en 18 met cohesie > 3 is stevige klei dus helling 3
            helling = 3
        else:  # in alle andere gevallen is het of slappere klei of zand dus helling 4
            helling = 4
        # Schrijf de regel en voeg ook alle informatie over de grond toe om controles te kunnen uitvoeren
        f.write(
            f"{soil['code']},{soil['yd']:.3f},{soil['ys']:.3f},{soil['cohesion']:.3f},{soil['friction_angle']:.3f},{helling}\n"
        )
