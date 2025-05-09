from geolib.models.dstability.internal import AnalysisTypeEnum

# waar staan alle basis stix bestanden
PATH_ORIGINAL_FILES = (
    r"C:\Users\brein\Development\leggerprofielen\data\input\stix\original"
)
PATH_ALL_STIX_FILES = r"C:\Users\brein\Development\leggerprofielen\data\input\stix\all"
PATH_STIX_ERRORS = r"C:\Users\brein\Development\leggerprofielen\data\input\stix\errors"


# waar moeten alle afgeronde berekeningen heen
PATH_SOLUTIONS = r"C:\Users\brein\Development\leggerprofielen\data\output\stix"

# waar moeten de berekeningen heen die fout zijn gegaan
PATH_ERRORS = r"C:\Users\brein\Development\leggerprofielen\data\output\errors"
PATH_DEBUG = r"C:\Users\brein\Development\leggerprofielen\data\output\debug"

# waar moet de uitvoer van de geslaagde berekeningen heen
PATH_SOLUTIONS_PLOTS = r"C:\Users\brein\Development\leggerprofielen\data\output\plots"
PATH_SOLUTIONS_CSV = r"C:\Users\brein\Development\leggerprofielen\data\output\csv"

# waar moet het logbestand komen
PATH_LOG_FILE = r"C:\Users\brein\Development\leggerprofielen\data\output\log"

# waar zijn de invoer csv bestanden te vinden
CSV_FILE_DTH = r"C:\Users\brein\Development\leggerprofielen\data\input\csv\dth.csv"
CSV_FILE_IPO = r"C:\Users\brein\Development\leggerprofielen\data\input\csv\ipo.csv"
CSV_FILE_ONDERHOUDSDIEPTE = (
    r"C:\Users\brein\Development\leggerprofielen\data\input\csv\onderhoudsdiepte.csv"
)

# waar komen de tijdelijke berekeningen (voor debugging)
PATH_TEMP_CALCULATIONS = r"C:\Users\brein\Development\leggerprofielen\data\temp"

# vastgestelde waarden
MIN_SLIP_PLANE_LENGTH = 2.0  # BBF minimale lengte glijvlak
MIN_SLIP_PLANE_DEPTH = 1.5  # BBF minimale diepte glijvlak
PL_SURFACE_OFFSET = 0.1  # Minimale afstand tussen maaiveld en freatische lijn
INITIAL_SLOPE_FACTOR = 1.0  # we starten met de hellingen conform het natuurlijk talud * INITIAL_SLOPE_FACTOR
MAX_ITERATIONS = 10  # we staan maximaal 10 iteraties toe, als het er meer worden dan is er waarschijnlijk toch geen oplossing
CREST_WIDTH = 3.0  # minimale kruinbreedte

# Indien er geen verkeersbelasting wordt gevonden worden de volgende waardes gebruikt
TRAFFIC_LOAD_WIDTH = 2.5  # aan te houden breedte voor de verkeersbelasting
TRAFFIC_LOAD_MAGNITUDE = 13.0  # aan te houden belasting voor de verkeersbelasting
UNITWEIGHT_WATER = 9.81  # gewicht van water

# ophoogmateriaal eigenschappen
OPH_YD = 17.0  # droog vm
OPH_YS = 17.0  # nat vm
OPH_C = 1.5  # cohesie
OPH_PHI = 22.5  # phi
OPH_SLOPE = 3  # helling van het ophoogmateriaal

# de aan te houden schematiseringsfactor
SCHEMATISERINGSFACTOR = 1.2

# aan te houden modelfactoren
MODELFACTOR = {
    AnalysisTypeEnum.BISHOP_BRUTE_FORCE: 1.0,
    AnalysisTypeEnum.SPENCER_GENETIC: 0.95,
    AnalysisTypeEnum.UPLIFT_VAN_PARTICLE_SWARM: 1.05,
}

# vertaling van IPO naar vereiste veiligheidsfactor
IPO_DICT = {
    "I": 0.90,
    "II": 0.90,
    "III": 0.90,
    "IV": 0.95,
    "V": 1.0,
}

# indien er opdrijven is geconstateerd wordt deze offset toegepast
# op de x coordinaten van de uiteindelijke leggerpunten vanaf
# (maar niet inclusief) het uittredepunt
UPLIFT_OFFSET = 10
