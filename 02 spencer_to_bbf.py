from leveelogic.helpers import case_insensitive_glob
from leveelogic.geolib.dstability_model_helper import DStabilityModelHelper
from geolib.models.dstability.internal import AnalysisTypeEnum
from tqdm import tqdm
from geolib.models.dstability.dstability_model import PersistablePoint
from pathlib import Path
from settings import *

INPUT_PATH = Path(PATH_ALL_STIX_FILES) / "spencer"
OUTPUT_PATH = Path(PATH_ALL_STIX_FILES) / "bbf"

stix_files = case_insensitive_glob(INPUT_PATH, ".stix")

for stix_file in tqdm(stix_files):
    dsh = DStabilityModelHelper.from_stix(Path(stix_file))

    if dsh.analysis_type() == AnalysisTypeEnum.SPENCER_GENETIC:
        dsh.sga_to_bff(add_constraints=True)
        filename = Path(stix_file).stem
        dsh.serialize(Path(OUTPUT_PATH) / f"{filename}_spencer2bff.stix")
