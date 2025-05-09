import os
import glob
from settings import *
from pathlib import Path
from tqdm import tqdm
import shutil

from geolib.models.dstability.internal import AnalysisTypeEnum
from leveelogic.geolib.dstability_model_helper import DStabilityModelHelper
from leveelogic.helpers import case_insensitive_glob

# LET OP; de bbf, spencer, liftvan directory wordt niet automatisch leeg gemaakt

# remove all files
for p in [
    PATH_TEMP_CALCULATIONS,
    PATH_ERRORS,
    PATH_DEBUG,
    PATH_SOLUTIONS_PLOTS,
    PATH_SOLUTIONS_CSV,
    PATH_SOLUTIONS,
    PATH_ALL_STIX_FILES,
]:
    files = glob.glob(os.path.join(p, "*"))
    for file in files:
        if os.path.isfile(file):
            os.remove(file)

# remove directories in debug
for path in sorted(
    Path(PATH_DEBUG).rglob("*"), key=lambda p: len(p.parts), reverse=True
):
    if path.is_dir() and not any(path.iterdir()):  # Check if the directory is empty
        path.rmdir()

# copy the original files to the all files directory
stix_files = case_insensitive_glob(PATH_ORIGINAL_FILES, ".stix")

for stix_file in tqdm(stix_files):
    try:
        dsh = DStabilityModelHelper.from_stix(Path(stix_file))

        if dsh.analysis_type() == AnalysisTypeEnum.BISHOP_BRUTE_FORCE:
            fname = Path(PATH_ALL_STIX_FILES) / "bbf" / stix_file.name
        elif dsh.analysis_type() == AnalysisTypeEnum.SPENCER_GENETIC:
            fname = Path(PATH_ALL_STIX_FILES) / "spencer" / stix_file.name
        elif dsh.analysis_type() == AnalysisTypeEnum.UPLIFT_VAN_PARTICLE_SWARM:
            fname = Path(PATH_ALL_STIX_FILES) / "liftvan" / stix_file.name
        else:
            print(f"Unhandled analysis type '{dsh.analysis_type()}'")
            fname = Path(PATH_ALL_STIX_FILES) / "invalid" / stix_file.name

    except Exception as e:
        print(e)
        break

    shutil.copy(str(stix_file), str(fname))
