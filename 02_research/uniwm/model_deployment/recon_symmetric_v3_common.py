"""Isolated paths for NWM/RAE completion of the symmetric Habitat cohort."""
from pathlib import Path

RUN='recon_symmetric_v3'
SEED=20261023


def experiment(study):return Path(study)/'v2'/RUN
