"""Gender-appropriate SMPL-X model resolution for RICH dataset subjects.

RICH scenes follow the naming convention:
    <location>_<pid1>[_<pid2>...]_<activity>

where person IDs are exactly 3 zero-padded decimal digits.  This module
maps each person ID to the correct SMPL-X model (male/female/neutral) via
a JSON file that records the gender of every RICH subject.

Usage:
    from utilities.rich_gender_plugin import resolve_smplx_models

    models = resolve_smplx_models(
        "BBQ_001_guitar",
        body_models_dir="body_models",
        gender_json_path="resource/rich_gender.json",
    )
    # → {1: PosixPath("body_models/SMPLX_MALE.pkl")}

    # Pass directly to BodyPlacer:
    placer = BodyPlacer(scene_dir, models)
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_GENDER_TO_MODEL: dict[str, str] = {
    "male":    "SMPLX_MALE.pkl",
    "female":  "SMPLX_FEMALE.pkl",
    "neutral": "SMPLX_NEUTRAL.pkl",
}


def parse_rich_person_ids(scene_name: str) -> list[int]:
    """Return RICH person IDs embedded in a scene name.

    Segments that are exactly 3 decimal digits (zero-padded) are treated as
    person IDs; all other segments (location, activity) are ignored.

    Examples:
        "BBQ_001_guitar"          → [1]
        "Pavallion_008_009_walk"  → [8, 9]
        "ParkingLot2_011_run"     → [11]
    """
    return [int(p) for p in scene_name.split("_") if re.fullmatch(r"\d{3}", p)]


def resolve_smplx_models(
    scene_name: str,
    body_models_dir: str | Path,
    gender_json_path: str | Path,
) -> dict[int, Path]:
    """Return {rich_pid: smplx_model_path} for every person in the scene.

    Persons not listed in the gender JSON default to SMPLX_NEUTRAL.pkl.
    """
    with open(gender_json_path) as f:
        gender_map: dict[int, str] = {int(k): v for k, v in json.load(f).items()}

    body_models_dir = Path(body_models_dir)
    result = {}
    for pid in parse_rich_person_ids(scene_name):
        gender = gender_map.get(pid, "neutral")
        model_path = body_models_dir / _GENDER_TO_MODEL[gender]
        logger.info(f"  [gender] scene={scene_name}  pid={pid}  gender={gender}  model={model_path.name}")
        result[pid] = model_path
    return result
