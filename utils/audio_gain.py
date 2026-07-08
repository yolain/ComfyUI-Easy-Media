import math


def audio_volume_db(settings: dict) -> float:
    raw_db = settings.get("volume_db")
    if isinstance(raw_db, (int, float)) and math.isfinite(float(raw_db)):
        return float(raw_db)
    return 0.0


def audio_is_muted(settings: dict) -> bool:
    return settings.get("muted") is True


def audio_db_to_gain(volume_db: float) -> float:
    return 10.0 ** (volume_db / 20.0)
