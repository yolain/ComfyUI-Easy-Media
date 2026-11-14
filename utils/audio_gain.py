import math


def audio_remix_strength(settings: dict) -> float:
    """Read a track's H3 audio preservation strength; legacy tracks hard-lock."""
    value = settings.get("audio_remix_strength")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(value):
            return max(0.0, min(1.0, float(value)))
    return 1.0


def audio_volume_db(settings: dict) -> float:
    raw_db = settings.get("volume_db")
    if isinstance(raw_db, (int, float)) and math.isfinite(float(raw_db)):
        return float(raw_db)
    return 0.0


def audio_is_muted(settings: dict) -> bool:
    return settings.get("muted") is True


def audio_db_to_gain(volume_db: float) -> float:
    return 10.0 ** (volume_db / 20.0)
