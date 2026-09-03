from __future__ import annotations


SLOT_REFERENCE_PREFIX = "__slot__:"
MAX_MULTITRACK_TASK_IMAGES = 9


def multitrack_slot_name(content: dict) -> str | None:
    """Return a canonical slot name from current or legacy media descriptors."""
    if not isinstance(content, dict):
        return None

    slot_name = content.get("slot_name")
    if isinstance(slot_name, str) and slot_name:
        return slot_name.removeprefix(SLOT_REFERENCE_PREFIX)

    for key in ("file_path", "local_path", "url", "file_name"):
        value = content.get(key)
        if isinstance(value, str) and value.startswith(SLOT_REFERENCE_PREFIX):
            return value.removeprefix(SLOT_REFERENCE_PREFIX) or None

    if content.get("source_type") == "slot":
        file_name = content.get("file_name")
        if isinstance(file_name, str) and file_name:
            return file_name
    return None


def canonicalize_multitrack_slot_content(content: dict) -> dict:
    """Normalize encoded ``__slot__:name`` paths into an explicit slot descriptor."""
    normalized = dict(content)
    slot_name = multitrack_slot_name(normalized)
    if slot_name is None:
        return normalized

    normalized["source_type"] = "slot"
    normalized["slot_name"] = slot_name
    normalized["file_name"] = slot_name
    for key in ("file_path", "local_path", "url"):
        value = normalized.get(key)
        if isinstance(value, str) and value.startswith(SLOT_REFERENCE_PREFIX):
            normalized.pop(key, None)
    return normalized


def multitrack_is_shared_reference(content: dict) -> bool:
    """Accept the unified flag and the legacy audio speaker-reference flag."""
    return isinstance(content, dict) and (
        content.get("shared_reference") is True
        or content.get("speaker_reference") is True
    )


def multitrack_media_identity(content: dict) -> tuple[str, str] | None:
    """Return the source/path identity used to match shared media references."""
    if not isinstance(content, dict):
        return None
    normalized = canonicalize_multitrack_slot_content(content)
    source_type = str(normalized.get("source_type", "input"))
    if source_type == "slot":
        path = normalized.get("slot_name")
    else:
        path = (
            normalized.get("file_path")
            or normalized.get("local_path")
            or normalized.get("url")
            or normalized.get("file_name")
        )
    return (source_type, str(path)) if path else None


def multitrack_shared_task_images(tracks: list) -> list[dict]:
    """Collect unique explicitly shared task images in stable timeline order."""
    shared: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for track in tracks:
        if not isinstance(track, dict) or track.get("type") != "task":
            continue
        for segment in track.get("segments", []):
            if not isinstance(segment, dict):
                continue
            content = segment.get("content", {})
            images = content.get("images", []) if isinstance(content, dict) else []
            for image in images if isinstance(images, list) else []:
                identity = multitrack_media_identity(image)
                if not multitrack_is_shared_reference(image) or identity is None or identity in seen:
                    continue
                normalized = canonicalize_multitrack_slot_content(image)
                normalized["shared_reference"] = True
                normalized.pop("speaker_reference", None)
                shared.append(normalized)
                seen.add(identity)
    return shared[:MAX_MULTITRACK_TASK_IMAGES]


def multitrack_task_images_with_shared(
    images: object,
    shared_images: list[dict],
) -> list[dict]:
    """Prefix shared images, auto-matching same-path items, under the 9-image cap."""
    local_images = [image for image in images if isinstance(image, dict)] if isinstance(images, list) else []
    shared_identities = {
        identity
        for image in shared_images
        if (identity := multitrack_media_identity(image)) is not None
    }
    prefixed: list[dict] = []
    for shared_image in shared_images:
        identity = multitrack_media_identity(shared_image)
        matching = next(
            (image for image in local_images if multitrack_media_identity(image) == identity),
            shared_image,
        )
        normalized = canonicalize_multitrack_slot_content(matching)
        normalized["shared_reference"] = True
        normalized.pop("speaker_reference", None)
        prefixed.append(normalized)
    local = [
        canonicalize_multitrack_slot_content(image)
        for image in local_images
        if multitrack_media_identity(image) not in shared_identities
    ]
    return (prefixed + local)[:MAX_MULTITRACK_TASK_IMAGES]


def multitrack_slot_media_types(data: dict) -> set[str]:
    """Return media types whose track descriptors reference an input slot."""
    required: set[str] = set()
    tracks = data.get("tracks", [])
    if not isinstance(tracks, list):
        return required

    for track in tracks:
        if not isinstance(track, dict):
            continue
        track_type = str(track.get("type", ""))
        segments = track.get("segments", [])
        if not isinstance(segments, list):
            continue
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            content = segment.get("content", {})
            if not isinstance(content, dict):
                continue
            if track_type == "task":
                images = content.get("images", [])
                if isinstance(images, list) and any(
                    isinstance(image, dict) and multitrack_slot_name(image) is not None
                    for image in images
                ):
                    required.add("image")
            elif (
                track_type in {"audio", "video"}
                and content.get("media_type") == track_type
                and multitrack_slot_name(content) is not None
            ):
                required.add(track_type)
    return required


def multitrack_segments_in_window(
    track: dict,
    start_frame: int,
    end_frame: int,
) -> list[dict]:
    """Clip media segments to a window and shift them to window-local frames."""
    track_type = track.get("type")
    segments = track.get("segments", [])
    if not isinstance(segments, list) or end_frame <= start_frame:
        return []

    clipped: list[dict] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        content = segment.get("content", {})
        if not isinstance(content, dict) or content.get("media_type") != track_type:
            continue
        try:
            segment_start = int(segment.get("start_frame", 0))
            segment_end = int(segment.get("end_frame", segment_start))
            origin_start = int(segment.get("origin_start_frame", segment_start))
        except (TypeError, ValueError, OverflowError):
            continue

        overlap_start = max(start_frame, segment_start)
        overlap_end = min(end_frame, segment_end)
        if overlap_end <= overlap_start:
            continue

        local_segment = dict(segment)
        local_segment["start_frame"] = overlap_start - start_frame
        local_segment["end_frame"] = overlap_end - start_frame
        local_segment["origin_start_frame"] = origin_start - start_frame
        local_segment["content"] = dict(content)
        clipped.append(local_segment)
    return clipped
