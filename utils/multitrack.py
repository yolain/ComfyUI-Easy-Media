from __future__ import annotations


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
                    isinstance(image, dict) and image.get("source_type") == "slot"
                    for image in images
                ):
                    required.add("image")
            elif (
                track_type in {"audio", "video"}
                and content.get("media_type") == track_type
                and content.get("source_type") == "slot"
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
