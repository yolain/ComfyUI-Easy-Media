import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.media import file_entry


def test_file_entry_encodes_unicode_view_url_components(tmp_path):
    subfolder = tmp_path / "中文 子目录"
    subfolder.mkdir()
    filename = (
        "jimeng-2026-02-13-9387-基于经典“子弹时间”段落进行镜头语言级复刻，"
        "镜头角度、景别、节奏与原片保持一致，....mp4"
    )
    media_file = subfolder / filename
    media_file.write_bytes(b"video")

    entry = file_entry(
        str(media_file),
        f"中文 子目录/{filename}",
        "inputs",
    )

    assert entry["url"] == (
        "/view?filename=jimeng-2026-02-13-9387-%E5%9F%BA%E4%BA%8E%E7%BB%8F%E5%85%B8"
        "%E2%80%9C%E5%AD%90%E5%BC%B9%E6%97%B6%E9%97%B4%E2%80%9D%E6%AE%B5%E8%90%BD"
        "%E8%BF%9B%E8%A1%8C%E9%95%9C%E5%A4%B4%E8%AF%AD%E8%A8%80%E7%BA%A7%E5%A4%8D"
        "%E5%88%BB%EF%BC%8C%E9%95%9C%E5%A4%B4%E8%A7%92%E5%BA%A6%E3%80%81%E6%99%AF"
        "%E5%88%AB%E3%80%81%E8%8A%82%E5%A5%8F%E4%B8%8E%E5%8E%9F%E7%89%87%E4%BF%9D"
        "%E6%8C%81%E4%B8%80%E8%87%B4%EF%BC%8C....mp4"
        "&type=input&subfolder=%E4%B8%AD%E6%96%87%20%E5%AD%90%E7%9B%AE%E5%BD%95"
    )
