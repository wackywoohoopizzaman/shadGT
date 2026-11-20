"""Verify the compositor captures written by rygt_openxr_stereo_test."""

import argparse
import json
from pathlib import Path
import struct


def pixels(bmp, viewport):
    x0, y0, width, height = viewport
    if bmp[:2] != b"BM":
        raise ValueError("Expected a BMP screenshot")
    offset, = struct.unpack_from("<I", bmp, 10)
    w, h = struct.unpack_from("<ii", bmp, 18)
    bpp, compression = struct.unpack_from("<HI", bmp, 28)
    if bpp != 24 or compression or not (0 <= x0 < x0 + width <= w and 0 <= y0 < y0 + height <= abs(h)):
        raise ValueError("Unexpected capture layout")
    stride = (3 * w + 3) & ~3
    green = red = 0
    for y in range(y0, y0 + height):
        row = abs(h) - 1 - y if h > 0 else y
        start = offset + row * stride + x0 * 3
        for b, g, r in struct.iter_unpack("BBB", bmp[start:start + width * 3]):
            green += g > 160 and r < 70 and b < 70
            red += r > 160 and g < 70 and b < 70
    return green, red


def verify(directory):
    def geometry(identifier):
        report = json.loads((directory / f"screenshot_status.{identifier}.json").read_text())
        return report["geometry"]

    for prefix in ("50", "80", "90"):
        valid = prefix * 16
        geo = geometry(valid)
        views = geo["projection_eyes"]
        if [v["eye"] for v in views] != ["left", "right"] or geo["surfaces"]:
            raise ValueError("Expected an independent stereo projection")
        bmp = (directory / f"screenshot.{valid}.bmp").read_bytes()
        left, right = [pixels(bmp, v["viewport"]) for v in views]
        if left[0] != 0 or right[0] < 20 or min(left[1], right[1]) < 100:
            raise ValueError(f"Foreground occlusion failed: left {left}, right {right}")
    movie_id = "a0" * 16
    movie = geometry(movie_id)
    if movie["projection_eyes"] or len(movie["surfaces"]) != 1:
        raise ValueError("Expected a single seated movie screen")
    surface = movie["surfaces"][0]
    w, h = surface["size"]
    if surface["visibility"] != 0 or abs(w / h - 16 / 9) > 0.001:
        raise ValueError("Movie screen must cover both eyes at 16:9")
    bitmap = (directory / f"screenshot.{movie_id}.bmp").read_bytes()
    for eye in movie["eyes"]:
        green, red = pixels(bitmap, eye["viewport"])
        if green or red < 100:
            raise ValueError(f"Wrong movie image in {eye['eye']} eye")
    for prefix in ("60", "70"):
        fallback = geometry(prefix * 16)
        if fallback["projection_eyes"] or len(fallback["surfaces"]) != 1:
            raise ValueError(f"Expected exactly one quad for rejected pair {prefix}")
    print(f"PASS: green post pixels (left, right) = ({left[0]}, {right[0]}), separate, packed and native game eyes passed, the movie appeared in both eyes, both rejected pairs used a quad")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("simulator_capture_directory", type=Path)
    verify(parser.parse_args().simulator_capture_directory)
