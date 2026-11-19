#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop one HMI annotation candidate")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--label", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--event-index", type=int, default=0)
    parser.add_argument("--padding-ratio", type=float, default=0.2)
    args = parser.parse_args()

    label = json.loads(args.label.read_text(encoding="utf-8"))
    events = label.get("hek_event_annotation", {}).get("matched_events", [])
    if not 0 <= args.event_index < len(events):
        raise SystemExit(f"event index {args.event_index} is outside 0..{len(events) - 1}")

    event = events[args.event_index]
    polygon = event.get("bbox_polygon_pixel")
    if not isinstance(polygon, list) or len(polygon) < 3:
        raise SystemExit("selected event has no usable bbox_polygon_pixel")

    xs = [float(point[0]) for point in polygon]
    ys = [float(point[1]) for point in polygon]
    with Image.open(args.image) as image:
        width, height = image.size
        box_width = max(xs) - min(xs)
        box_height = max(ys) - min(ys)
        pad_x = max(16, round(box_width * args.padding_ratio))
        pad_y = max(16, round(box_height * args.padding_ratio))
        crop_box = (
            max(0, int(min(xs)) - pad_x),
            max(0, int(min(ys)) - pad_y),
            min(width, int(max(xs) + 1) + pad_x),
            min(height, int(max(ys) + 1) + pad_y),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        image.crop(crop_box).save(args.output, format="PNG")

    manifest = {
        "source_image": str(args.image.resolve()),
        "source_label": str(args.label.resolve()),
        "event_index": args.event_index,
        "event_id": event.get("SOL_standard") or event.get("solar_object_locator"),
        "event_type": event.get("event_type") or event.get("eventtype"),
        "polygon_pixel": polygon,
        "crop_box_xyxy": crop_box,
        "padding_ratio": args.padding_ratio,
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
