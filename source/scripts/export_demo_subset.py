#!/usr/bin/env python3
"""Export a small, reviewable six-family/eight-class demonstration package.

The exporter consumes a private data root and an aggregated evaluation CSV.  It
never changes the input data and it writes the generated package to an explicit
output directory.  By default only small metadata and thumbnails are copied;
the full dataset, model weights and per-sample private ledgers stay outside the
repository.

The metric rows are the all-30-package P1--P4 test result.  They are deliberately
labelled as P0-relative technical validation in the generated report: a roadshow
package without an independent adjudication set cannot be presented as an
official roadshow F1.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any


FAMILIES = (
    "AIA_AR_sample",
    "HMI_AR_sample",
    "CME_C2_sample",
    "EPB_seg_sample",
    "DMSP_SSUSI",
    "ULF_sample",
)
CLASSES = (
    "L0-01",
    "L2-G",
    "L2-P",
    "L2-T",
    "L2-F",
    "L3-06",
    "L3-07",
    "L3-MIS",
)
CLASS_NAMES = {
    "L0-01": "数据文件缺失",
    "L2-G": "边界与位置偏差",
    "L2-P": "类别标注错误",
    "L2-T": "起止时刻/时间声明错误",
    "L2-F": "元数据尺寸矛盾",
    "L3-06": "漏标",
    "L3-07": "虚标",
    "L3-MIS": "覆盖不足或覆盖过度",
}
EVIDENCE_FORM = {
    "L0-01": "文件路径与哈希清单",
    "L2-G": "图像像素区域（多边形/框）或时间区间",
    "L2-P": "标注记录 JSON Pointer 与类别对照",
    "L2-T": "起止时间字段或时间区间",
    "L2-F": "元数据字段与实际尺寸对照",
    "L3-06": "图像区域、标注记录或时间区间候选",
    "L3-07": "图像区域、标注记录或时间区间候选",
    "L3-MIS": "图像区域/掩膜或时间区间候选",
}
CN_FAMILY = {
    "AIA_AR_sample": "AIA（日冕活动区）",
    "HMI_AR_sample": "HMI（磁场活动区）",
    "CME_C2_sample": "CME（日冕物质抛射）",
    "EPB_seg_sample": "EPB（等离子体泡分割）",
    "DMSP_SSUSI": "DMSP（极光/电离层图像）",
    "ULF_sample": "ULF（超低频时间序列）",
}


def _num(row: dict[str, str], key: str) -> int:
    try:
        return int(float(row.get(key, 0) or 0))
    except (TypeError, ValueError):
        return 0


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _label_data_pair(package: Path, label: Path) -> Path | None:
    name = str(_read_json(label).get("label_header", {}).get("fileName", ""))
    candidates = []
    if name:
        candidates += [package / "data" / name, package / "dataset_root" / "data" / name]
        candidates += list(package.rglob(name))
    stem = label.stem
    candidates += list(package.rglob(stem + ".png"))
    candidates += list(package.rglob(stem + ".csv"))
    for candidate in candidates:
        if candidate.is_file() and candidate.suffix.lower() in {".png", ".jpg", ".jpeg", ".csv"}:
            return candidate
    return None


def _polygon_points(obj: dict[str, Any]) -> list[list[float]] | None:
    if isinstance(obj.get("points"), list) and len(obj["points"]) >= 3:
        try:
            pts = [[float(p[0]), float(p[1])] for p in obj["points"]]
            return pts if all(math.isfinite(v) for p in pts for v in p) else None
        except (TypeError, ValueError, IndexError):
            return None
    raw = obj.get("bbox_polygon_pixel")
    if isinstance(raw, list) and len(raw) >= 3:
        try:
            pts = [[float(p[0]), float(p[1])] for p in raw]
            return pts if all(math.isfinite(v) for p in pts for v in p) else None
        except (TypeError, ValueError, IndexError):
            return None
    return None


def _collect_geometry(label_obj: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect native geometry without reducing polygons to rectangles."""
    found: list[dict[str, Any]] = []
    science = label_obj.get("science_annotation", {})
    mask = science.get("mask", {}) if isinstance(science, dict) else {}
    for i, shape in enumerate(mask.get("shapes", []) if isinstance(mask, dict) else []):
        if not isinstance(shape, dict):
            continue
        pts = _polygon_points(shape)
        if pts:
            found.append({"kind": "polygon", "label": shape.get("label"), "points": pts, "source": f"/science_annotation/mask/shapes/{i}"})
    events = label_obj.get("hek_event_annotation", {}).get("matched_events", [])
    if isinstance(events, list):
        for i, event in enumerate(events):
            if not isinstance(event, dict):
                continue
            pts = _polygon_points(event)
            if pts:
                found.append({"kind": "polygon", "label": event.get("event_type", event.get("concept")), "points": pts, "source": f"/hek_event_annotation/matched_events/{i}"})
    special = science.get("specialShapes", []) if isinstance(science, dict) else []
    if isinstance(special, list):
        for i, shape in enumerate(special):
            if isinstance(shape, dict):
                pts = _polygon_points(shape)
                if pts:
                    found.append({"kind": "polygon", "label": shape.get("label"), "points": pts, "source": f"/science_annotation/specialShapes/{i}"})
    return found


def _render_image(data: Path, label_obj: dict[str, Any], destination: Path) -> dict[str, Any]:
    """Render a thumbnail. Polygon labels remain polygons in the preview."""
    geometry = _collect_geometry(label_obj)
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        shutil.copy2(data, destination)
        return {"renderer": "copy-only", "geometry_count": len(geometry), "geometry": geometry}
    with Image.open(data) as source:
        image = source.convert("RGB")
        image.thumbnail((1000, 1000))
        scale_x = image.width / source.width
        scale_y = image.height / source.height
        draw = ImageDraw.Draw(image, "RGBA")
        for item in geometry:
            pts = [(round(x * scale_x), round(y * scale_y)) for x, y in item["points"]]
            # A polygon is drawn with its original vertices.  We do not replace
            # EPB masks with bounding rectangles; the bbox is only metadata.
            if len(pts) >= 3:
                draw.polygon(pts, fill=(255, 193, 7, 42), outline=(255, 100, 0, 255))
                draw.line(pts + [pts[0]], fill=(255, 100, 0, 255), width=max(2, image.width // 500), joint="curve")
        image.save(destination, quality=90)
    return {"renderer": "Pillow-native-polygon", "geometry_count": len(geometry), "geometry": geometry}


def _render_csv(data: Path, destination: Path) -> dict[str, Any]:
    rows = list(csv.reader(data.open(encoding="utf-8-sig", errors="replace")))
    header = rows[0] if rows else []
    numeric_cols = []
    for idx, name in enumerate(header):
        vals = []
        for row in rows[1: min(len(rows), 1200)]:
            if idx < len(row):
                try:
                    vals.append(float(row[idx]))
                except ValueError:
                    pass
        if len(vals) >= 3:
            numeric_cols.append((idx, name, vals))
    width, height = 1000, 420
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    colors = ["#2563eb", "#e11d48", "#059669", "#9333ea"]
    for j, (idx, name, vals) in enumerate(numeric_cols[:4]):
        lo, hi = min(vals), max(vals)
        span = hi - lo or 1.0
        pts = []
        for k, val in enumerate(vals):
            x = 30 + 930 * k / max(1, len(vals) - 1)
            y = 360 - 300 * (val - lo) / span
            pts.append(f"{x:.1f},{y:.1f}")
        svg.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{colors[j]}" stroke-width="1.5"/>')
        svg.append(f'<text x="40" y="{30 + j * 20}" font-size="14" fill="{colors[j]}">{html.escape(name)}</text>')
    svg.append('<line x1="30" y1="360" x2="960" y2="360" stroke="#334155"/>')
    svg.append("</svg>")
    destination.write_text("\n".join(svg), encoding="utf-8")
    return {"renderer": "standard-library-svg", "columns": [x[1] for x in numeric_cols[:4]], "rows": len(rows) - 1}


def _aggregate(metrics_path: Path, variant: str) -> dict[str, dict[str, dict[str, Any]]]:
    totals: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    with metrics_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("variant") != variant or row.get("family") not in FAMILIES or row.get("tier") == "P0" or row.get("code") not in CLASSES:
                continue
            cell = totals[row["family"]][row["code"]]
            for key in ("records", "known", "unknown", "positives", "TP", "FP", "FN", "TN"):
                cell[key] += _num(row, key)
    result: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for family in FAMILIES:
        for code in CLASSES:
            cell = totals[family][code]
            tp, fp, fn = cell["TP"], cell["FP"], cell["FN"]
            p, r = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
            f1 = _ratio(2 * tp, 2 * tp + fp + fn)
            result[family][code] = dict(cell, code=code, precision=p, recall=r, F1=f1,
                predicted_samples=tp + fp, missed_samples=fn,
                evidence_form=EVIDENCE_FORM[code], class_name=CLASS_NAMES[code])
    return result


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.2f}%"


def _make_case(package: Path, family: str, media_dir: Path, copy_media: bool) -> dict[str, Any] | None:
    labels = sorted(package.rglob("*.json"))
    data_by_name: dict[str, Path] = {}
    for data_path in package.rglob("*"):
        if data_path.is_file() and data_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".csv"}:
            data_by_name.setdefault(data_path.name, data_path)
    candidates: list[tuple[Path, Path, dict[str, Any], list[dict[str, Any]]]] = []
    for label in labels:
        try:
            obj = _read_json(label)
        except (OSError, json.JSONDecodeError):
            continue
        declared_name = str(obj.get("label_header", {}).get("fileName", ""))
        data = data_by_name.get(declared_name) if declared_name else None
        if data is None:
            # Keep the fallback for unusual nested packages, but avoid a full
            # recursive search for every ordinary label.
            data = _label_data_pair(package, label)
        if data is None:
            continue
        geometry = _collect_geometry(obj)
        candidates.append((label, data, obj, geometry))
    # Prefer a sample with native geometry so the reviewer can see an actual
    # annotation. If a family has no geometry (for example a CME parameter
    # record or an empty DMSP frame), keep the first valid data/label pair.
    if candidates:
        label, data, obj, geometry = max(candidates, key=lambda item: (len(item[3]), str(item[0])))
        record = {"family": family, "package": package.name, "label_file": str(label.relative_to(package)),
                  "data_file": str(data.relative_to(package)), "data_sha256": _sha(data),
                  "label_sha256": _sha(label), "media_included": False}
        if copy_media:
            media_dir.mkdir(parents=True, exist_ok=True)
            base = media_dir / f"{family}__{data.stem}"
            if data.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                dest = base.with_suffix(".jpg")
                render = _render_image(data, obj, dest)
            else:
                dest = base.with_suffix(".svg")
                render = _render_csv(data, dest)
            record.update(media_file=str(dest.relative_to(media_dir.parent)), media_included=True, **render)
        else:
            record["geometry"] = geometry
        return record
    return None


def _write_html(output: Path, aggregates: dict[str, dict[str, dict[str, Any]]], cases: list[dict[str, Any]], variant: str) -> None:
    case_by_family = {x["family"]: x for x in cases}
    rows = []
    for family in FAMILIES:
        rows.append(f'<section><h2>{html.escape(CN_FAMILY[family])} <small>{html.escape(family)}</small></h2>')
        case = case_by_family.get(family)
        if case and case.get("media_included"):
            media = case["media_file"]
            if media.endswith(".svg"):
                rows.append(f'<div class="case"><img src="{html.escape(media)}" alt="{html.escape(family)}时间序列"/><p>展示样本：{html.escape(case["package"])} / {html.escape(case["data_file"])}；原始 JSON 保留在服务器，不复制到仓库。</p></div>')
            else:
                rows.append(f'<div class="case"><img src="{html.escape(media)}" alt="{html.escape(family)}标注预览"/><p>展示样本：{html.escape(case["package"])} / {html.escape(case["data_file"])}；橙色轮廓按原始多边形顶点绘制。</p></div>')
        rows.append('<table><thead><tr><th>代码</th><th>缺陷</th><th>预测样本</th><th>参考阳性</th><th>精确率</th><th>召回率</th><th>F1</th><th>输出证据形式</th></tr></thead><tbody>')
        for code in CLASSES:
            item = aggregates[family][code]
            rows.append("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in [code, item["class_name"], item["predicted_samples"], item["positives"], _percent(item["precision"]), _percent(item["recall"]), _percent(item["F1"]), item["evidence_form"]]) + "</tr>")
        rows.append('</tbody></table></section>')
    scope = "全30包 P1–P4 严格配对测试（P0 相对技术监督）"
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>SpaceQA 六类八缺陷展示子集</title>
<style>body{{font:15px/1.6 Arial,"Microsoft YaHei",sans-serif;background:#f4f7fb;color:#1f2937;margin:0;padding:24px}}main{{max-width:1500px;margin:auto;background:#fff;padding:28px;border-radius:14px}}h1{{margin:0 0 4px}}h2{{margin-top:34px;color:#174a7e}}small{{font-weight:normal;color:#64748b}}table{{border-collapse:collapse;width:100%;font-size:13px;margin:14px 0 24px}}th,td{{border-bottom:1px solid #dbe4ee;padding:8px;text-align:left;vertical-align:top}}th{{background:#edf4fb}}tr:hover{{background:#f8fbff}}.case{{display:flex;gap:18px;align-items:flex-start;background:#f8fafc;padding:12px;border-radius:10px}}.case img{{max-width:600px;max-height:440px;border:1px solid #d9e2ec}}.note{{background:#fff7ed;border-left:4px solid #f59e0b;padding:10px 14px}}</style><main>
<h1>SpaceQA · 六类数据 × 八类缺陷展示子集</h1><p>模型口径：{html.escape(variant)}；统计范围：{scope}。</p>
<div class="note">表格是全30包的分类验证统计，用于展示“八类预报如何输出”。没有独立路演真值时，不能把这些 P/R/F1 写成路演官方成绩；展示样本只用于说明数据、标注和定位证据的形式。EPB 使用原始 polygon 顶点绘制，不降级为矩形框。</div>
{''.join(rows)}
</main></html>'''
    (output / "index.html").write_text(page, encoding="utf-8")


def export(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output exists and is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    aggregates = _aggregate(args.metrics, args.variant)
    cases = []
    for family in FAMILIES:
        packages = sorted(p for p in args.data_root.iterdir() if p.is_dir() and p.name.startswith(family))
        if not packages:
            continue
        case = _make_case(packages[0], family, output / "media", args.copy_media)
        if case:
            cases.append(case)
    all_rows = []
    for family in FAMILIES:
        for code in CLASSES:
            all_rows.append(dict(family=family, **aggregates[family][code]))
    (output / "metrics_by_family.json").write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "metrics_by_family.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["family", "code", "class_name", "records", "known", "unknown", "positives", "predicted_samples", "missed_samples", "TP", "FP", "FN", "TN", "precision", "recall", "F1", "evidence_form"])
        writer.writeheader()
        writer.writerows(all_rows)
    (output / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {"status": "complete", "families": list(FAMILIES), "classes": [{"code": c, "name": CLASS_NAMES[c], "evidence_form": EVIDENCE_FORM[c]} for c in CLASSES], "variant": args.variant, "scope": "all30 P1-P4 strict-mask technical validation", "media_included": args.copy_media, "cases": len(cases)}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    readme = f"""# 六类数据 × 八类缺陷展示子集\n\n本包用于演示 SpaceQA 的输出形式。包含六类数据的八类缺陷统计、证据形式说明和每类一个小样本预览。\n\n- 统计模型：`{args.variant}`。\n- 统计范围：全 30 包 P1–P4 严格配对测试，P0 相对技术监督。\n- `precision/recall/F1` 是内部技术验证指标，不是无独立真值的路演官方成绩。\n- `predicted_samples` 是模型预测为该类的问题样本数；`missed_samples` 是相对参考标签漏掉的样本数。\n- EPB 原始标签的 `science_annotation.mask.shapes[*].shape_type` 为 `polygon`，预览按顶点绘制多边形，绝不把多边形替换成矩形。\n- `cases.json` 只记录展示样本的哈希和相对路径；完整数据、权重和逐样本运行台账仍留在服务器。\n\n双击 `index.html` 查看交互式展示；`metrics_by_family.csv` 可直接导入表格。\n"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    _write_html(output, aggregates, cases, args.variant)
    if args.zip:
        archive = args.zip.resolve()
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for p in output.rglob("*"):
                if p.is_file():
                    z.write(p, p.relative_to(output.parent))
    return {"output": str(output), "families": len(FAMILIES), "classes": len(CLASSES), "cases": len(cases), "media": args.copy_media, "zip": str(args.zip.resolve()) if args.zip else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True, type=Path, help="private root containing the 30 package directories")
    ap.add_argument("--metrics", required=True, type=Path, help="all30_classes_test.csv")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--variant", default="selected_eight")
    ap.add_argument("--copy-media", action="store_true", help="copy one small preview per family")
    ap.add_argument("--zip", type=Path, help="optional zip output")
    args = ap.parse_args()
    print(json.dumps(export(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
