"""Image-only, grouped, weakly supervised localization pilot.

Original P0 annotation regions are weak targets, never scientific ground truth.
The network sees RGB pixels only. Known family selects one output channel. Current
annotations are read after prediction, only to produce review candidates.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path

FAMILIES = ("AIA_AR_sample", "HMI_AR_sample", "EPB_seg_sample", "DMSP_SSUSI")
NA_FAMILIES = ("CME_C2_sample", "ULF_sample")
ROOT = Path("/staff/tangjiarui/OPC")
SEED = 20261103
VERSION = "image-only-localizer-pilot-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def emit(stage, **kwargs):
    print(json.dumps(dict(stage=stage, **kwargs)), flush=True)


def annotation_module(root):
    path = root / "phase2_visual_audit_xzj/src/annotation_diff.py"
    spec = importlib.util.spec_from_file_location("localizer_annotation", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


def load_rows(root, split, p0_only=True):
    """Read existing assignments, never generate a replacement random split."""
    source = root / "opc-agent/runs/all-family-training-20261031" / (split + ".jsonl")
    rows = []
    keys = ("id", "family", "tier", "base_sample_id", "training_group_id",
            "split_group_id", "data_sha256", "label_sha256", "data_path", "label_path",
            "input_missing", "split")
    with source.open() as f:
        for line in f:
            r = json.loads(line)
            if r["family"] not in FAMILIES or (p0_only and r["tier"] != "P0"):
                continue
            if r.get("split") != split or not r.get("training_group_id"):
                raise ValueError("Missing or conflicting existing grouped assignment")
            rows.append({k: r.get(k) for k in keys})
    return rows, source


def extract_geometry(objects, width, height):
    """Keep complete finite in-image regions; do not silently invent geometry."""
    geometry, errors = [], []
    for index, obj in enumerate(objects):
        g = obj.get("geometry", {})
        p = g.get("points")
        kind = g.get("kind")
        if not g.get("valid") or not p or kind not in ("polygon", "rectangle"):
            errors.append({"index": index, "reason": "unavailable_pixel_geometry"})
            continue
        try:
            points = [[float(a), float(b)] for a, b in p]
            if any(not math.isfinite(v) for q in points for v in q):
                raise ValueError("nonfinite")
            if any(a < 0 or b < 0 or a > width or b > height for a, b in points):
                errors.append({"index": index, "reason": "out_of_image_bounds"})
                continue
            if (kind == "rectangle" and len(points) != 2) or (kind == "polygon" and len(points) < 3):
                raise ValueError("point_count")
        except (TypeError, ValueError):
            errors.append({"index": index, "reason": "invalid_points"})
            continue
        geometry.append(dict(index=index, kind=kind, points=points, pointer=g.get("pointer")))
    return geometry, errors


def region_mask(geometry, width, height, size=256, crop=None):
    """Rasterize original coordinates directly into a full image or crop canvas."""
    import numpy as np
    from PIL import Image, ImageDraw
    x0, y0, x1, y1 = crop or (0, 0, width, height)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Empty crop")
    image = Image.new("L", (size, size))
    draw = ImageDraw.Draw(image)
    for g in geometry:
        p = [((a-x0)/(x1-x0)*size, (b-y0)/(y1-y0)*size) for a, b in g["points"]]
        if g["kind"] == "rectangle":
            draw.rectangle([min(q[0] for q in p), min(q[1] for q in p),
                            max(q[0] for q in p), max(q[1] for q in p)], fill=1)
        else:
            draw.polygon(p, fill=1)
    return np.asarray(image, dtype="uint8")


def inspect_row(row, ann):
    from PIL import Image
    if row.get("input_missing") or not row.get("data_path") or not Path(row["data_path"]).exists():
        return dict(row, usable=False, reason="missing_input")
    try:
        if sha(row["label_path"]) != row["label_sha256"]:
            return dict(row, usable=False, reason="label_hash_mismatch")
        with Image.open(row["data_path"]) as im:
            width, height = im.size
        label = json.loads(Path(row["label_path"]).read_text(encoding="utf-8-sig"))
        objects = ann.extract_annotation(label, row["family"])["objects"]
        geometry, errors = extract_geometry(objects, width, height)
        mask = region_mask(geometry, width, height)
        # Any unidentified positive object invalidates a complete segmentation target.
        usable = not errors and (not objects or mask.any())
        return dict(row, width=width, height=height, geometry=geometry,
                    object_count=len(objects), geometry_errors=errors,
                    foreground_fraction=float(mask.mean()), positive=bool(mask.any()),
                    usable=bool(usable), reason=None if usable else "incomplete_or_zero_pixel_geometry")
    except Exception as exc:
        return dict(row, usable=False, reason="decode_or_schema_error", error=type(exc).__name__ + ":" + str(exc))


def summarize(rows):
    result = {}
    for family in FAMILIES:
        rr = [r for r in rows if r["family"] == family]
        usable = [r for r in rr if r["usable"]]
        result[family] = dict(records=len(rr), usable=len(usable),
            positive=sum(bool(r.get("positive")) for r in usable),
            empty=sum(not r.get("positive") for r in usable),
            incomplete_geometry=sum(bool(r.get("geometry_errors")) for r in rr),
            excluded_reasons=dict(collections.Counter(r.get("reason") for r in rr if not r["usable"])),
            mean_foreground_fraction=sum(r["foreground_fraction"] for r in usable)/max(1, len(usable)))
    return result


def audit(root, out):
    ann, annpath = annotation_module(root)
    rows, source = load_rows(root, "train")
    inspected = [inspect_row(r, ann) for r in rows]
    report = dict(version=VERSION, seed=SEED, split="train", source=str(source),
        source_sha256=sha(source), annotation_parser=str(annpath), annotation_parser_sha256=sha(annpath),
        weak_supervision="P0 annotation regions, NOT official scientific truth", families=summarize(inspected),
        not_applicable={f: "No verified pixel-region supervision; existing rules retained" for f in NA_FAMILIES},
        image_input="Original RGB only. Never composite previews, annotation overlays, field vectors or paired P0 at inference.")
    write(out / "train_audit.json", report)
    write(out / "train_inventory.json", inspected)
    emit("audit_complete", families=report["families"])
    return inspected, report


def capped(rows, cap, salt):
    """Reproducible per-family cap, preserving both weak positive and empty targets."""
    selected = []
    for family in FAMILIES:
        rr = sorted([r for r in rows if r["family"] == family and r["usable"]], key=lambda r: digest([SEED, salt, r["id"]]))
        positive = [r for r in rr if r["positive"]]
        empty = [r for r in rr if not r["positive"]]
        # At least a quarter empty when available, otherwise use every available positive.
        e = min(len(empty), max(cap//4, cap-min(len(positive), cap)))
        take = empty[:e] + positive[:cap-e]
        selected.extend(sorted(take, key=lambda r: digest([SEED, r["id"]])))
    return selected


def roi_boxes(width, height):
    """Annotation-independent inference grid; boundaries are fixed by image dimensions."""
    edge = 1024 if max(width, height) >= 2048 else 512
    if max(width, height) <= edge:
        return []
    xs = list(range(0, max(width-edge, 0)+1, edge))
    ys = list(range(0, max(height-edge, 0)+1, edge))
    if xs[-1] != max(width-edge, 0): xs.append(max(width-edge, 0))
    if ys[-1] != max(height-edge, 0): ys.append(max(height-edge, 0))
    return [(x, y, min(x+edge, width), min(y+edge, height)) for y in ys for x in xs]


def prepare_train_cache(rows, out, size):
    """Training foreground crops may use TRAIN targets; evaluation grids never do."""
    import numpy as np
    from PIL import Image
    cache = out / "train_cache"
    cache.mkdir(exist_ok=True)
    samples = []
    for n, r in enumerate(rows):
        boxes = [(0, 0, r["width"], r["height"])]
        rois = roi_boxes(r["width"], r["height"])
        if rois:
            rng = random.Random(int(digest([SEED, r["id"]])[:16], 16))
            foreground = [b for b in rois if region_mask(r["geometry"], r["width"], r["height"], 32, b).any()]
            boxes.append(rng.choice(foreground if foreground and rng.random() < .75 else rois))
        with Image.open(r["data_path"]) as original:
            im = original.convert("RGB")
            for view, box in enumerate(boxes):
                token = digest([r["id"], view, box])
                ip, mp = cache/(token+".jpg"), cache/(token+".png")
                mask = region_mask(r["geometry"], r["width"], r["height"], size, box)
                im.crop(box).resize((size, size), Image.Resampling.BILINEAR).save(ip, quality=95)
                Image.fromarray(mask * 255).save(mp)
                samples.append(dict(id=r["id"], family=r["family"], group=r["training_group_id"],
                    image=str(ip), mask=str(mp), view="global" if view == 0 else "train_roi", crop=list(box),
                    foreground_fraction=float(np.mean(mask))))
        if (n+1) % 200 == 0: emit("train_cache", records=n+1, total=len(rows))
    write(out/"train_samples.json", samples)
    return samples


def torch_model():
    import torch
    from torch import nn
    from torch.nn import functional as F
    class Block(nn.Sequential):
        def __init__(self, a, b):
            super().__init__(nn.Conv2d(a, b, 3, padding=1), nn.GroupNorm(4, b), nn.SiLU(),
                             nn.Conv2d(b, b, 3, padding=1), nn.GroupNorm(4, b), nn.SiLU())
    class Localizer(nn.Module):
        def __init__(self):
            super().__init__()
            self.a=Block(3, 8); self.b=Block(8,16); self.c=Block(16,32); self.d=Block(32,64)
            self.u2=Block(96,32); self.u1=Block(48,16); self.u0=Block(24,8)
            self.head=nn.Conv2d(8, len(FAMILIES), 1)
        def forward(self, x):
            a=self.a(x); b=self.b(F.max_pool2d(a,2)); c=self.c(F.max_pool2d(b,2)); d=self.d(F.max_pool2d(c,2))
            def up(t, s): return F.interpolate(t, size=s.shape[-2:], mode="bilinear", align_corners=False)
            z=self.u2(torch.cat([up(d,c),c],1)); z=self.u1(torch.cat([up(z,b),b],1))
            return self.head(self.u0(torch.cat([up(z,a),a],1)))
    return Localizer()


def assert_disjoint(train, evaluation):
    for key in ("training_group_id", "base_sample_id", "data_sha256"):
        a = {r[key] for r in train if r.get(key)}
        b = {r[key] for r in evaluation if r.get(key)}
        if a & b: raise ValueError("Split leakage: " + key)


def train_models(samples, rows, out, args, started):
    import numpy as np
    import torch
    from PIL import Image
    from torch.nn import functional as F
    torch.set_num_threads(4)
    torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    torch.cuda.set_device(args.device)
    torch.cuda.set_per_process_memory_fraction(.09, args.device)
    torch.backends.cudnn.benchmark = False
    class Dataset(torch.utils.data.Dataset):
        def __len__(self): return len(samples)
        def __getitem__(self, i):
            r=samples[i]
            with Image.open(r["image"]) as im: x=np.asarray(im.convert("RGB"), dtype=np.float32).copy()/255
            with Image.open(r["mask"]) as im: y=np.asarray(im,dtype=np.float32).copy()/255
            return torch.from_numpy(x.transpose(2,0,1)), torch.from_numpy(y[None]), FAMILIES.index(r["family"])
    counts=collections.Counter(r["family"] for r in samples)
    weights=[1/counts[r["family"]] for r in samples]
    fractions={f:np.mean([r["foreground_fraction"] for r in samples if r["family"]==f]) for f in FAMILIES}
    pos_weights=torch.tensor([min(20.,max(1.,(1-fractions[f])/max(.001,fractions[f]))) for f in FAMILIES], device="cuda")
    records=[]
    # Both routes finish ALL family training before this function returns or evaluation can start.
    for route in ("weighted_bce_dice", "bce_ablation"):
        torch.manual_seed(SEED)
        model=torch_model().cuda(); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        generator=torch.Generator().manual_seed(SEED)
        sampler=torch.utils.data.WeightedRandomSampler(weights,len(samples),replacement=True,generator=generator)
        loader=torch.utils.data.DataLoader(Dataset(),batch_size=args.batch,sampler=sampler,num_workers=2,pin_memory=True)
        history=[]; torch.cuda.reset_peak_memory_stats(); begin=time.monotonic()
        for epoch in range(args.epochs):
            model.train(); losses=[]; family_seen=collections.Counter()
            for x,y,family in loader:
                if time.monotonic()-started>args.budget_seconds: raise TimeoutError("Controlled pilot runtime budget reached")
                x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);family=family.cuda(non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    all_logits=model(x); logits=all_logits[torch.arange(len(x),device=x.device),family][:,None]
                    if route=="weighted_bce_dice":
                        bce=F.binary_cross_entropy_with_logits(logits,y,reduction="none")
                        bce=(bce*torch.where(y>.5,pos_weights[family,None,None,None],1.)).mean()
                        p=logits.sigmoid(); dice=1-((2*(p*y).sum((1,2,3))+1)/(p.sum((1,2,3))+y.sum((1,2,3))+1)).mean()
                        loss=bce+dice
                    else: loss=F.binary_cross_entropy_with_logits(logits,y)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
                losses.append(float(loss));family_seen.update(FAMILIES[i] for i in family.cpu().tolist())
            if set(family_seen)!=set(FAMILIES): raise RuntimeError("Training epoch did not cover all supported families")
            item=dict(epoch=epoch+1,loss=float(np.mean(losses)),family_seen=dict(family_seen),seconds=time.monotonic()-begin)
            history.append(item);emit("training",route=route,**item)
            write(out/"state.json",dict(status="training",route=route,history=history))
        path=out/(route+".pt")
        torch.save(dict(version=VERSION,state_dict=model.cpu().state_dict(),families=FAMILIES,seed=SEED,
                        size=args.size,train_groups=sorted({r["training_group_id"] for r in rows})),path)
        records.append(dict(route=route,checkpoint=str(path),sha256=sha(path),bytes=path.stat().st_size,
            parameters=sum(p.numel() for p in model.parameters()),history=history,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            seconds=time.monotonic()-begin,pos_weights=pos_weights.cpu().tolist() if route=="weighted_bce_dice" else None))
        del model,opt,loader;torch.cuda.empty_cache()
    marker=dict(version=VERSION,all_supported_families_trained=True,routes=records,
        train_group_ids=sorted({r["training_group_id"] for r in rows}),seed=SEED,
        gpu_shared="GPU0 shares existing Qwen/vLLM and small replay task; GPU1 never used",
        excluded_family_tasks={f:"NA: rules, not pixel segmentation" for f in NA_FAMILIES})
    write(out/"training_complete.json",marker)
    return records


def image_prediction(model, row, size, batch, use_roi=True, instance_size=1024):
    """No label/geometry reads. Output probability is in a fixed 256px coordinate frame."""
    import numpy as np
    import torch
    from PIL import Image
    from torch.nn import functional as F
    with Image.open(row["data_path"]) as original:
        im=original.convert("RGB");width,height=im.size
        boxes=[(0,0,width,height)]+(roi_boxes(width,height) if use_roi else [])
        arrays=[np.asarray(im.crop(box).resize((size,size),Image.Resampling.BILINEAR),dtype=np.float32).transpose(2,0,1)/255 for box in boxes]
    predictions=[]
    for start in range(0,len(arrays),batch):
        x=torch.from_numpy(np.stack(arrays[start:start+batch])).cuda()
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            p=model(x)[:,FAMILIES.index(row["family"])].sigmoid().float().cpu().numpy()
        predictions.extend(p)
    global_map=predictions[0]
    native_global=F.interpolate(torch.from_numpy(global_map)[None,None],size=(instance_size,instance_size),mode="bilinear",align_corners=False)[0,0].numpy()
    local_sum=np.zeros_like(native_global);count=np.zeros_like(native_global)
    for box,p in zip(boxes[1:],predictions[1:]):
        x0,y0,x1,y1=box
        x0=int(round(x0/width*instance_size));x1=int(round(x1/width*instance_size));y0=int(round(y0/height*instance_size));y1=int(round(y1/height*instance_size))
        t=torch.from_numpy(p)[None,None]
        resized=F.interpolate(t,size=(y1-y0,x1-x0),mode="bilinear",align_corners=False)[0,0].numpy()
        local_sum[y0:y1,x0:x1]+=resized;count[y0:y1,x0:x1]+=1
    native_fused=np.where(count>0,.5*native_global+.5*local_sum/np.maximum(count,1),native_global)
    fused=F.interpolate(torch.from_numpy(native_fused)[None,None],size=(size,size),mode="bilinear",align_corners=False)[0,0].numpy()
    return fused,global_map,len(boxes),native_fused


def mask_metrics(pred, truth):
    import numpy as np
    tp=int(np.logical_and(pred,truth).sum());fp=int(np.logical_and(pred,~truth).sum());fn=int(np.logical_and(~pred,truth).sum())
    return dict(tp=tp,fp=fp,fn=fn,iou=tp/(tp+fp+fn) if tp+fp+fn else 1.,
                dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
                truth_positive=bool(truth.any()),prediction_positive=bool(pred.any()))


def instances(pred, probability, min_pixels=8, max_components=64):
    import numpy as np
    from scipy.ndimage import label,find_objects
    lab,n=label(pred); found=[]
    for k, sl in enumerate(find_objects(lab),1):
        if sl is None:continue
        mask=lab==k;area=int(mask.sum())
        if area<min_pixels:continue
        yy,xx=np.where(mask)
        found.append(dict(component_id=k,area_pixels=area,bbox=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],
                          confidence=float(probability[mask].mean()),mask=mask))
    return sorted(found,key=lambda c:c["area_pixels"],reverse=True)[:max_components]


def match_instances(predicted, geometry, width, height, size=256, min_iou=.1):
    """Independent predicted components matched only AFTER the image forward pass."""
    import numpy as np
    targets=[region_mask([g],width,height,size).astype(bool) for g in geometry]
    from scipy.optimize import linear_sum_assignment
    weights=np.zeros((max(len(predicted),len(targets),1),)*2,dtype=float)
    ious={}
    for i,p in enumerate(predicted):
        for j,t in enumerate(targets):
            union=np.logical_or(p["mask"],t).sum();inter=np.logical_and(p["mask"],t).sum()
            score=float(inter/union) if union else 0.
            if score>=min_iou:
                weights[i,j]=max(len(predicted),len(targets))+1+score
                ious[i,j]=score
    usedp=set();usedt=set();matches=[]
    pp,tt=linear_sum_assignment(weights,maximize=True)
    for i,j in zip(pp,tt):
        if (i,j) not in ious:continue
        usedp.add(int(i));usedt.add(int(j));matches.append(dict(component=int(i),annotation_index=geometry[j]["index"],iou=ious[i,j]))
    return dict(matches=matches,
        possible_omission_components=[i for i in range(len(predicted)) if i not in usedp],
        unmatched_annotations_abstain=[geometry[j]["index"] for j in range(len(geometry)) if j not in usedt],
        boundary_review_matches=[m for m in matches if m["iou"]<.5],
        status="pending_review",scoreEligible=False,
        limitation="Connected components may merge adjacent objects; P0 weak labels and image predictions do not establish official defect truth.")


def native_reference(row, components, size, threshold, route):
    """Class-agnostic reference usable by the core matcher, without label leakage."""
    return dict(status="ok", independent_of_annotations=True, class_known=False,
        coverage_complete=False, confidence_calibrated=False, source="model", model_version=VERSION+":"+route,
        threshold=dict(pixel_probability=threshold, selection="P0 validation weak-mask IoU", min_component_pixels=max(8,8*(size//256)**2)),
        objects=[dict(id=str(c["component_id"]), label=None, class_known=False,
            sample_id=row["id"], confidence=c["confidence"], confidence_source="uncalibrated mean foreground probability",
            locator=dict(frame_id=row["data_sha256"],coordinate_system="pixel",
                bbox=[c["bbox"][0]/size*row["width"],c["bbox"][1]/size*row["height"],
                      c["bbox"][2]/size*row["width"],c["bbox"][3]/size*row["height"]])) for c in components])


def aggregate(items):
    positive=[v for v in items if v["truth_positive"]]
    tp=sum(v["tp"] for v in items);fp=sum(v["fp"] for v in items);fn=sum(v["fn"] for v in items)
    return dict(samples=len(items),positive_samples=len(positive),empty_samples=len(items)-len(positive),
        mean_iou=sum(v["iou"] for v in items)/max(1,len(items)),
        positive_mean_iou=sum(v["iou"] for v in positive)/len(positive) if positive else None,
        micro_iou=tp/(tp+fp+fn) if tp+fp+fn else None,
        micro_dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        empty_prediction_rate=sum(v["prediction_positive"] for v in items if not v["truth_positive"])/max(1,len(items)-len(positive)))


def evaluate(root,out,training,train_rows,args,started):
    import numpy as np
    import torch
    from PIL import Image
    if not (out/"training_complete.json").exists():raise RuntimeError("Training barrier not satisfied")
    ann,_=annotation_module(root);splits={};sources={}
    # Validation and test are first opened only after ALL route/family training finishes.
    for split in ("val","test"):
        rows,source=load_rows(root,split)
        assert_disjoint(train_rows,rows)
        splits[split]=capped([inspect_row(r,ann) for r in rows],args.eval_cap,split)
        sources[split]=dict(path=str(source),sha256=sha(source))
    assert_disjoint(splits["val"],splits["test"])
    results=[];instances_out=(out/"test_instances.jsonl").open("w",encoding="utf-8")
    for run in training:
        model=torch_model().cuda();model.load_state_dict(torch.load(run["checkpoint"],map_location="cpu",weights_only=False)["state_dict"]);model.eval()
        thresholds={};summaries={};cache={}
        for split in ("val","test"):
            metrics=collections.defaultdict(list);global_metrics=collections.defaultdict(list)
            for n,row in enumerate(splits[split]):
                if time.monotonic()-started>args.budget_seconds:raise TimeoutError("Controlled pilot runtime budget reached")
                probability,global_map,views,native_probability=image_prediction(model,row,args.size,args.batch,instance_size=args.instance_size)
                truth=region_mask(row["geometry"],row["width"],row["height"],args.size).astype(bool)
                cache[row["id"]]=(probability,global_map,truth)
                if split=="val":
                    for threshold in (.2,.35,.5,.65,.8):metrics[(row["family"],threshold)].append(mask_metrics(probability>=threshold,truth))
                else:
                    threshold=thresholds[row["family"]]
                    pred=probability>=threshold
                    metrics[row["family"]].append(mask_metrics(pred,truth));global_metrics[row["family"]].append(mask_metrics(global_map>=threshold,truth))
                    components=instances(native_probability>=threshold,native_probability,min_pixels=max(8,8*(args.instance_size//256)**2))
                    matched=match_instances(components,row["geometry"],row["width"],row["height"],args.instance_size)
                    output=dict(version=VERSION,route=run["route"],id=row["id"],family=row["family"],tier="P0",split="test",
                        data_path=row["data_path"],label_path=row["label_path"],data_sha256=row["data_sha256"],label_sha256=row["label_sha256"],
                        coordinate_frame=dict(width=args.instance_size,height=args.instance_size,original_width=row["width"],original_height=row["height"]),
                        threshold=threshold,views=views,instances=[{k:v for k,v in c.items() if k!="mask"} for c in components],
                        reference=native_reference(row,components,args.instance_size,threshold,run["route"]),**matched)
                    instances_out.write(json.dumps(output)+"\n")
                    if n<12:
                        preview=out/"previews"/run["route"];preview.mkdir(parents=True,exist_ok=True)
                        Image.fromarray((probability*255).astype("uint8")).save(preview/(digest(row["id"])+"_probability.png"))
                if (n+1)%64==0:emit("evaluation",route=run["route"],split=split,records=n+1,total=len(splits[split]))
            if split=="val":
                table={f:{str(t):aggregate(metrics[(f,t)]) for t in (.2,.35,.5,.65,.8)} for f in FAMILIES}
                for family in FAMILIES:
                    # Positive IoU prevents the many empty HMI labels selecting all-background.
                    thresholds[family]=max((.2,.35,.5,.65,.8),key=lambda t:(table[family][str(t)]["positive_mean_iou"] or 0.,-table[family][str(t)]["empty_prediction_rate"]))
                summaries[split]=table
                write(out/(run["route"]+"_frozen_thresholds.json"),dict(source="P0 validation weak masks only",thresholds=thresholds))
            else:summaries[split]={f:dict(global_plus_roi=aggregate(metrics[f]),global_only=aggregate(global_metrics[f])) for f in FAMILIES}
        # Limited independent localization on contaminated TEST-group observations.
        # These have no scientific truth. No P0 mask is transferred to changed images.
        tier_rows,tier_source=load_rows(root,"test",p0_only=False)
        assert_disjoint(train_rows,tier_rows)
        selected_bases=set()
        for family in FAMILIES:
            bases=sorted({r["base_sample_id"] for r in splits["test"] if r["family"]==family},key=lambda b:digest([SEED,b]))
            selected_bases.update(bases[:args.tier_cap])
        tier_counts=collections.Counter()
        with (out/(run["route"]+"_contaminated_instances.jsonl")).open("w",encoding="utf-8") as output_file:
            for row in tier_rows:
                if row["tier"]=="P0" or row["base_sample_id"] not in selected_bases:continue
                current=inspect_row(row,ann)
                if current.get("input_missing") or "width" not in current:continue
                if time.monotonic()-started>args.budget_seconds:raise TimeoutError("Controlled pilot runtime budget reached")
                _,_,views,probability=image_prediction(model,current,args.size,args.batch,instance_size=args.instance_size)
                threshold=thresholds[row["family"]];components=instances(probability>=threshold,probability,min_pixels=max(8,8*(args.instance_size//256)**2))
                matched=match_instances(components,current["geometry"],current["width"],current["height"],args.instance_size)
                output=dict(id=row["id"],family=row["family"],tier=row["tier"],split="test",route=run["route"],
                    data_path=row["data_path"],label_path=row["label_path"],data_sha256=row["data_sha256"],label_sha256=row["label_sha256"],
                    reference=native_reference(current,components,args.instance_size,threshold,run["route"]),
                    scientific_defect_metrics=None,geometry_errors=current.get("geometry_errors"),views=views,**matched)
                output_file.write(json.dumps(output)+"\n");tier_counts[row["family"]+":"+row["tier"]]+=1
        results.append(dict(route=run["route"],thresholds=thresholds,results=summaries,contaminated_prediction_counts=dict(tier_counts)))
        del model;torch.cuda.empty_cache()
    instances_out.close()
    report=dict(version=VERSION,seed=SEED,status="complete",sources=sources,training=training,routes=results,
        seconds=time.monotonic()-started,
        scientific_scope="P0 annotation-mask reproduction, not official localized defect F1/recall. No hidden/roadshow truth.",
        excluded_families={f:"NA; existing deterministic checks retained" for f in NA_FAMILIES},
        train_sample_cap_per_family=args.train_cap,eval_cap_per_family=args.eval_cap)
    write(out/"evaluation.json",report);write(out/"state.json",dict(status="complete",seconds=report["seconds"]))
    emit("complete",output=str(out),seconds=report["seconds"])
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("audit","run"));p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--out",type=Path,required=True);p.add_argument("--train-cap",type=int,default=384)
    p.add_argument("--eval-cap",type=int,default=64);p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--tier-cap",type=int,default=4,help="TEST-group bases per family for P1-P4 localization; scientific metrics unknown")
    p.add_argument("--batch",type=int,default=8);p.add_argument("--size",type=int,default=256)
    p.add_argument("--instance-size",type=int,default=1024,help="Retain ROI detail in independent localization, separate from 256px weak-mask metrics")
    p.add_argument("--device",type=int,default=0);p.add_argument("--budget-seconds",type=int,default=1800)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    try:
        if not 1<=args.epochs<=3:raise ValueError("Pilot requires 1–3 epochs")
        if args.size%8:raise ValueError("Image size must be divisible by 8")
        rows,report=audit(args.root,args.out)
        if args.mode=="audit":return
        rows=capped(rows,args.train_cap,"train")
        if set(r["family"] for r in rows)!=set(FAMILIES):raise ValueError("All four supported family training inputs are required")
        plan=dict(version=VERSION,seed=SEED,arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            selected_by_family=dict(collections.Counter(r["family"] for r in rows)),source_sha256=report["source_sha256"],
            gpu_budget="CUDA device0, <=9% of A10040GB allocator; shared GPU0; do not touch GPU1",source_code_sha256=sha(__file__))
        write(args.out/"plan.json",plan)
        samples=prepare_train_cache(rows,args.out,args.size)
        training=train_models(samples,rows,args.out,args,started)
        evaluate(args.root,args.out,training,rows,args,started)
    except Exception as exc:
        write(args.out/"state.json",dict(status="failed",error=type(exc).__name__+":"+str(exc),seconds=time.monotonic()-started))
        raise


if __name__=="__main__":main()
