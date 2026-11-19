"""Prepare current-only multimodal review cases from a new raw sample."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path


def prepare_case(*, data_path: Path, label_path: Path, family: str,
                 output: Path, signals: dict | None=None) -> dict:
    # Pillow is optional for the base rule engine, required only for this entry.
    from PIL import Image,ImageDraw,ImageOps
    from . import refinement as r
    if output.exists():raise ValueError("Choose a new case output directory")
    if output.resolve().is_relative_to(label_path.parent.resolve()) or output.resolve().is_relative_to(data_path.parent.resolve()):
        raise ValueError("Write prepared cases outside source directories")
    annotation=json.loads(label_path.read_text(encoding="utf-8-sig"))
    label_sha=hashlib.sha256(label_path.read_bytes()).hexdigest()
    data_sha=hashlib.sha256(data_path.read_bytes()).hexdigest() if data_path.is_file() else None
    sid="CASE-"+r.digest([data_sha,label_sha,family])[:20]
    case=dict(case_id=sid,family=family,annotation=annotation,
        sample_path="current/data"+data_path.suffix,label_path="current/label.json",
        data_facts={"exists":data_path.is_file()},frame_id=data_sha,media={},objects=[])
    signals=signals or {}
    for key in ["fusion_probabilities","fine_probabilities","experts","structural_features","feature_semantics","domain_definitions","rule_evidence"]:
        if key in signals:case[key]=signals[key]
    media=output/"media";media.mkdir(parents=True,exist_ok=False)
    if data_path.is_file() and data_path.suffix.lower()!=".csv":
        with Image.open(data_path) as raw:
            raw.load();im=raw.convert("RGB");case["image_size"]=list(im.size)
        drawn=im.copy();draw=ImageDraw.Draw(drawn)
        def walk(v,p=""):
            if isinstance(v,dict):
                for key,value in v.items():
                    pointer=p+"/"+key.replace("~","~0").replace("/","~1")
                    if key in {"points","bbox_polygon_pixel"} and isinstance(value,list) and len(value)>2 and all(isinstance(x,list) and len(x)==2 and all(isinstance(a,(int,float)) and not isinstance(a,bool) and math.isfinite(a) for a in x) for x in value):
                        points=[tuple(x) for x in value];draw.line(points+[points[0]],fill="red",width=max(1,im.width//300))
                        case["objects"].append({"annotation_pointer":pointer,"bbox":[min(x[0] for x in value),min(x[1] for x in value),max(x[0] for x in value),max(x[1] for x in value)],"label":v.get("label",v.get("event_type"))})
                    else:walk(value,pointer)
            elif isinstance(v,list):
                for i,x in enumerate(v):walk(x,p+"/"+str(i))
        walk(annotation)
        for name,view in [("observation",im),("annotation",drawn)]:
            # Keep aspect ratio, no padding/cropping, explicit native image size.
            view=ImageOps.contain(view,(768,768));view.save(media/(sid+"-"+name+".jpg"),quality=92)
            case["media"][name]="media/"+sid+"-"+name+".jpg"
        case["image_instruction"]="First view=current observation; second=current input annotation, red outlines. No predictions are drawn. Previews preserve aspect ratio without padding/cropping. Report coordinates in original image_size pixels."
    elif data_path.is_file():
        with data_path.open(encoding="utf-8-sig") as stream:reader=csv.DictReader(stream);rows=list(reader);columns=reader.fieldnames
        case["data_facts"].update(csv_rows=len(rows),columns=columns)
        case["data_summary"]={"first_rows":rows[:2],"last_rows":rows[-2:]}
        # ULF t is the documented relative-seconds column. Never infer duration
        # from row count or metadata sampling_rate_hz.
        times=[float(row["t"]) for row in rows] if rows and "t" in rows[0] else []
        if times and all(math.isfinite(t) for t in times) and min(times)<max(times):
            case["time_axis"]={"id":"current_CSV_t_relative_seconds","start":min(times),"end":max(times)}
            case["data_facts"]["time_column"]="t"
            chart=Image.new("RGB",(768,384),"white");draw=ImageDraw.Draw(chart);draw.rectangle((50,20,748,350),outline="black")
            draw.text((55,355),f"t (relative seconds): {min(times):g} .. {max(times):g}",fill="black")
            for col,color in [("H","blue"),("Z","green"),("D","red")]:
                if col not in rows[0]:continue
                values=[float(row[col]) for row in rows]
                if not all(math.isfinite(v) for v in values):continue
                lo,hi=min(values),max(values);step=max(1,len(rows)//1400)
                pts=[(50+698*(times[i]-min(times))/(max(times)-min(times)),350-330*(values[i]-lo)/max(1e-12,hi-lo)) for i in range(0,len(rows),step)]
                if len(pts)>1:draw.line(pts,fill=color,width=1)
            chart.save(media/(sid+"-observation.jpg"),quality=92)
            case["media"]["observation"]="media/"+sid+"-observation.jpg"
            case["image_instruction"]="Current CSV plot only. H/Z/D independently min-max scaled to show shapes, not physical amplitudes. Read numerical summaries/CSV for units. Annotation ISO timestamps cannot be converted to relative t without an explicit verified time origin."
    case["rule_evidence"]=list(case.get("rule_evidence",[]))+r.measure_dimensions(case)
    r.validate_case(case)
    (output/(sid+".json")).write_text(json.dumps(case,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (output/"provenance.json").write_text(json.dumps(dict(case_id=sid,data_path=str(data_path.resolve()),label_path=str(label_path.resolve()),data_sha256=data_sha,label_sha256=label_sha),ensure_ascii=False,indent=2),encoding="utf-8")
    return {"case_id":sid,"case":str(output/(sid+".json")),"images":[str(output/x) for x in case["media"].values()]}
