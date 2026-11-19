"""Create reviewable timer-only copies; canonical files are never changed."""
import argparse
import hashlib
import json
from pathlib import Path


def generate(root, remote):
    contract = []

    def patch(text, old, new, label):
        count = text.count(old)
        assert count == 1, (label, count, old[:120])
        contract.append(dict(label=label, matches=count))
        return text.replace(old, new)

    for source, output, phase in [
        ('source_roadshow.py', 'classification_bench.py', 'classification'),
        ('source_ground_all_refresh.py', 'grounding_bench.py', 'grounding'),
        ('source_finish_refresh.py', 'finalize_bench.py', 'finalize'),
    ]:
        original = (root / source).read_text(encoding='utf-8-sig')
        text = patch(original, 'from __future__ import annotations',
                     'from __future__ import annotations\nimport bench_metrics as bm\nbm.script_enter(' + repr(phase) + ')', phase + ': entry')
        if phase == 'classification':
            text = patch(text, "    for folder in sorted(folders.iterdir()):", "    bm.mark('shared_imports_policy_and_fusion_model_load')\n    for folder in sorted(folders.iterdir()):", 'classification: shared init')
            text = patch(text, "        records=list(e.rows(folder/'inventory.jsonl'));name=folder.name", "        bm.package_begin(folder.name)\n        records=list(e.rows(folder/'inventory.jsonl'));name=folder.name", 'classification: package begin')
            text = patch(text, "        with ThreadPoolExecutor(max_workers=8) as pool:ledgers=list(pool.map(fresh,records))", "        bm.mark('inventory_and_prior_metadata')\n        with ThreadPoolExecutor(max_workers=8) as pool:ledgers=list(pool.map(fresh,records))\n        bm.mark('fresh_rules_and_current_input_hashes')", 'classification: rules')
            text = patch(text, "        probabilities={};cache_hashes={}", "        bm.mark('assemble_rule_flags')\n        probabilities={};cache_hashes={}", 'classification: flags')
            text = patch(text, "        p=np.array([[probabilities[n]", "        bm.mark('verify_and_load_21_expert_caches')\n        p=np.array([[probabilities[n]", 'classification: cache')
            text = patch(text, "        target=out/'packages'/name;target.mkdir(parents=True)", "        bm.mark('feature_construction_and_eight_head_fusion')\n        target=out/'packages'/name;target.mkdir(parents=True)", 'classification: fusion')
            text = patch(text, "    assert len(packages)==12", "        bm.mark('write_classification_and_scorecard')\n        bm.package_end(samples=len(records),family=records[0]['family'])\n    assert len(packages)==12", 'classification: package end')
            text = patch(text, "    ranked=q.rank(cards);", "    bm.mark('shared_loop_finalization')\n    ranked=q.rank(cards);", 'classification: ranking start')
            text = patch(text, "\n\n\nif __name__=='__main__':", "\n    bm.mark('shared_global_ranking_and_manifest')\n    bm.done()\n\n\nif __name__=='__main__':", 'classification: done')
        elif phase == 'grounding':
            text = patch(text, "    for folder in sorted((args.roadshow/'packages').iterdir()):", "    bm.mark('shared_imports_torch_and_localizer_initialization')\n    for folder in sorted((args.roadshow/'packages').iterdir()):\n        bm.package_begin(folder.name)\n        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()", 'grounding: begin')
            text = patch(text, "        if route not in {'not_pixel_task','abstain'} and route not in models:", "        bm.mark('read_predictions_ledgers_inventory_and_route')\n        if route not in {'not_pixel_task','abstain'} and route not in models:\n            bm.shared_begin('load_pixel_model')", 'grounding: load begin')
            text = patch(text, "            models[route]=(model,saved['size'],h.sha(path),legacy)", "            models[route]=(model,saved['size'],h.sha(path),legacy)\n            torch.cuda.synchronize()\n            bm.shared_end('load_pixel_model',route=route,checkpoint=str(path))", 'grounding: load end')
            text = patch(text, "        with (target/'regions.jsonl').open('w') as detail", "        bm.mark('prepare_candidates_and_output')\n        with (target/'regions.jsonl').open('w') as detail", 'grounding: setup')
            text = patch(text, "                probabilities,masks,presences=predict_batch(", "                bm.forward_begin(torch)\n                probabilities,masks,presences=predict_batch(", 'grounding: forward begin')
            text = patch(text, "                for item,probability,mask,present in zip(items,probabilities,masks,presences):", "                bm.forward_end(torch)\n                for item,probability,mask,present in zip(items,probabilities,masks,presences):", 'grounding: forward end')
            text = patch(text, "        for ledger in ledgers:\n            review=reviews.get", "        bm.mark('fresh_decode_forward_component_match_and_regions_write')\n        for ledger in ledgers:\n            review=reviews.get", 'grounding: inference end')
            text = patch(text, "        print(json.dumps(summaries[-1]),flush=True)", "        print(json.dumps(summaries[-1]),flush=True)\n        bm.mark('candidate_risk_severity_aggregate_and_write')\n        bm.package_end(samples=len(ledgers),family=family,images=executed,route=route,\n            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,\n            torch_peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0)", 'grounding: end')
            text = patch(text, "    q.dump(out/'scorecards.json'", "    bm.mark('shared_loop_finalization')\n    q.dump(out/'scorecards.json'", 'grounding: final begin')
            text = patch(text, "\n\n\nif __name__=='__main__':", "\n    bm.mark('shared_global_scorecards_and_severity_ranking')\n    bm.done()\n\n\nif __name__=='__main__':", 'grounding: done')
        else:
            text = patch(text, "OUT=BASE/'refresh_20261105'", "OUT=Path(" + repr(remote) + ")", 'finalize: isolated output')
            text = patch(text, "    for folder in sorted((OUT/'grounded/packages').iterdir()):", "    bm.mark('shared_imports_annotation_module_and_report_reference')\n    for folder in sorted((OUT/'grounded/packages').iterdir()):\n        bm.package_begin(folder.name)", 'finalize: begin')
            text = patch(text, "        for ledger in ledgers:\n            sid=ledger['sampleId'];signal=signals[sid]", "        bm.mark('read_current_grounded_ledgers_signals_regions')\n        for ledger in ledgers:\n            sid=ledger['sampleId'];signal=signals[sid]", 'finalize: read')
            text = patch(text, "        card=q.aggregate(ledgers);cards.append(card)", "        bm.mark('object_and_time_context_enrichment_including_actual_csv_reads')\n        card=q.aggregate(ledgers);cards.append(card)", 'finalize: context')
            text = patch(text, "        total=len(ledgers);family=ledgers[0]['family'];present=", "        bm.mark('aggregate_and_write_final_ledgers_scorecard_regions')\n        total=len(ledgers);family=ledgers[0]['family'];present=", 'finalize: aggregate')
            text = patch(text, "        print(json.dumps(dict(stage='localization_completed',**summaries[-1])),flush=True)", "        print(json.dumps(dict(stage='localization_completed',**summaries[-1])),flush=True)\n        bm.mark('coverage_per_class_counts_and_report_metadata')\n        bm.package_end(samples=total,family=family,csv_numeric_axes_read=numeric_domains,csv_absolute_domains_read=domains)", 'finalize: end')
            text = patch(text, "    assert len(reports)==12", "    bm.mark('shared_loop_finalization')\n    assert len(reports)==12", 'finalize: global begin')
            text = patch(text, "\n\n\nif __name__=='__main__':main()", "\n    bm.mark('shared_global_ranking_index_and_completion_write')\n    bm.done()\n\n\nif __name__=='__main__':main()", 'finalize: done')
        compile(text, output, 'exec')
        (root / output).write_text(text, encoding='utf-8')
        contract.append(dict(file=source, source_sha256=hashlib.sha256(original.encode()).hexdigest(),
                             output=output, output_sha256=hashlib.sha256(text.encode()).hexdigest()))
    (root / 'instrumentation_contract.json').write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--remote', required=True)
    args = parser.parse_args()
    generate(Path(__file__).parent, args.remote)
