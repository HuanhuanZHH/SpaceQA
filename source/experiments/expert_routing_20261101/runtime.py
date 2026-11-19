"""Numpy-only routing runtime: request experts BEFORE receiving their predictions."""
import json
from pathlib import Path
import numpy as np

FAMILIES = ['AIA_AR_sample', 'HMI_AR_sample', 'CME_C2_sample',
            'EPB_seg_sample', 'DMSP_SSUSI', 'ULF_sample']


class ExpertRouter:
    def __init__(self, directory, route=None):
        self.directory = Path(directory)
        frozen = json.loads((self.directory / 'frozen_policy.json').read_text())
        self.experts = frozen['experts']
        self.policy = frozen['policies'][route or frozen['selected_route']]
        self.bundle = (json.loads((self.directory / self.policy['bundle']).read_text())
                       if self.policy['kind'] == 'learned' else None)

    def requests(self, current_features, family, rule_bits, probe_probabilities=None, input_available=True):
        """No filename, sample ID, tier, reference label or ground truth is accepted."""
        fi = FAMILIES.index(family); policy = self.policy; kind = policy['kind']
        plan = dict(family_index=fi, rules=np.asarray(rule_bits, bool), available=input_available)
        if not input_available or kind == 'rules':
            return [], dict(plan, weights={})
        if kind == 'learned':
            if probe_probabilities is None:
                return [self.experts[policy['probe']]], dict(plan, awaiting_probe=True)
            features = np.asarray(current_features, np.float32)
            probe = np.asarray(probe_probabilities, np.float32)
            assert features.shape == (256,) and probe.shape == (3,)
            vector = np.concatenate([features, np.eye(6)[fi], probe, plan['rules']])[None]
            bundle = self.bundle
            z = np.clip((vector - np.array(bundle['mean'])) / np.array(bundle['std']), -10, 10)
            weights = []
            for item in bundle['models']:
                parameters = item['state']
                hidden = np.maximum(z @ np.array(parameters['0.weight']).T + parameters['0.bias'], 0)
                logits = hidden @ np.array(parameters['2.weight']).T + parameters['2.bias']
                values = np.exp(logits - logits.max(1, keepdims=True))
                weights.append((values / values.sum(1, keepdims=True))[0])
            weight = np.mean(weights, axis=0)
            chosen = np.argsort(-weight, kind='stable')[:policy['topk']]
            weight = weight[chosen] / weight[chosen].sum()
            mapping = {self.experts[policy['pool'][int(j)]]: float(w) for j, w in zip(chosen, weight)}
            required = sorted(set(mapping) | {self.experts[policy['probe']]})
            return required, dict(plan, weights=mapping)
        if kind == 'single':
            mapping = {self.experts[policy['expert']]: 1.}
        elif kind == 'family':
            mapping = {self.experts[policy['assignment'][fi]]: 1.}
        elif kind == 'defect':
            assignment = [self.experts[j] for j in policy['assignment'][fi]]
            return sorted(set(assignment)), dict(plan, class_assignment=assignment)
        elif kind == 'uniform':
            mapping = {self.experts[j]: 1. / len(policy['pool']) for j in policy['pool']}
        else:
            raise ValueError(kind)
        return sorted(mapping), dict(plan, weights=mapping)

    def combine(self, plan, expert_probabilities):
        """Only selected experts are needed. The caller batches their requests."""
        if plan.get('awaiting_probe'):
            raise ValueError('Run the probe and call requests again')
        if not plan['available']:
            return plan['rules'].copy()
        if 'class_assignment' in plan:
            value = np.array([expert_probabilities[name][k] for k, name in enumerate(plan['class_assignment'])])
        else:
            value = np.zeros(3)
            for name, weight in plan['weights'].items():
                value += weight * np.asarray(expert_probabilities[name])
        threshold = np.asarray(self.policy['thresholds'])[plan['family_index']]
        return (value >= threshold) | plan['rules']
