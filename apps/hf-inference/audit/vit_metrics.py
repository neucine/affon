"""Shared strict ViT comparison metrics; no tolerance relaxation."""
import torch


def compare(actual, expected):
    assert actual.shape == expected.shape and actual.numel() > 0
    error = (actual.double() - expected.double()).abs()
    failures = (~torch.isfinite(error)) | (error > 1e-4 + 1e-4 * expected.double().abs())
    return {
        'passed': not failures.any().item(),
        'elements': actual.numel(),
        'mismatches': failures.sum().item(),
        'max_absolute_error': error.max().item(),
        'rms_error': error.square().mean().sqrt().item(),
    }


def summarize(hidden, logits, reference, prefix):
    layers = [compare(value, reference[f'{prefix}.hidden_{i}']) for i, value in enumerate(hidden)]
    final = reference[f'{prefix}.hidden_{len(hidden)-1}']
    return {
        'first_failing_hidden_state': next((i for i, row in enumerate(layers) if not row['passed']), None),
        'hidden_states': layers,
        'final_cls_token': compare(hidden[-1][:, :1], final[:, :1]),
        'final_patch_tokens': compare(hidden[-1][:, 1:], final[:, 1:]),
        'logits': compare(logits, reference[f'{prefix}.output']),
        'top1_matches': logits.argmax(-1).tolist() == reference[f'{prefix}.output'].argmax(-1).tolist(),
    }
