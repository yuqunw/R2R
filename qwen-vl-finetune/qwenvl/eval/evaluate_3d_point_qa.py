#!/usr/bin/env python

import os

os.environ.setdefault("VGGT_MULTI_LAYER_INDICES", "23")

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from transformers import AutoProcessor, AutoTokenizer

from qwenvl.data.data_geometry import (
    ARROW_COLOR_MAP,
    build_geometry_transform,
    draw_arrows_on_image,
    make_arrow_mask,
)
from qwenvl.geometry.constants import NUM_SEG_TOKENS

DEFAULT_BASE_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_DATA_ROOT = str(_PROJECT_ROOT / "data")
MIN_PIXELS = 784
MAX_PIXELS = 200704


def load_model(checkpoint, base_model):
    from qwenvl.geometry.modeling_qwen3vl_geometry import (
        Qwen3VLGeometryForConditionalGeneration,
    )

    print(f"Loading model from {checkpoint} ...")
    model = Qwen3VLGeometryForConditionalGeneration.from_pretrained(
        checkpoint, dtype=torch.bfloat16
    )
    if model.geometry_tower is not None:
        model.geometry_tower.load_model()
        model.geometry_tower.to(torch.bfloat16)
    model = model.eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    processor = AutoProcessor.from_pretrained(base_model)
    processor.tokenizer = tokenizer
    ip = processor.image_processor
    if hasattr(ip, "min_pixels") and hasattr(ip, "max_pixels"):
        ip.min_pixels = MIN_PIXELS
        ip.max_pixels = MAX_PIXELS
    if hasattr(ip, "size") and isinstance(ip.size, dict):
        ip.size["shortest_edge"] = MIN_PIXELS
        ip.size["longest_edge"] = MAX_PIXELS
    return model, tokenizer, processor


_ARROW_MASK = make_arrow_mask(arrow_size=11, thickness=3)
_GEOMETRY_TRANSFORM_448 = build_geometry_transform(448)


def render_arrowed_images(sample, data_root):
    num_images = len(sample["images"])
    points_per_image = defaultdict(list)
    for pt in sample.get("points", []):
        points_per_image[pt.get("img_idx", 0)].append(pt)

    images = []
    for i in range(num_images):
        image = Image.open(os.path.join(data_root, sample["images"][i])).convert("RGB")
        pts = points_per_image.get(i, [])
        if pts:
            coords, colors = [], []
            for pt in pts:
                px = int(pt["norm_xy"][0] / 1000.0 * image.width)
                py = int(pt["norm_xy"][1] / 1000.0 * image.height)
                coords.append((max(0, min(px, image.width - 1)),
                               max(0, min(py, image.height - 1))))
                colors.append(ARROW_COLOR_MAP.get(pt["color"], (255, 255, 255)))
            image = draw_arrows_on_image(image, coords, colors, _ARROW_MASK)
        images.append(image)
    return images


def extract_question(sample):
    convs = sample.get("conversations", [])
    if not convs:
        return None
    first = 1 if convs[0].get("from") == "system" else 0
    return convs[first]["value"]


def extract_ground_truth(sample):
    for conv in sample.get("conversations", []):
        if conv.get("from") == "gpt":
            return conv["value"]
    return None


def build_prompt_inputs(processor, images, question):
    question_clean = re.sub(r"<image>\n?", "", question).strip()
    content = [{"type": "image", "image": im} for im in images]
    content.append({"type": "text", "text": question_clean})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=list(images), return_tensors="pt")
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "pixel_values": inputs["pixel_values"],
        "image_grid_thw": inputs["image_grid_thw"],
    }, question_clean


@torch.no_grad()
def run_generate(model, tokenizer, prompt_inputs, images, max_new_tokens):
    device = model.device
    kw = {k: v.to(device) for k, v in prompt_inputs.items()}
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    if model.config.fusion_block is not None:
        n = len(images)
        geo_px = torch.stack([_GEOMETRY_TRANSFORM_448(im) for im in images]).to(
            device=device, dtype=torch.bfloat16
        )
        gen_kwargs.update(
            geometry_pixel_values=[geo_px],
            geometry_patch_embeds=[torch.zeros(n, 1, 1, device=device)],
            geometry_camera_embeds=[torch.zeros(n, 1, 1, device=device)],
            post_decoder_num_frames=torch.tensor([n], device=device),
            semantic_target_frame_idx=torch.tensor([-1], device=device),
            gt_semantic_indices=torch.full((1, NUM_SEG_TOKENS), -100, device=device),
        )
    out = model.generate(**kw, **gen_kwargs)
    gen_ids = out[0, kw["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


_FLOAT_RE = re.compile(r'-?\d+(?:\.\d+)?')


def parse_color_distances(text):
    out = {}
    for color in ('red', 'green', 'blue', 'yellow', 'purple'):
        m = re.search(rf'{color}\s*:\s*({_FLOAT_RE.pattern})', text, re.IGNORECASE)
        if m:
            try:
                out[color] = float(m.group(1))
            except ValueError:
                pass
    return out


def parse_color_choice(text):
    text_l = text.strip().lower()
    if 'green' in text_l and 'blue' not in text_l:
        return 'green'
    if 'blue' in text_l and 'green' not in text_l:
        return 'blue'
    pg = text_l.find('green')
    pb = text_l.find('blue')
    if pg == -1 and pb == -1:
        return None
    if pg == -1:
        return 'blue'
    if pb == -1:
        return 'green'
    return 'green' if pg < pb else 'blue'


def parse_single_distance(text):
    m = _FLOAT_RE.search(text)
    return float(m.group(0)) if m else None


def parse_position_xy(text):
    m = re.search(r'\(?\s*(' + _FLOAT_RE.pattern + r')\s*,\s*(' + _FLOAT_RE.pattern + r')\s*\)?', text)
    if not m:
        return None
    return (float(m.group(1)), float(m.group(2)))


def parse_color_xyz(text):
    out = {}
    for color in ('red', 'green', 'blue', 'yellow', 'purple'):
        m = re.search(
            rf'{color}\s*:\s*\(?\s*({_FLOAT_RE.pattern})\s*,\s*'
            rf'({_FLOAT_RE.pattern})\s*,\s*({_FLOAT_RE.pattern})\s*\)?',
            text, re.IGNORECASE,
        )
        if m:
            try:
                out[color] = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
            except ValueError:
                pass
    return out


def parse_counting(text, n_views):
    per = []
    for i in range(1, n_views + 1):
        m = re.search(rf'Image\s*{i}\s*:\s*(\d+)', text, re.IGNORECASE)
        per.append(int(m.group(1)) if m else None)
    m = re.search(r'Total\s*:\s*(\d+)', text, re.IGNORECASE)
    total = int(m.group(1)) if m else None
    return per, total


def _delta_1_25(pred, gt):
    if gt is None or gt <= 0:
        return None
    if pred is None or pred <= 0:
        return 0.0
    return float(max(pred / gt, gt / pred) < 1.25)


def _rel_err(pred, gt):
    if gt is None or gt <= 0 or pred is None:
        return None
    return abs(pred - gt) / gt


def mra(pred, gt, start=0.5, end=0.95, interval=0.05):
    if pred is None or gt is None:
        return None
    if gt == 0:
        return float(pred == 0)
    num_pts = int((end - start) / interval + 2)
    thresholds = np.linspace(start, end, num_pts)
    rel_err = abs(pred - gt) / abs(gt)
    return float(np.mean(rel_err <= (1 - thresholds)))


def compute_metrics(results):
    by_task = defaultdict(list)
    for r in results:
        by_task[r.get('task_type', '')].append(r)

    metrics = {}

    rows = by_task.get('distance_to_camera', [])
    rels, dels, n = [], [], 0
    for r in rows:
        gt_dists = r['sample'].get('ground_truth_dists_m', [])
        points = r['sample'].get('points', [])
        pred_map = parse_color_distances(r.get('prediction', ''))
        for pt, gt in zip(points, gt_dists):
            p = pred_map.get(pt['color'])
            re_ = _rel_err(p, gt)
            de_ = _delta_1_25(p, gt)
            if re_ is not None:
                rels.append(re_)
            if de_ is not None:
                dels.append(de_)
            n += 1
    if n:
        metrics['distance_to_camera'] = {
            'count': n,
            'parsed': len(rels),
            'REL': float(np.mean(rels)) if rels else None,
            'delta_1.25': float(np.mean(dels)) if dels else None,
        }

    rows = by_task.get('distance_prediction', [])
    rels, dels = [], []
    for r in rows:
        gt = r['sample'].get('distance_m')
        p = parse_single_distance(r.get('prediction', ''))
        re_ = _rel_err(p, gt)
        de_ = _delta_1_25(p, gt)
        if re_ is not None:
            rels.append(re_)
        if de_ is not None:
            dels.append(de_)
    if rows:
        metrics['distance_prediction'] = {
            'count': len(rows),
            'parsed': len(rels),
            'REL': float(np.mean(rels)) if rels else None,
            'delta_1.25': float(np.mean(dels)) if dels else None,
        }

    rows = by_task.get('distance_infer', [])
    correct, total = 0, 0
    for r in rows:
        gt = (r['sample'].get('closer_color') or '').lower()
        pred = parse_color_choice(r.get('prediction', ''))
        if pred is None or not gt:
            continue
        total += 1
        correct += int(pred == gt)
    if rows:
        metrics['distance_infer'] = {
            'count': len(rows),
            'parsed': total,
            'accuracy': (correct / total) if total else None,
            'note': 'classification task — REL/delta_1.25 do not apply; reporting accuracy',
        }

    rows = by_task.get('position_matching', [])
    diag = math.sqrt(1000.0 ** 2 + 1000.0 ** 2)
    pck01, pck05 = [], []
    for r in rows:
        gt_xy = r['sample'].get('answer_norm_xy')
        pred = parse_position_xy(r.get('prediction', ''))
        if gt_xy is None or pred is None:
            continue
        d = math.hypot(pred[0] - gt_xy[0], pred[1] - gt_xy[1]) / diag
        pck01.append(float(d < 0.01))
        pck05.append(float(d < 0.05))
    if rows:
        metrics['position_matching'] = {
            'count': len(rows),
            'parsed': len(pck01),
            'PCK@0.01': float(np.mean(pck01)) if pck01 else None,
            'PCK@0.05': float(np.mean(pck05)) if pck05 else None,
        }

    rows = by_task.get('spatial_imagination_3d', [])
    sq_dists = []
    dists = []
    for r in rows:
        gts = r['sample'].get('positions_m', [])
        points = r['sample'].get('points', [])
        pred_map = parse_color_xyz(r.get('prediction', ''))
        for pt, gt in zip(points, gts):
            p = pred_map.get(pt['color'])
            if p is None or gt is None:
                continue
            d2 = (p[0] - gt[0]) ** 2 + (p[1] - gt[1]) ** 2 + (p[2] - gt[2]) ** 2
            sq_dists.append(d2)
            dists.append(math.sqrt(d2))
    if rows:
        metrics['spatial_imagination_3d'] = {
            'count': len(rows),
            'parsed_points': len(sq_dists),
            'MSE_m2': float(np.mean(sq_dists)) if sq_dists else None,
            'RMSE_m': float(np.sqrt(np.mean(sq_dists))) if sq_dists else None,
            'pct_within_0.5m': float(np.mean([d < 0.5 for d in dists])) if dists else None,
        }

    rows = by_task.get('counting', [])
    per_frame_mras, total_mras = [], []
    for r in rows:
        gt_per = r['sample'].get('per_frame_counts', [])
        gt_total = r['sample'].get('count')
        n_views = r['sample'].get('n_views', len(gt_per))
        pred_per, pred_total = parse_counting(r.get('prediction', ''), n_views)
        for p, g in zip(pred_per, gt_per):
            v = mra(p, g)
            if v is not None:
                per_frame_mras.append(v)
        v = mra(pred_total, gt_total)
        if v is not None:
            total_mras.append(v)
    if rows:
        per_mra = float(np.mean(per_frame_mras)) if per_frame_mras else None
        tot_mra = float(np.mean(total_mras)) if total_mras else None
        mean_mra = None
        if per_mra is not None and tot_mra is not None:
            mean_mra = (per_mra + tot_mra) / 2
        metrics['counting'] = {
            'count': len(rows),
            'per_frame_MRA': per_mra,
            'total_MRA': tot_mra,
            'mean_MRA': mean_mra,
        }

    return metrics


def print_metrics(metrics):
    print('\n' + '=' * 60)
    print('Low-Level QA Metrics')
    print('=' * 60)
    for task, m in metrics.items():
        print(f'\n[{task}]  n={m.get("count", 0)}')
        for k, v in m.items():
            if k == 'count':
                continue
            if isinstance(v, float):
                print(f'  {k}: {v:.4f}')
            else:
                print(f'  {k}: {v}')
    print('=' * 60 + '\n')


SUMMARY_COLUMNS = [
    ('distance_to_camera',     'delta_1.25'),
    ('distance_prediction',    'delta_1.25'),
    ('distance_infer',         'accuracy'),
    ('position_matching',      'PCK@0.05'),
    ('spatial_imagination_3d', 'pct_within_0.5m'),
    ('counting',               'mean_MRA'),
]


def write_summary_xlsx(metrics, xlsx_path):
    import pandas as pd
    rows = []
    for task, field in SUMMARY_COLUMNS:
        v = metrics.get(task, {}).get(field)
        rows.append({
            'task': task,
            'metric': field,
            'value (%)': (v * 100.0) if isinstance(v, (int, float)) else None,
        })
    df = pd.DataFrame(rows, columns=['task', 'metric', 'value (%)'])
    df.to_excel(xlsx_path, index=False, sheet_name='LowLevelQA')
    print(f'Summary XLSX written to {xlsx_path}')


def load_samples(annotation_file):
    samples = []
    with open(annotation_file) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def shard_name(num_shards, shard_index):
    return f"predictions_lowlevelqa_shard{shard_index}of{num_shards}.jsonl"


def merge_metrics(output_dir):
    preds = []
    for p in sorted(Path(output_dir).glob("predictions_lowlevelqa_shard*.jsonl")):
        with open(p) as f:
            preds.extend(json.loads(l) for l in f if l.strip())
    print(f"[merge] pooled {len(preds)} predictions from shard files")
    metrics = compute_metrics(preds)
    metrics_path = os.path.join(output_dir, "metrics_lowlevelqa.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[merge] wrote {metrics_path}")
    print_metrics(metrics)
    try:
        write_summary_xlsx(metrics, os.path.join(output_dir, "low_level_qa_summary.xlsx"))
    except Exception as e:
        print(f"[merge] xlsx export skipped: {e}")
    return metrics


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--annotation-file", required=True)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="cap #samples (smoke runs)")
    ap.add_argument("--merge-metrics-only", action="store_true",
                    help="pool per-shard prediction jsonls and recompute metrics")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.merge_metrics_only:
        merge_metrics(args.output_dir)
        return

    samples = load_samples(args.annotation_file)
    if args.limit is not None:
        samples = samples[: args.limit]
    shard = samples[args.shard_index :: args.num_shards]
    print(
        f"[low_level_qa] total={len(samples)} "
        f"shard={args.shard_index}/{args.num_shards} -> {len(shard)} samples"
    )

    model, tokenizer, processor = load_model(args.checkpoint, args.base_model)

    out_file = os.path.join(args.output_dir, shard_name(args.num_shards, args.shard_index))
    written = 0
    with open(out_file, "w") as fout:
        for sample in tqdm(shard, desc=f"low_level_qa shard{args.shard_index}"):
            try:
                images = render_arrowed_images(sample, args.data_root)
                question = extract_question(sample)
                prompt_inputs, question_clean = build_prompt_inputs(
                    processor, images, question
                )
                assert prompt_inputs["image_grid_thw"].shape[0] == len(images), (
                    f"image_grid_thw rows {prompt_inputs['image_grid_thw'].shape[0]} "
                    f"!= num images {len(images)}"
                )
                pred = run_generate(
                    model, tokenizer, prompt_inputs, images, args.max_new_tokens
                )
                rec = {
                    "id": sample.get("id"),
                    "task_type": sample.get("task_type"),
                    "scene_id": sample.get("scene_id"),
                    "question": question_clean,
                    "ground_truth": extract_ground_truth(sample),
                    "prediction": pred,
                    "sample": sample,
                }
            except Exception as e:
                import traceback

                traceback.print_exc()
                rec = {
                    "id": sample.get("id"),
                    "task_type": sample.get("task_type"),
                    "scene_id": sample.get("scene_id"),
                    "question": extract_question(sample),
                    "ground_truth": extract_ground_truth(sample),
                    "prediction": "",
                    "error": str(e),
                    "sample": sample,
                }
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            written += 1
    print(f"[low_level_qa] wrote {written} predictions -> {out_file}")

    if args.num_shards == 1:
        merge_metrics(args.output_dir)


if __name__ == "__main__":
    main()
