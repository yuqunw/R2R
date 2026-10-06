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
from torch.utils.data import Dataset

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from transformers import AutoProcessor, AutoTokenizer
from transformers.video_utils import VideoMetadata

from qwenvl.data.data_geometry import GeometryLazySupervisedDataset
from qwenvl.data.data_processor import update_processor_pixels

DEFAULT_BASE_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_DATA_ROOT = str(_PROJECT_ROOT / "data")

REVSI_NOMINAL_FPS = 1.0

REVSI_TYPE_MAP = {
    "object_counting_single": "object_counting",
    "object_counting_multiple": "object_counting",
    "object_abs_distance": "object_abs_distance",
    "object_size_estimation": "object_size_estimation",
    "object_rel_distance_closest": "object_rel_distance",
    "object_rel_distance_farthest": "object_rel_distance",
    "object_rel_direction_backward_easy": "object_rel_direction_easy",
    "object_rel_direction_backward_hard": "object_rel_direction_hard",
    "object_rel_direction_forward_easy": "object_rel_direction_easy",
    "object_rel_direction_forward_hard": "object_rel_direction_hard",
}

NA_INSTRUCTION = "Please answer the question using a single word or phrase."
MC_INSTRUCTION = "Answer with the option's letter from the given choices directly."


def _make_data_args(min_pixels, max_pixels):
    from qwenvl.train.argument import DataArguments

    da = DataArguments()
    da.min_pixels = min_pixels
    da.max_pixels = max_pixels
    da.video_min_pixels = min_pixels
    da.video_max_pixels = max_pixels
    return da


class EvalBuilder(GeometryLazySupervisedDataset):

    def __init__(self, processor, tokenizer, max_num_frame, min_pixels, max_pixels):
        Dataset.__init__(self)
        self.data_args = _make_data_args(min_pixels, max_pixels)
        processor = update_processor_pixels(processor, self.data_args)
        processor.tokenizer = tokenizer
        self.processor = processor
        self.tokenizer = tokenizer
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)
        self.max_num_frame = max_num_frame
        self.use_qwen_grid_geometry = True
        self.use_timestamp_markers = True
        self.use_vsi_video_path = False
        self.use_semantic_video_path = False
        self._anchor_token_ids = None
    def video_prompt(self, frames, question, stems, fps):
        q = question
        for marker in ("<image>\n", "<video>\n", "<image>", "<video>"):
            q = q.replace(marker, "")
        messages = [
            {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": q.strip()}]}
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        video = np.stack([np.asarray(f) for f in frames])
        meta = VideoMetadata(total_num_frames=len(frames), fps=fps, frames_indices=stems)
        inputs = self.processor(
            text=[text],
            videos=[video],
            video_metadata=[meta],
            do_sample_frames=False,
            return_tensors="pt",
            size={
                "shortest_edge": len(frames) * self.data_args.min_pixels,
                "longest_edge": len(frames) * self.data_args.max_pixels,
            },
        )
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values_videos": inputs["pixel_values_videos"],
            "video_grid_thw": inputs["video_grid_thw"],
        }

    def image_prompt(self, frames, question, markers):
        special = "".join(f"{m}<image>" for m in markers)
        q = question
        for token in ("<image>\n", "<video>\n", "<image>", "<video>"):
            q = q.replace(token, "")
        user_text = special + "\n" + q.strip()
        image_pool = list(frames)
        content = []
        for seg in re.split(r"(<image>)", user_text):
            if seg == "<image>":
                content.append({"type": "image", "image": image_pool.pop(0)})
            elif seg.strip():
                content.append({"type": "text", "text": seg.strip()})
        messages = [{"role": "user", "content": content}]
        res = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        return {
            "input_ids": res["input_ids"],
            "attention_mask": res["attention_mask"],
            "pixel_values": res["pixel_values"],
            "image_grid_thw": res["image_grid_thw"],
        }


VSI_MCA_QUESTION_TYPES = {
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
}


def _vsi_format_prompt(question, question_type, options):
    if question_type in VSI_MCA_QUESTION_TYPES and options:
        opts = "\n".join(str(o) for o in options)
        return f"These are frames of a video.\n{question}\nOptions:\n{opts}\n{MC_INSTRUCTION}"
    return f"These are frames of a video.\n{question}\n{NA_INSTRUCTION}"


def load_vsi_samples(annotation, video_root):
    samples = []
    with open(annotation) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            samples.append(
                {
                    "id": str(d["id"]),
                    "data_source": d["dataset"],
                    "scene_name": d["scene_name"],
                    "video": os.path.join(video_root, d["dataset"], f"{d['scene_name']}.mp4"),
                    "question": _vsi_format_prompt(d["question"], d["question_type"], d["options"]),
                    "ground_truth": str(d["ground_truth"]),
                    "question_type": d["question_type"],
                }
            )
    return samples


def _revsi_format_prompt(question, options):
    if options:
        opts = "\n".join(
            f'{chr(ord("A") + i)}. '
            + (o.split(". ", 1)[-1] if o[:2] in ("A.", "B.", "C.", "D.") else o)
            for i, o in enumerate(options)
        )
        return f"These are frames of a video.\n{question}\nOptions:\n{opts}\n{MC_INSTRUCTION}"
    return f"These are frames of a video.\n{question}\n{NA_INSTRUCTION}"


def load_revsi_samples(revsi_config, revsi_split, video_root):
    from datasets import load_dataset

    ds = load_dataset("3dlg-hcvc/ReVSI", revsi_config, split=revsi_split)
    samples = []
    for s in ds:
        options = list(s["options"]) if s["options"] else None
        samples.append(
            {
                "id": f"revsi_{s['id']}",
                "data_source": s["dataset"],
                "scene_name": s["scene_id"],
                "video": os.path.join(video_root, revsi_config, f"{s['scene_id']}.mp4"),
                "question": _revsi_format_prompt(s["question"], options),
                "ground_truth": s["ground_truth"],
                "question_type": REVSI_TYPE_MAP.get(s["question_type"], s["question_type"]),
                "original_question_type": s["question_type"],
            }
        )
    return samples


def load_video(path, max_num_frame):
    from decord import VideoReader

    vr = VideoReader(path, num_threads=1)
    idx = np.linspace(0, len(vr) - 1, min(max_num_frame, len(vr)), dtype=int).tolist()
    frames = [Image.fromarray(f) for f in vr.get_batch(idx).asnumpy()]
    return frames, idx, vr.get_avg_fps()


@torch.no_grad()
def run_generate(model, tokenizer, prompt_inputs, geometry, num_frames, max_new_tokens):
    device = model.device
    geo_px, patch, cam = geometry
    kw = {k: v.to(device) for k, v in prompt_inputs.items()}
    out = model.generate(
        **kw,
        geometry_pixel_values=geo_px.to(device=device, dtype=torch.bfloat16),
        geometry_patch_embeds=patch.to(device),
        geometry_camera_embeds=cam.to(device),
        post_decoder_num_frames=torch.tensor([num_frames], device=device),
        semantic_target_frame_idx=torch.tensor([-1], device=device),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    gen_ids = out[0, kw["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


VSI_MCA_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]
VSI_NA_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]
VSI_DISPLAY_ORDER = [
    "object_counting",
    "object_abs_distance",
    "object_size_estimation",
    "room_size_estimation",
    "object_rel_distance",
    "object_rel_direction",
    "route_planning",
    "obj_appearance_order",
]
VSI_MACRO_AVG_BASE_TYPES = {"object_rel_direction"}


def _extract_letter(text):
    if not text:
        return None
    s = str(text)
    m = re.match(r"^([A-D])\b", s.strip().upper())
    if m:
        return m.group(1)
    m = re.search(r"(?:answer|option)(?:\s+is)?[\s:]+([A-D])\b", s, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-D])\b", s.upper())
    if m:
        return m.group(1)
    return None


def _extract_num(text):
    if not text:
        return None
    m = re.search(r"(\d+\.?\d*)", str(text))
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _mra(pred, target, start=0.5, end=0.95, interval=0.05):
    num_pts = int((end - start) / interval + 2)
    conf = np.linspace(start, end, num_pts)
    return float((abs(pred - target) / target <= 1 - conf).mean())


def _acc_mca(results):
    correct = total = 0
    for r in results:
        g = _extract_letter(r.get("ground_truth", ""))
        p = _extract_letter(r.get("prediction", ""))
        if g and p:
            total += 1
            correct += int(g == p)
    return (correct / total * 100 if total else 0.0), correct, total


def _acc_na(results):
    scores = []
    for r in results:
        g = _extract_num(str(r.get("ground_truth", "")))
        p = _extract_num(str(r.get("prediction", "")))
        if g is not None and p is not None and g > 0:
            scores.append(_mra(p, g))
    if not scores:
        return 0.0, 0, 0
    mean = float(np.mean(scores)) * 100
    correct = sum(1 for s in scores if s >= 0.5)
    return mean, correct, len(scores)


def compute_vsi_metrics(results):
    by_type = defaultdict(list)
    for r in results:
        qt = r.get("question_type")
        if qt:
            by_type[qt].append(r)

    grouped = defaultdict(list)
    for qt, rs in by_type.items():
        base = qt.replace("_easy", "").replace("_medium", "").replace("_hard", "")
        grouped[base].extend(rs)

    mca_base = {t.replace("_easy", "").replace("_medium", "").replace("_hard", "") for t in VSI_MCA_TYPES}
    ordered = [k for k in VSI_DISPLAY_ORDER if k in grouped]
    ordered += [k for k in sorted(grouped) if k not in VSI_DISPLAY_ORDER]

    per_type = []
    mca_results, na_results = [], []
    for base in ordered:
        rs = grouped[base]
        if base in mca_base:
            mca_results.extend(rs)
            if base in VSI_MACRO_AVG_BASE_TYPES:
                per_diff, csum, tsum = [], 0, 0
                for qt, qrs in by_type.items():
                    if qt.replace("_easy", "").replace("_medium", "").replace("_hard", "") != base:
                        continue
                    acc, c, t = _acc_mca(qrs)
                    if t > 0:
                        per_diff.append(acc)
                        csum += c
                        tsum += t
                acc = float(np.mean(per_diff)) if per_diff else 0.0
                correct, total = csum, tsum
                mtype = "MCA (macro-avg)"
            else:
                acc, correct, total = _acc_mca(rs)
                mtype = "MCA"
        elif base in VSI_NA_TYPES:
            na_results.extend(rs)
            acc, correct, total = _acc_na(rs)
            mtype = "NA (MRA)"
        else:
            continue
        per_type.append(
            {"question_type": base, "num_questions": total, "accuracy": acc, "metric_type": mtype}
        )

    out = {"num_samples": len(results), "per_question_type": per_type}
    mca_acc, mca_c, mca_t = _acc_mca(mca_results) if mca_results else (0.0, 0, 0)
    na_acc, na_c, na_t = _acc_na(na_results) if na_results else (0.0, 0, 0)
    if mca_results:
        out["overall_mca"] = {"accuracy": mca_acc, "correct": mca_c, "total": mca_t}
    if na_results:
        out["overall_na"] = {"accuracy": na_acc, "correct": na_c, "total": na_t}
    if mca_results and na_results:
        out["overall_weighted"] = (mca_acc * mca_t + na_acc * na_t) / (mca_t + na_t)
    if per_type:
        out["macro_avg"] = sum(p["accuracy"] for p in per_type) / len(per_type)
    return out


REVSI_TASK_STRUCTURE = {
    "object_counting": {"mode": "na", "sub_buckets": ["object_counting_single", "object_counting_multiple"]},
    "object_abs_distance": {"mode": "na", "sub_buckets": ["object_abs_distance"]},
    "object_size_estimation": {"mode": "na", "sub_buckets": ["object_size_estimation"]},
    "room_size_estimation": {"mode": "na", "sub_buckets": ["room_size_estimation_single", "room_size_estimation_multiple"]},
    "object_rel_distance": {"mode": "mcq", "sub_buckets": ["object_rel_distance_closest", "object_rel_distance_farthest"]},
    "object_rel_direction": {
        "mode": "mcq",
        "sub_buckets": [
            "object_rel_direction_forward_easy",
            "object_rel_direction_backward_easy",
            "object_rel_direction_forward_hard",
            "object_rel_direction_backward_hard",
        ],
    },
    "route_planning": {"mode": "mcq", "sub_buckets": ["route_planning"]},
}


def _revsi_score_mcq(pred, gt):
    p, g = _extract_letter(pred), _extract_letter(gt)
    if not p or not g:
        return 0.0
    return 1.0 if p == g else 0.0


def _revsi_score_na(pred, gt):
    p, g = _extract_num(pred), _extract_num(gt)
    if p is None or g is None or g <= 0:
        return 0.0
    return _mra(p, g)


def compute_revsi_metrics(results, missing_as="zero"):
    by_bucket = defaultdict(list)
    for r in results:
        sb = r.get("original_question_type") or r.get("question_type")
        if sb:
            by_bucket[sb].append(r)

    rows = []
    for base, info in REVSI_TASK_STRUCTURE.items():
        score_fn = _revsi_score_mcq if info["mode"] == "mcq" else _revsi_score_na
        bucket_accs, breakdown, total_n = [], [], 0
        for sb in info["sub_buckets"]:
            rs = by_bucket.get(sb, [])
            total_n += len(rs)
            if not rs:
                breakdown.append({"bucket": sb, "n": 0, "acc": None})
                continue
            acc = float(np.mean([score_fn(r.get("prediction", ""), r.get("ground_truth", "")) for r in rs])) * 100
            bucket_accs.append(acc)
            breakdown.append({"bucket": sb, "n": len(rs), "acc": acc})
        if bucket_accs:
            base_acc = float(np.mean(bucket_accs))
        elif missing_as == "nan":
            base_acc = math.nan
        else:
            base_acc = 0.0
        rows.append(
            {
                "question_type": base,
                "num_questions": total_n,
                "accuracy": base_acc,
                "metric_type": "MCA" if info["mode"] == "mcq" else "NA (MRA)",
                "sub_bucket_breakdown": breakdown,
            }
        )

    accs = [r["accuracy"] for r in rows]
    out = {"num_samples": len(results), "missing_as": missing_as, "per_question_type": rows}
    if missing_as == "nan":
        finite = [a for a in accs if not math.isnan(a)]
        out["macro_avg_present"] = float(np.mean(finite)) if finite else float("nan")
    else:
        out["macro_avg_all"] = float(np.mean(accs)) if accs else float("nan")
        finite = [r["accuracy"] for r in rows if r["num_questions"] > 0]
        out["macro_avg_present"] = float(np.mean(finite)) if finite else float("nan")
    return out


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
    return model, tokenizer, processor


def read_video_pair_geometry(checkpoint):
    with open(os.path.join(checkpoint, "config.json")) as f:
        return bool(json.load(f).get("video_pair_geometry", False))


@torch.no_grad()
def load_scene_context(builder, sample, max_num_frame, is_revsi, model=None):
    frames, stems, fps = load_video(sample["video"], max_num_frame)
    if is_revsi:
        fps = REVSI_NOMINAL_FPS
    geo_px = builder._geometry_pixels(frames)
    if model is not None and model.geometry_tower is not None and model.fusion_block is not None:
        views = geo_px.to(device=model.device, dtype=model.get_input_embeddings().weight.dtype)
        cam, patch, *_ = model.extract_geometry_feature(views.unsqueeze(0))
    else:
        patch, cam = builder._zero_geometry(len(frames))
    return frames, stems, fps, (geo_px, patch, cam)


def build_qa_inputs(builder, sample, vision_mode, max_num_frame, is_revsi, ctx=None):
    if ctx is None:
        ctx = load_scene_context(builder, sample, max_num_frame, is_revsi)
    frames, stems, fps, geometry = ctx
    num_frames = len(frames)
    if vision_mode == "video":
        prompt = builder.video_prompt(frames, sample["question"], stems, fps)
    else:
        markers = [f"<{s / fps:.1f} seconds>" for s in stems]
        prompt = builder.image_prompt(frames, sample["question"], markers)
    return prompt, geometry, num_frames


def resolve_vision_mode(args, checkpoint):
    if args.vision_mode != "auto":
        return args.vision_mode
    return "video" if read_video_pair_geometry(checkpoint) else "image"


def shard_name(benchmark, max_num_frame, num_shards, shard_index):
    tag = f"{benchmark}_{max_num_frame}f"
    if num_shards > 1:
        return f"predictions_{tag}_shard{shard_index}of{num_shards}.jsonl"
    return f"predictions_{tag}_shard0of1.jsonl"


_XLSX_COLUMNS = {
    "vsibench": [
        ("Obj. Cnt.", "object_counting"), ("Abs. Dist.", "object_abs_distance"),
        ("Obj. Size", "object_size_estimation"), ("Room Size", "room_size_estimation"),
        ("Rel. Dist.", "object_rel_distance"), ("Rel. Dir.", "object_rel_direction"),
        ("Route Plan", "route_planning"), ("Appr. Order", "obj_appearance_order"),
        ("Overall MCA", "@overall_mca"), ("Overall NA", "@overall_na"),
        ("Overall (weighted)", "@overall_weighted"), ("Macro avg", "@macro_avg"),
    ],
    "revsi": [
        ("Obj. Cnt.", "object_counting"), ("Abs. Dist.", "object_abs_distance"),
        ("Obj. Size", "object_size_estimation"), ("Room Size", "room_size_estimation"),
        ("Rel. Dist.", "object_rel_distance"), ("Rel. Dir.", "object_rel_direction"),
        ("Route Plan", "route_planning"),
        ("Overall (macro present)", "@macro_avg_present"), ("Overall (macro all)", "@macro_avg_all"),
    ],
}


def export_metrics_xlsx(metrics, benchmark, xlsx_path):
    cols = _XLSX_COLUMNS.get(benchmark)
    if cols is None:
        return
    by_type = {p["question_type"]: p for p in metrics.get("per_question_type", [])}
    rows = []
    for label, key in cols:
        if key.startswith("@"):
            v = metrics.get(key[1:])
            if isinstance(v, dict):
                v = v.get("accuracy")
        else:
            p = by_type.get(key)
            v = p.get("accuracy") if (p and (p.get("num_questions") or 0) > 0) else None
        rows.append({"metric": label, "key": key.lstrip("@"),
                     "value (%)": (round(v, 2) if isinstance(v, (int, float)) else None)})
    import pandas as pd
    sheet = {"vsibench": "VSI-Bench", "revsi": "ReVSI"}.get(benchmark, benchmark)
    pd.DataFrame(rows, columns=["metric", "key", "value (%)"]).to_excel(
        xlsx_path, index=False, sheet_name=sheet)
    print(f"[merge] wrote {xlsx_path}")


def merge_metrics(args):
    tag = f"{args.benchmark}_{args.max_num_frame}f"
    preds = []
    for p in sorted(Path(args.output_dir).glob(f"predictions_{tag}_shard*.jsonl")):
        with open(p) as f:
            preds.extend(json.loads(l) for l in f if l.strip())
    print(f"[merge] pooled {len(preds)} predictions from shard files")
    if args.benchmark == "vsibench":
        metrics = compute_vsi_metrics(preds)
    else:
        metrics = compute_revsi_metrics(preds, missing_as=args.missing_as)
    out_path = os.path.join(args.output_dir, f"metrics_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"[merge] wrote {out_path}")
    try:
        export_metrics_xlsx(metrics, args.benchmark,
                            os.path.join(args.output_dir, f"metrics_{tag}.xlsx"))
    except Exception as e:
        print(f"[merge] xlsx export skipped: {e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--benchmark", required=True, choices=["vsibench", "revsi"])
    ap.add_argument("--max-num-frame", type=int, default=32)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--limit", type=int, default=None, help="cap #samples (smoke runs)")
    ap.add_argument("--vision-mode", choices=["auto", "video", "image"], default="auto")
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--min-pixels", type=int, default=784)
    ap.add_argument("--max-pixels", type=int, default=200704)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--missing-as", choices=["zero", "nan"], default="zero", help="revsi metric fill")
    ap.add_argument("--vsi-annotation", default=None, help="official VSI-Bench test.jsonl")
    ap.add_argument("--vsi-video-root", default=None, help="dir with <dataset>/<scene>.mp4")
    ap.add_argument("--revsi-config", default=None)
    ap.add_argument("--revsi-split", default="test")
    ap.add_argument("--revsi-video-root", default=None, help="dir with <N>_frame/<scene>.mp4")
    ap.add_argument("--merge-metrics-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="build first samples, print prompts, no GPU")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.merge_metrics_only:
        merge_metrics(args)
        return

    if args.vsi_annotation is None:
        args.vsi_annotation = os.path.join(args.data_root, "VSI-Bench", "test.jsonl")
    if args.vsi_video_root is None:
        args.vsi_video_root = os.path.join(args.data_root, "VSI-Bench")
    if args.revsi_config is None:
        args.revsi_config = f"{args.max_num_frame}_frame"
    if args.revsi_video_root is None:
        args.revsi_video_root = os.path.join(args.data_root, "ReVSI", "videos")

    vision_mode = resolve_vision_mode(args, args.checkpoint)
    is_revsi = args.benchmark == "revsi"
    max_new = args.max_new_tokens or 64

    if args.benchmark == "vsibench":
        samples = load_vsi_samples(args.vsi_annotation, args.vsi_video_root)
    else:
        samples = load_revsi_samples(args.revsi_config, args.revsi_split, args.revsi_video_root)

    if args.limit is not None:
        samples = samples[: args.limit]
    _scene_key = lambda s: s["video"]
    _scene_order, _scene_map = [], defaultdict(list)
    for _s in samples:
        k = _scene_key(_s)
        if k not in _scene_map:
            _scene_order.append(k)
        _scene_map[k].append(_s)
    _my_scenes = _scene_order[args.shard_index :: args.num_shards]
    shard = [s for k in _my_scenes for s in _scene_map[k]]
    print(
        f"[{args.benchmark}] vision_mode={vision_mode} frames={args.max_num_frame} "
        f"total={len(samples)} shard={args.shard_index}/{args.num_shards} -> {len(shard)} samples"
    )

    if args.dry_run:
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=False)
        processor = AutoProcessor.from_pretrained(args.base_model)
        builder = EvalBuilder(processor, tokenizer, args.max_num_frame, args.min_pixels, args.max_pixels)
        for sample in shard[:3]:
            print("\n" + "=" * 70)
            print(f"id={sample['id']} scene={sample.get('scene_name')} qtype={sample.get('question_type')}")
            prompt, geometry, nf = build_qa_inputs(builder, sample, vision_mode, args.max_num_frame, is_revsi)
            geo_px, patch, cam = geometry
            ids = prompt["input_ids"]
            decoded = tokenizer.decode(ids[0], skip_special_tokens=False)
            print(f"num_frames={nf}  input_ids={tuple(ids.shape)}")
            for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
                if key in prompt:
                    print(f"  {key}: {tuple(prompt[key].shape)}  {prompt[key].tolist() if key.endswith('grid_thw') else ''}")
            print(f"  geometry_pixel_values={tuple(geo_px.shape)} patch={tuple(patch.shape)} cam={tuple(cam.shape)}")
            n_video_pad = int((ids[0] == 151656).sum())
            n_image_pad = int((ids[0] == 151655).sum())
            print(f"  video_pad_tokens={n_video_pad}  image_pad_tokens={n_image_pad}")
            print(f"  '<... seconds>' markers present: {bool(re.search(r'<\d+\.\d+ seconds>', decoded))}")
            print(f"  prompt tail: ...{decoded[-320:]}")
        print("\n[dry-run] OK — no GPU used.")
        return

    model, tokenizer, processor = load_model(args.checkpoint, args.base_model)
    builder = EvalBuilder(processor, tokenizer, args.max_num_frame, args.min_pixels, args.max_pixels)

    out_file = os.path.join(args.output_dir, shard_name(args.benchmark, args.max_num_frame, args.num_shards, args.shard_index))
    from tqdm import tqdm

    written = 0
    _ctx_key = None
    _ctx = None
    with open(out_file, "w") as fout:
        for sample in tqdm(shard, desc=f"{args.benchmark} shard{args.shard_index}"):
            try:
                k = _scene_key(sample)
                if k != _ctx_key:
                    _ctx = load_scene_context(builder, sample, args.max_num_frame, is_revsi, model=model)
                    _ctx_key = k
                prompt, geometry, nf = build_qa_inputs(
                    builder, sample, vision_mode, args.max_num_frame, is_revsi, ctx=_ctx)
                pred = run_generate(model, tokenizer, prompt, geometry, nf, max_new)
                rec = {
                    "id": sample["id"],
                    "scene_name": sample["scene_name"],
                    "data_source": sample.get("data_source"),
                    "question": sample["question"],
                    "ground_truth": sample["ground_truth"],
                    "prediction": pred,
                    "num_frames": nf,
                    "question_type": sample.get("question_type", ""),
                }
                if "original_question_type" in sample:
                    rec["original_question_type"] = sample["original_question_type"]
            except Exception as e:
                import traceback

                traceback.print_exc()
                rec = {"id": sample.get("id"), "scene_name": sample.get("scene_name"), "error": str(e), "prediction": ""}
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            written += 1
    print(f"[{args.benchmark}] wrote {written} predictions -> {out_file}")

    if args.num_shards == 1:
        merge_metrics(args)


if __name__ == "__main__":
    main()
