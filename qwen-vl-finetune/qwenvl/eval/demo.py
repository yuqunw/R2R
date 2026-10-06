import os

os.environ.setdefault("VGGT_MULTI_LAYER_INDICES", "23")

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from qwenvl.eval.evaluate_geometry_bench import (
    DEFAULT_BASE_MODEL,
    EvalBuilder,
    load_model,
    run_generate,
)

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def load_video(path, num_frames):
    from decord import VideoReader

    vr = VideoReader(path, num_threads=1)
    idx = np.linspace(0, len(vr) - 1, min(num_frames, len(vr)), dtype=int)
    frames = [Image.fromarray(f) for f in vr.get_batch(idx).asnumpy()]
    times = [i / vr.get_avg_fps() for i in idx]
    return frames, times


def load_frames(frame_dir, num_frames, fps):
    files = sorted(f for f in os.listdir(frame_dir) if f.lower().endswith(IMAGE_EXTS))
    idx = np.linspace(0, len(files) - 1, min(num_frames, len(files)), dtype=int)
    frames = [Image.open(os.path.join(frame_dir, files[i])).convert("RGB") for i in idx]
    times = [i / fps for i in idx]
    return frames, times


def main():
    ap = argparse.ArgumentParser(description="Ask a spatial question about a video with Render2Reason.")
    ap.add_argument("--checkpoint", required=True, help="local checkpoint directory")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="input video file")
    src.add_argument("--frames", help="directory of frames (sorted by file name)")
    ap.add_argument("--question", required=True)
    ap.add_argument("--num-frames", type=int, default=32)
    ap.add_argument("--fps", type=float, default=1.0, help="frame rate of --frames, used for timestamps")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    args = ap.parse_args()

    if args.video:
        frames, times = load_video(args.video, args.num_frames)
    else:
        frames, times = load_frames(args.frames, args.num_frames, args.fps)

    model, tokenizer, processor = load_model(args.checkpoint, args.base_model)
    builder = EvalBuilder(processor, tokenizer, args.num_frames, 784, 200704)

    markers = [f"<{t:.1f} seconds>" for t in times]
    prompt = builder.image_prompt(frames, args.question, markers)
    geometry = builder.qwen_grid_geometry(frames, scene_dir=None)
    answer = run_generate(model, tokenizer, prompt, geometry, len(frames), args.max_new_tokens)
    print(answer)


if __name__ == "__main__":
    main()
