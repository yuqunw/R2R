import json
import os
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms as T
import transformers
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode

from ..geometry.constants import (
    ALL_GEOMETRY_OUTPUT_TOKENS,
    ALL_SEMANTIC_OUTPUT_TOKENS,
    GEOMETRY_CAM_DIM,
    GEOMETRY_PATCH_DIM,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_GEOMETRY_PATCHES,
    NUM_SEG_TOKENS,
)
from ..geometry.question_type_constants import question_type_to_id
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from .data_processor import (
    DataCollatorForSupervisedDataset,
    update_processor_pixels,
)
from .rope2d import get_rope_index_3

IGNORE_INDEX = -100


class GeometryCacheStaleError(RuntimeError):
    pass


def _qtype_tensor(data_item):
    qt = data_item.get('task_type') or data_item.get('question_type')
    return torch.tensor([question_type_to_id(qt)], dtype=torch.long)

ARROW_COLOR_MAP = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 180, 255),
    'purple': (180, 0, 255),
    'yellow': (255, 255, 0),
}


def build_geometry_transform(input_size=448):
    size = (input_size, input_size) if isinstance(input_size, int) else tuple(input_size)
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize(size, interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_anchor_token_ids(tokenizer) -> torch.Tensor:
    ids = []
    unk = getattr(tokenizer, 'unk_token_id', None)
    for tok in list(ALL_GEOMETRY_OUTPUT_TOKENS) + list(ALL_SEMANTIC_OUTPUT_TOKENS):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != unk:
            ids.append(int(tid))
    return torch.tensor(sorted(set(ids)), dtype=torch.long)


def mask_anchor_token_labels_(labels: torch.Tensor, anchor_ids: torch.Tensor) -> torch.Tensor:
    if anchor_ids.numel() == 0:
        return labels
    mask = torch.isin(labels, anchor_ids.to(labels.device))
    if mask.any():
        labels.masked_fill_(mask, IGNORE_INDEX)
    return labels


def make_arrow_mask(arrow_size=11, thickness=3):
    half_size = arrow_size // 2
    height = arrow_size + 1
    width = arrow_size + half_size + 2
    mask = np.zeros((height, width), dtype=bool)
    center_y = height // 2
    t = max(1, int(thickness))

    def paint(x, y):
        r = t // 2
        y0 = max(0, y - r)
        y1 = min(height, y + r + 1)
        x0 = max(0, x - r)
        x1 = min(width, x + r + 1)
        mask[y0:y1, x0:x1] = True

    for x in range(width):
        paint(x, center_y)
    tip_x = width - 1
    for dy in range(1, half_size + 1):
        x_pos = tip_x - dy
        if center_y - dy >= 0:
            paint(x_pos, center_y - dy)
        if center_y + dy < height:
            paint(x_pos, center_y + dy)
    return mask


def draw_arrows_on_image(image, coords, colors, arrow_mask):
    img_array = np.array(image)
    img_h, img_w = img_array.shape[:2]
    mask_h, mask_w = arrow_mask.shape
    gap = 1

    for (pixel_x, pixel_y), color in zip(coords, colors):
        top_left_y = pixel_y - mask_h // 2
        top_left_x = pixel_x - mask_w - gap

        mask_y_start = max(0, -top_left_y)
        mask_y_end = min(mask_h, img_h - top_left_y)
        mask_x_start = max(0, -top_left_x)
        mask_x_end = min(mask_w, img_w - top_left_x)

        img_y_start = max(0, top_left_y)
        img_y_end = img_y_start + (mask_y_end - mask_y_start)
        img_x_start = max(0, top_left_x)
        img_x_end = img_x_start + (mask_x_end - mask_x_start)

        if mask_y_end > mask_y_start and mask_x_end > mask_x_start:
            mask_region = arrow_mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]
            img_array[img_y_start:img_y_end, img_x_start:img_x_end][mask_region] = color

    return Image.fromarray(img_array)


def original_video_fps(path_str):
    if ('scannetpp' in path_str) or ('arkitscenes' in path_str):
        return 60.0
    if 'scannet' in path_str:
        return 24.0
    raise RuntimeError(f"Unknown dataset name: {path_str}")


def frame_stem_seconds(filename, fps):
    return int(os.path.splitext(os.path.basename(filename))[0].split('_')[-1]) / fps


def load_frames_for_training(frames_dir, max_num_frames=32):
    fps = original_video_fps(frames_dir)
    rgb_files = sorted(os.listdir(frames_dir))
    files_num = len(rgb_files)
    frame_idx = [i for i in range(0, files_num)]
    if len(frame_idx) > max_num_frames:
        uniform_sampled_frames = np.linspace(0, files_num - 1, max_num_frames, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
    frames = [Image.open(os.path.join(frames_dir, rgb_files[i])).convert('RGB') for i in frame_idx]
    timestamps = [frame_stem_seconds(rgb_files[i], fps) for i in frame_idx]
    return frames, timestamps


class GeometryLazySupervisedDataset(Dataset):

    def __init__(self, processor, data_args):
        super().__init__()
        meta = json.load(open(data_args.meta_path))
        self.entries = []
        for name, cfg in meta.items():
            with open(cfg['annotation']) as f:
                lines = f.readlines()
            repeat_time = cfg.get('repeat_time', 1)
            if repeat_time < 1:
                lines = lines[:int(len(lines) * repeat_time)]
            else:
                lines = lines * int(repeat_time)
            self.entries.extend((line, cfg['root']) for line in lines)
            print(f'[data_geometry] dataset {name}: {len(lines)} samples (root={cfg["root"]})')
        print(f'[data_geometry] total samples: {len(self.entries)}')

        processor = update_processor_pixels(processor, data_args)
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.data_args = data_args
        self.merge_size = getattr(processor.image_processor, 'merge_size', 2)
        self.geometry_transform = build_geometry_transform(448)
        self.max_num_frame = data_args.max_num_frame
        self.cached_geometry_features = data_args.cached_geometry_features
        self.video_cache_name = (
            'VGGT_features_4_16.pt' if self.max_num_frame == 16 else 'VGGT_features.pt'
        )
        self.use_timestamp_markers = getattr(data_args, 'use_timestamp_markers', False)
        self.use_qwen_grid_geometry = getattr(data_args, 'use_qwen_grid_geometry', False)
        if self.use_qwen_grid_geometry:
            assert self.max_num_frame in (16, 32), (
                f'no Qwen-grid cache for max_num_frame={self.max_num_frame} '
                f'(extracted: 16, 32)'
            )
            assert GEOMETRY_PATCH_DIM == 2048, (
                'use_qwen_grid_geometry expects last-layer features — '
                'set VGGT_MULTI_LAYER_INDICES=23 '
                f'(got GEOMETRY_PATCH_DIM={GEOMETRY_PATCH_DIM})'
            )
            self.video_cache_name = (
                'VGGT_features_24_32_last.pt' if self.max_num_frame == 16
                else 'VGGT_features_24_32_last_32frames.pt'
            )
        if data_args.mask_anchor_token_labels:
            self._anchor_token_ids = build_anchor_token_ids(self.tokenizer)
        else:
            self._anchor_token_ids = None
        self._arrow_mask = make_arrow_mask(arrow_size=11, thickness=3)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3
        for attempt_idx in range(num_base_retries):
            try:
                return self._route(i)
            except Exception as e:
                print(f'[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:', e)
                time.sleep(1)
        for attempt_idx in range(num_base_retries):
            next_index = min(i + 1 + attempt_idx, len(self.entries) - 1)
            try:
                return self._route(next_index)
            except Exception as e:
                print(f'[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:', e)
        return self._route(i)

    def _route(self, i):
        line, root = self.entries[i]
        data_item = json.loads(line)
        if 'points' in data_item and 'images' in data_item:
            return self.low_level_qa_get_item(data_item, root)
        elif data_item.get('task_type') == 'semantic_rendering':
            return self.semantic_rendering_get_item(data_item, root)
        elif 'image' in data_item and data_item['image']:
            return self.image_qa_get_item(data_item, root)
        elif 'video' in data_item and data_item['video']:
            return self.video_get_item(data_item, root)
        else:
            raise ValueError(f'Unsupported record (keys={list(data_item.keys())}, '
                             f'task_type={data_item.get("task_type")})')


    def _encode_conversation(self, conversations: List[Dict], images: List[Image.Image]) -> Dict:
        image_pool = list(images)
        messages = []
        for turn in conversations:
            src = turn['from']
            if src == 'human':
                role = 'user'
            elif src == 'gpt':
                role = 'assistant'
            elif src == 'system':
                role = 'system'
            else:
                raise NotImplementedError(f'Invalid role: {src!r}')
            text = turn['value']
            if role == 'user':
                content = []
                for seg in re.split(r'(<image>)', text):
                    if seg == '<image>':
                        if not image_pool:
                            raise ValueError('More <image> placeholders than images')
                        content.append({'type': 'image', 'image': image_pool.pop(0)})
                    elif seg.strip():
                        content.append({'type': 'text', 'text': seg.strip()})
                messages.append({'role': role, 'content': content})
            else:
                messages.append({'role': role, 'content': [{'type': 'text', 'text': text}]})
        if image_pool:
            raise ValueError(f'{len(image_pool)} image(s) remain unused')

        full_result = self.processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors='pt'
        )
        input_ids = full_result['input_ids']
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids).unsqueeze(0)

        labels = torch.full_like(input_ids, IGNORE_INDEX)
        input_ids_flat = input_ids[0].tolist()
        L = len(input_ids_flat)
        pos = 0
        while pos < L:
            if input_ids_flat[pos] == 77091:
                ans_start = pos + 2
                ans_end = ans_start
                while ans_end < L and input_ids_flat[ans_end] != 151645:
                    ans_end += 1
                if ans_end < L:
                    labels[0, ans_start:ans_end + 2] = input_ids[0, ans_start:ans_end + 2]
                    pos = ans_end
            pos += 1

        grid_thw = full_result.get('image_grid_thw')
        position_ids, _ = get_rope_index_3(
            self.merge_size,
            input_ids,
            image_grid_thw=grid_thw,
        )

        ret = {
            'input_ids': input_ids,
            'labels': labels,
            'position_ids': position_ids,
            'attention_mask': [input_ids[0].size(0)],
            'pixel_values': full_result['pixel_values'],
            'image_grid_thw': grid_thw,
        }
        if self._anchor_token_ids is not None:
            mask_anchor_token_labels_(ret['labels'], self._anchor_token_ids)
        return ret

    def _zero_geometry(self, num_frames):
        return (
            torch.zeros(num_frames, 1, 1),
            torch.zeros(num_frames, 1, 1),
        )

    def _qwen_grid_vggt_hw(self, img: Image.Image) -> Tuple[int, int]:
        h_q, w_q = smart_resize(
            img.height, img.width, factor=32,
            min_pixels=self.data_args.min_pixels,
            max_pixels=self.data_args.max_pixels,
        )
        return h_q // 16 * 14, w_q // 16 * 14

    def _geometry_pixels(self, images: List[Image.Image]) -> torch.Tensor:
        if self.use_qwen_grid_geometry:
            transform = build_geometry_transform(self._qwen_grid_vggt_hw(images[0]))
            return torch.stack([transform(img) for img in images])
        return torch.stack([self.geometry_transform(img) for img in images])

    def _load_cached_geometry(self, cache_path, num_frames, expected_patches=None):
        if not os.path.exists(cache_path):
            return None
        geometry_embeds = torch.load(cache_path, map_location='cpu')
        camera_tokens, patch_tokens = geometry_embeds[0], geometry_embeds[1]
        if patch_tokens.size(0) != num_frames or camera_tokens.size(0) != num_frames:
            return None
        if patch_tokens.size(-1) < GEOMETRY_PATCH_DIM:
            return None
        if patch_tokens.size(-1) > GEOMETRY_PATCH_DIM:
            patch_tokens = patch_tokens[..., -GEOMETRY_PATCH_DIM:].contiguous()
        if expected_patches is not None and patch_tokens.size(1) != expected_patches:
            raise RuntimeError(
                f'{cache_path}: cached patch count {patch_tokens.size(1)} != '
                f'expected Qwen grid {expected_patches} — stale cache? '
                f'Re-run extract_vggt_features_24_32_last.py for this scene.'
            )
        return camera_tokens, patch_tokens


    def semantic_rendering_get_item(self, data_item, root):
        scene_dir = os.path.join(root, data_item['scene_dir'])
        frames_subdir = data_item.get('frames_subdir', 'rgb')
        input_frame_names = data_item['input_frames']
        target_frame_name = data_item['target_frame']

        ctx_images = [
            Image.open(os.path.join(scene_dir, frames_subdir, f)).convert('RGB')
            for f in input_frame_names
        ]
        target_image = Image.open(
            os.path.join(scene_dir, frames_subdir, target_frame_name)).convert('RGB')
        num_input_frames = len(ctx_images)
        num_all_frames = num_input_frames + 1

        conversations = deepcopy(data_item['conversations'])
        first_turn_idx = 1 if conversations[0]['value'] == 'system' else 0
        if self.use_timestamp_markers:
            fps = original_video_fps(data_item['scene_dir'])
            timestamps = [frame_stem_seconds(f, fps) for f in input_frame_names]
            special_tokens = ''.join([f'<{t:.1f} seconds><image>' for t in timestamps])
        else:
            special_tokens = '\n'.join([f'Frame-{i + 1}: <image>' for i in range(num_input_frames)])
        conversations[first_turn_idx]['value'] = conversations[first_turn_idx]['value'].replace(
            '<video>\n', special_tokens + '\n')

        ret = self._encode_conversation(conversations, ctx_images)

        ret['geometry_pixel_values'] = self._geometry_pixels(ctx_images + [target_image])
        patch_z, cam_z = self._zero_geometry(num_all_frames)
        ret['geometry_patch_embeds'] = patch_z
        ret['geometry_camera_embeds'] = cam_z

        gt_indices = np.load(os.path.join(root, data_item['semantic_index_path'])).astype(np.int64)
        assert gt_indices.size == NUM_SEG_TOKENS, (
            f'semantic GT {gt_indices.shape} ({gt_indices.size}) != NUM_SEG_TOKENS '
            f'({NUM_SEG_TOKENS}); set SEMANTIC_GRID_SIZE to match the GT grid.'
        )
        ret['gt_semantic_indices'] = torch.from_numpy(gt_indices).reshape(1, -1)
        ret['semantic_target_frame_idx'] = torch.tensor([num_input_frames], dtype=torch.long)
        ret['post_decoder_num_frames'] = torch.tensor([num_all_frames], dtype=torch.long)
        ret['question_type_id'] = _qtype_tensor(data_item)
        return ret

    def low_level_qa_get_item(self, data_item, root):
        points = data_item['points']
        num_images = len(data_item['images'])

        points_per_image = {}
        for pt in points:
            points_per_image.setdefault(pt.get('img_idx', 0), []).append(pt)

        images = []
        for img_i in range(num_images):
            image = Image.open(os.path.join(root, data_item['images'][img_i])).convert('RGB')
            img_points = points_per_image.get(img_i, [])
            if img_points:
                coords, colors = [], []
                for pt in img_points:
                    px = int(pt['norm_xy'][0] / 1000.0 * image.width)
                    py = int(pt['norm_xy'][1] / 1000.0 * image.height)
                    coords.append((max(0, min(px, image.width - 1)),
                                   max(0, min(py, image.height - 1))))
                    colors.append(ARROW_COLOR_MAP.get(pt['color'], (255, 255, 255)))
                image = draw_arrows_on_image(image, coords, colors, self._arrow_mask)
            images.append(image)

        conversations = deepcopy(data_item['conversations'])
        first_turn_idx = 1 if conversations[0]['value'] == 'system' else 0
        if '<image>' not in conversations[first_turn_idx]['value']:
            conversations[first_turn_idx]['value'] = (
                '<image>\n' * num_images + conversations[first_turn_idx]['value'])

        ret = self._encode_conversation(conversations, images)

        ret['geometry_pixel_values'] = self._geometry_pixels(images)
        patch_z, cam_z = self._zero_geometry(num_images)
        ret['geometry_patch_embeds'] = patch_z
        ret['geometry_camera_embeds'] = cam_z
        ret['gt_semantic_indices'] = torch.full((1, NUM_SEG_TOKENS), IGNORE_INDEX, dtype=torch.long)
        ret['semantic_target_frame_idx'] = torch.tensor([-1], dtype=torch.long)
        ret['post_decoder_num_frames'] = torch.tensor([num_images], dtype=torch.long)
        ret['question_type_id'] = _qtype_tensor(data_item)
        return ret

    def image_qa_get_item(self, data_item, root):
        image_field = data_item['image']
        image_paths = [image_field] if isinstance(image_field, str) else list(image_field)
        images = [Image.open(os.path.join(root, p)).convert('RGB') for p in image_paths]
        num_images = len(images)

        conversations = deepcopy(data_item['conversations'])
        first_turn_idx = 1 if conversations[0]['value'] == 'system' else 0
        if '<image>' not in conversations[first_turn_idx]['value']:
            conversations[first_turn_idx]['value'] = (
                '<image>\n' * num_images + conversations[first_turn_idx]['value'])

        ret = self._encode_conversation(conversations, images)

        ret['geometry_pixel_values'] = self._geometry_pixels(images)
        cached = None
        if self.cached_geometry_features and num_images == 1 and not self.use_qwen_grid_geometry:
            img_path = os.path.join(root, image_paths[0])
            stem = os.path.splitext(os.path.basename(img_path))[0]
            cache_path = os.path.join(
                os.path.dirname(os.path.dirname(img_path)), 'vggt_features', stem + '.pt')
            cached = self._load_cached_geometry(cache_path, num_images)
        if cached is not None:
            ret['geometry_camera_embeds'] = cached[0]
            ret['geometry_patch_embeds'] = cached[1]
        else:
            patch_z, cam_z = self._zero_geometry(num_images)
            ret['geometry_patch_embeds'] = patch_z
            ret['geometry_camera_embeds'] = cam_z

        ret['gt_semantic_indices'] = torch.full((1, NUM_SEG_TOKENS), IGNORE_INDEX, dtype=torch.long)
        ret['semantic_target_frame_idx'] = torch.tensor([-1], dtype=torch.long)
        ret['post_decoder_num_frames'] = torch.tensor([num_images], dtype=torch.long)
        ret['question_type_id'] = _qtype_tensor(data_item)
        return ret

    def video_get_item(self, data_item, root):
        video_file = data_item['video']
        frame_dir = os.path.join(
            root, video_file.split('/')[0],
            video_file.split('/')[-1].split('.')[0], 'frames')
        frames, timestamps = load_frames_for_training(frame_dir, max_num_frames=self.max_num_frame)
        num_frames = len(frames)

        conversations = deepcopy(data_item['conversations'])
        first_turn_idx = 1 if conversations[0]['value'] == 'system' else 0
        if '<video>' not in conversations[first_turn_idx]['value']:
            conversations[first_turn_idx]['value'] = conversations[first_turn_idx]['value'].replace(
                '<image>\n', '<video>\n', 1)
            if '<video>' not in conversations[first_turn_idx]['value']:
                conversations[first_turn_idx]['value'] = '<video>\n' + conversations[first_turn_idx]['value']
        if self.use_timestamp_markers:
            special_tokens = ''.join([f'<{t:.1f} seconds><image>' for t in timestamps])
        else:
            special_tokens = '\n'.join([f'Frame-{i + 1}: <image>' for i in range(num_frames)])
        conversations[first_turn_idx]['value'] = conversations[first_turn_idx]['value'].replace(
            '<video>\n', special_tokens + '\n')

        ret = self._encode_conversation(conversations, frames)

        ret['geometry_pixel_values'] = self._geometry_pixels(frames)

        scene_dir = os.path.join(
            root, video_file.split('/')[0], video_file.split('/')[-1].split('.')[0])
        cached = None
        if self.cached_geometry_features:
            expected_patches = None
            if self.use_qwen_grid_geometry:
                vh, vw = self._qwen_grid_vggt_hw(frames[0])
                expected_patches = (vh // 14) * (vw // 14)
            cached = self._load_cached_geometry(
                os.path.join(scene_dir, self.video_cache_name), num_frames,
                expected_patches=expected_patches)
        if cached is not None:
            ret['geometry_camera_embeds'] = cached[0]
            ret['geometry_patch_embeds'] = cached[1]
        else:
            patch_z, cam_z = self._zero_geometry(num_frames)
            ret['geometry_patch_embeds'] = patch_z
            ret['geometry_camera_embeds'] = cam_z

        ret['gt_semantic_indices'] = torch.full((1, NUM_SEG_TOKENS), IGNORE_INDEX, dtype=torch.long)
        ret['semantic_target_frame_idx'] = torch.tensor([-1], dtype=torch.long)
        ret['post_decoder_num_frames'] = torch.tensor([num_frames], dtype=torch.long)
        ret['question_type_id'] = _qtype_tensor(data_item)
        return ret


@dataclass
class GeometryDataCollator(DataCollatorForSupervisedDataset):

    tokenizer: transformers.PreTrainedTokenizer = None

    CONCAT_KEYS = (
        'gt_semantic_indices',
        'semantic_target_frame_idx',
        'post_decoder_num_frames',
        'question_type_id',
    )
    LIST_KEYS = (
        'geometry_pixel_values',
        'geometry_patch_embeds',
        'geometry_camera_embeds',
    )

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = super().__call__(instances)
        for key in self.CONCAT_KEYS:
            batch[key] = torch.cat([inst[key] for inst in instances], dim=0)
        for key in self.LIST_KEYS:
            batch[key] = [inst[key] for inst in instances]
        return batch


def make_geometry_data_module(processor, data_args) -> Dict[str, Any]:
    train_dataset = GeometryLazySupervisedDataset(processor, data_args=data_args)
    data_collator = GeometryDataCollator(tokenizer=processor.tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
