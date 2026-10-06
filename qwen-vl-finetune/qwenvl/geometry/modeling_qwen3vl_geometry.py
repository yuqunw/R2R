import os
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLForConditionalGeneration,
)

from .constants import NUM_SEG_TOKENS
from .cross_attention_fusion import CrossAttentionFusion
from .vggt_spatial_encoder import VGGTSpatialTower


class PostDecoderSemanticHead(nn.Module):

    def __init__(self, llm_dim: int = 1024, num_classes: int = 10, hidden_dim: int = 1024):
        super().__init__()
        self.llm_dim = llm_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.LayerNorm(llm_dim),
            nn.Linear(llm_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.reinit_last_layer()

    def reinit_last_layer(self):
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, seg_hidden: torch.Tensor) -> torch.Tensor:
        return self.mlp(seg_hidden)


class Qwen3VLGeometryForConditionalGeneration(Qwen3VLForConditionalGeneration):

    def __init__(self, config):
        super().__init__(config)
        text_hidden = config.text_config.hidden_size
        geometry_hidden = getattr(config, 'geometry_hidden_size', 2048)

        if getattr(config, 'geometry_tower', None) is not None:
            self.geometry_tower = VGGTSpatialTower(
                weights_path=getattr(config, 'geometry_tower_weights', 'facebook/VGGT-1B'),
                output_point=getattr(config, 'output_point', False),
                output_depth=getattr(config, 'output_depth', False),
                output_camera=getattr(config, 'output_camera', False),
                delay_load=True,
            )
        else:
            self.geometry_tower = None

        fusion_type = getattr(config, 'fusion_block', None)
        if fusion_type == 'cross_attention':
            self.fusion_block = CrossAttentionFusion(
                vision_dim=text_hidden,
                geometry_dim=geometry_hidden,
                attn_dim=text_hidden,
                num_heads=8,
            )
        elif fusion_type is None:
            self.fusion_block = None
        else:
            raise ValueError(
                f"fusion_block={fusion_type!r} not supported in the Qwen3-VL port "
                f"(only 'cross_attention')."
            )

        self.predict_semantic_rendering = getattr(config, 'predict_semantic_rendering', False)
        if self.predict_semantic_rendering:
            self.post_decoder_semantic_head = PostDecoderSemanticHead(
                llm_dim=text_hidden,
                num_classes=getattr(config, 'semantic_num_classes', 10),
            )
        else:
            self.post_decoder_semantic_head = None
        self.post_decoder_semantic_loss_weight = getattr(config, 'post_decoder_semantic_loss_weight', 1.0)

        self.use_semantic_ce_weight = getattr(config, 'use_semantic_ce_weight', True)
        if self.predict_semantic_rendering and self.use_semantic_ce_weight:
            default_w = torch.tensor(
                [0.0, 1.4882, 1.0, 2.7601, 1.9326, 2.5506, 1.9023, 1.8669, 2.1964, 1.0],
                dtype=torch.float32,
            )
            self.register_buffer('semantic_ce_weight_buf', default_w, persistent=False)
        else:
            self.semantic_ce_weight_buf = None

        if self.predict_semantic_rendering and self.fusion_block is not None:
            self.cam_mapping = nn.Linear(geometry_hidden, text_hidden)
        else:
            self.cam_mapping = None

        nts = getattr(config, 'new_token_id_start', None)
        nte = getattr(config, 'new_token_id_end', None)
        if nts is not None and nte is not None:
            self.new_token_embeddings = nn.Parameter(torch.zeros(int(nte) - int(nts) + 1, text_hidden))
            self.new_token_id_start = int(nts)
            self.new_token_id_end = int(nte)
        else:
            self.new_token_embeddings = None
            self.new_token_id_start = None
            self.new_token_id_end = None

        self.loss_reduction = getattr(config, 'loss_reduction', 'square')
        if self.loss_reduction != 'square':
            raise NotImplementedError(
                f"loss_reduction={self.loss_reduction!r} is not supported in the "
                f"Qwen3-VL port (only 'square')."
            )

    def _dump_debug_batch(self, tag, **tensors):
        count = getattr(self, '_debug_dump_count', 0)
        if count >= 2:
            return None
        self._debug_dump_count = count + 1
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        out_dir = os.environ.get('GEOM_DEBUG_DIR', 'nan_debug')
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f'nan_batch_rank{rank}_{tag}_{count}.pt')
        payload = {}
        for k, v in tensors.items():
            if torch.is_tensor(v):
                payload[k] = v.detach().to('cpu')
            elif isinstance(v, (list, tuple)) and v and all(torch.is_tensor(t) for t in v):
                payload[k] = [t.detach().to('cpu') for t in v]
            elif v is not None:
                payload[k] = v
        torch.save(payload, path)
        print(f'[NAN-DUMP] saved offending micro-batch to {path}', flush=True)
        return path

    def reinit_new_module_weights(self):
        if self.fusion_block is not None:
            self.fusion_block._init_weights()
        if self.post_decoder_semantic_head is not None:
            self.post_decoder_semantic_head.reinit_last_layer()

    def setup_new_token_embeddings(self, id_start: int, id_end: int):
        assert id_end >= id_start, f'invalid range [{id_start}, {id_end}]'
        base_emb = self.get_input_embeddings().weight
        init_rows = base_emb.data[id_start:id_end + 1].detach().clone()
        self.new_token_embeddings = nn.Parameter(init_rows.contiguous())
        self.new_token_id_start = int(id_start)
        self.new_token_id_end = int(id_end)

    def _apply_new_token_embeddings(
        self, input_embeds: torch.Tensor, input_ids: torch.LongTensor
    ) -> torch.Tensor:
        if self.new_token_embeddings is None:
            return input_embeds

        orig_shape = input_embeds.shape
        flat_embeds = input_embeds.reshape(-1, orig_shape[-1])
        flat_ids = input_ids.reshape(-1)

        mask = (flat_ids >= self.new_token_id_start) & (flat_ids <= self.new_token_id_end)
        if not mask.any():
            flat_embeds = flat_embeds + self.new_token_embeddings.sum() * 0.0
            return flat_embeds.reshape(orig_shape)

        offsets = (flat_ids[mask] - self.new_token_id_start).long()
        new_rows = self.new_token_embeddings[offsets].to(flat_embeds.dtype)
        flat_embeds[mask] = flat_embeds[mask] * 0.0 + new_rows
        return flat_embeds.reshape(orig_shape)

    def extract_geometry_feature(self, pixel_values):
        if self.geometry_tower is None:
            return None
        with torch.no_grad():
            return self.geometry_tower(pixel_values)

    def _fuse_vision_with_geometry(self, image_embeds_list, geometry_camera_embeds, geometry_patch_embeds):
        n = len(image_embeds_list)
        assert n == geometry_camera_embeds.shape[0] == geometry_patch_embeds.shape[0], (
            f'#images ({n}) != #geometry context frames '
            f'({geometry_camera_embeds.shape[0]}). Check sample ordering.'
        )
        lens = [e.shape[0] for e in image_embeds_list]
        fused = [None] * n
        dtype = image_embeds_list[0].dtype
        for L in sorted(set(lens)):
            idxs = [i for i, l in enumerate(lens) if l == L]
            q = torch.stack([image_embeds_list[i] for i in idxs], dim=0)
            idx_t = torch.tensor(idxs, dtype=torch.long, device=q.device)
            cam = geometry_camera_embeds[idx_t].to(dtype)
            patch = geometry_patch_embeds[idx_t].to(dtype)
            out, _ = self.fusion_block(q, cam, patch)
            for j, i in enumerate(idxs):
                fused[i] = out[j]
        return torch.cat(fused, dim=0)

    def _square_lm_loss(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        all_ignored = bool((shift_labels == -100).all())
        if all_ignored:
            shift_labels = shift_labels * 0
        loss = CrossEntropyLoss(reduction='none')(
            shift_logits.view(-1, logits.shape[-1]),
            shift_labels.view(-1).to(shift_logits.device))
        loss_weight = (labels != -100).sum(dim=-1).float()
        loss_weight = 1 / loss_weight.sqrt()
        loss_weight = torch.where(labels != -100, loss_weight.unsqueeze(1), 0.0)
        shift_weights = loss_weight[..., 1:].contiguous().view(-1).to(shift_logits.device)
        shift_weights_sum = shift_weights.sum()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(shift_weights_sum, op=torch.distributed.ReduceOp.AVG)
        loss = (loss * shift_weights).sum() / shift_weights_sum
        if all_ignored or not torch.isfinite(loss):
            loss = sum(p.sum() for p in self.parameters() if p.requires_grad) * 0.0
        return loss

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        geometry_pixel_values: Optional[torch.FloatTensor] = None,
        geometry_patch_embeds: Optional[torch.FloatTensor] = None,
        geometry_camera_embeds: Optional[torch.FloatTensor] = None,
        gt_semantic_indices: Optional[torch.LongTensor] = None,
        semantic_target_frame_idx: Optional[torch.LongTensor] = None,
        post_decoder_num_frames: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[tuple, Qwen3VLCausalLMOutputWithPast]:
        if post_decoder_num_frames is None or self.fusion_block is None:
            out = super().forward(
                input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=past_key_values,
                inputs_embeds=inputs_embeds, labels=None,
                pixel_values=pixel_values, pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
                cache_position=cache_position, logits_to_keep=logits_to_keep,
                **kwargs,
            )
            if labels is not None:
                out = Qwen3VLCausalLMOutputWithPast(
                    loss=self._square_lm_loss(out.logits, labels),
                    logits=out.logits,
                    past_key_values=out.past_key_values,
                )
            return out

        if (
            pixel_values is None
            and past_key_values is not None
            and past_key_values.get_seq_length() > 0
        ):
            inputs_embeds = self.get_input_embeddings()(input_ids).clone()
            inputs_embeds = self._apply_new_token_embeddings(inputs_embeds, input_ids)
            if position_ids is None:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
            outputs = self.model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=cache_position,
            )
            hidden_states = outputs.last_hidden_state
            slice_indices = (
                slice(-logits_to_keep, None)
                if isinstance(logits_to_keep, int) and logits_to_keep > 0
                else slice(None)
            )
            logits = self.lm_head(hidden_states[:, slice_indices, :])
            return Qwen3VLCausalLMOutputWithPast(
                logits=logits,
                past_key_values=outputs.past_key_values,
            )

        assert pixel_values_videos is None, (
            'The geometry pipeline feeds video frames as individual images; '
            'pixel_values_videos is not supported.'
        )

        if os.environ.get('GEOM_DEBUG_MEM') == '1' and torch.cuda.is_available():
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            print(
                f'[mem r{rank}] alloc={torch.cuda.memory_allocated()/2**30:.1f}G '
                f'reserved={torch.cuda.memory_reserved()/2**30:.1f}G '
                f'peak={torch.cuda.max_memory_allocated()/2**30:.1f}G '
                f'seq={tuple(input_ids.shape)} frames={int(post_decoder_num_frames.sum())}',
                flush=True,
            )

        device = input_ids.device
        bs = post_decoder_num_frames.view(-1).shape[0]

        nf = post_decoder_num_frames.view(-1).cpu().tolist()
        frame_starts = [0]
        for F_b in nf:
            frame_starts.append(frame_starts[-1] + F_b)

        if semantic_target_frame_idx is not None:
            t_idx = semantic_target_frame_idx.view(-1).cpu().tolist()
        else:
            t_idx = [-1] * bs

        expected_patch_dim = self.fusion_block.patch_mapping[0].normalized_shape[0]
        model_dtype = self.get_input_embeddings().weight.dtype

        def _per_sample(x):
            if x is None:
                return [None] * bs
            if torch.is_tensor(x):
                return [x[frame_starts[b]:frame_starts[b + 1]] for b in range(bs)]
            return list(x)

        geometry_frames_list = _per_sample(geometry_pixel_values)
        patch_list = _per_sample(geometry_patch_embeds)
        cam_list = _per_sample(geometry_camera_embeds)

        per_sample_patch_nt = []
        per_sample_cam_nt = []
        target_cam_list = []
        for b in range(bs):
            F_b = frame_starts[b + 1] - frame_starts[b]
            t_b = t_idx[b]

            sample_patch = patch_list[b]
            sample_cam = cam_list[b]

            sample_needs_regen = (
                sample_patch is None
                or sample_cam is None
                or sample_patch.shape[-1] != expected_patch_dim
                or bool((sample_patch == 0).all().item())
                or not bool(torch.isfinite(sample_patch).all().item())
                or not bool(torch.isfinite(sample_cam).all().item())
            )
            if sample_needs_regen:
                views = geometry_frames_list[b].to(
                    device=device, dtype=model_dtype).unsqueeze(0)
                temp_cam, temp_patch, *_ = self.extract_geometry_feature(views)
                if not torch.isfinite(temp_patch).all() or not torch.isfinite(temp_cam).all():
                    print(f'[WARN vggt-regen non-finite] sample {b}: frames={end - start} '
                          f'target={t_idx[b]} patch_finite={bool(torch.isfinite(temp_patch).all())} '
                          f'cam_finite={bool(torch.isfinite(temp_cam).all())}', flush=True)
                sample_patch = temp_patch
                sample_cam = temp_cam
            sample_patch = sample_patch.to(device=device, dtype=model_dtype)
            sample_cam = sample_cam.to(device=device, dtype=model_dtype)

            if t_b >= 0:
                target_cam_list.append(sample_cam[t_b:t_b + 1].clone())
                keep = [f for f in range(F_b) if f != t_b]
                keep_t = torch.tensor(keep, dtype=torch.long, device=sample_patch.device)
                sample_patch = sample_patch[keep_t]
                sample_cam = sample_cam[keep_t]

            per_sample_patch_nt.append(sample_patch)
            per_sample_cam_nt.append(sample_cam)

        patch_shapes = {tuple(t.shape[1:]) for t in per_sample_patch_nt}
        assert len(patch_shapes) == 1, (
            f'Samples with different geometry patch grids in one batch: '
            f'{sorted(patch_shapes)}. Batch same-grid samples together or '
            f'use per_device_train_batch_size 1.'
        )
        geometry_patch_nt = torch.cat(per_sample_patch_nt, dim=0)
        geometry_cam_nt = torch.cat(per_sample_cam_nt, dim=0)
        target_cam_embeds = torch.cat(target_cam_list, dim=0) if target_cam_list else None

        image_embeds_list, deepstack_image_embeds = self.model.get_image_features(
            pixel_values, image_grid_thw
        )
        image_embeds = self._fuse_vision_with_geometry(
            list(image_embeds_list), geometry_cam_nt, geometry_patch_nt
        )

        _nan_probes = os.environ.get('GEOM_NAN_PROBES') == '1'
        _nan_stage = None
        if _nan_probes:
            if not torch.isfinite(geometry_patch_nt).all() or not torch.isfinite(geometry_cam_nt).all():
                _nan_stage = 'geometry_features'
            elif not torch.isfinite(torch.cat(list(image_embeds_list), dim=0)).all():
                _nan_stage = 'vision_tower_output'
            elif not torch.isfinite(image_embeds).all():
                _nan_stage = 'fusion_output'
        if _nan_stage is not None:
            print(f'[NAN-PROBE] first non-finite at stage={_nan_stage} '
                  f'frames={post_decoder_num_frames.view(-1).tolist()}', flush=True)
            self._dump_debug_batch(
                f'probe_{_nan_stage}',
                input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, labels=labels,
                pixel_values=pixel_values, image_grid_thw=image_grid_thw,
                geometry_pixel_values=geometry_pixel_values,
                geometry_patch_embeds=geometry_patch_embeds,
                geometry_camera_embeds=geometry_camera_embeds,
                gt_semantic_indices=gt_semantic_indices,
                semantic_target_frame_idx=semantic_target_frame_idx,
                post_decoder_num_frames=post_decoder_num_frames,
            )

        inputs_embeds = self.get_input_embeddings()(input_ids).clone()
        inputs_embeds = self._apply_new_token_embeddings(inputs_embeds, input_ids)

        ignore = False
        try:
            image_mask, _ = self.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds.to(inputs_embeds.dtype))
        except ValueError as e:
            if labels is None:
                raise
            print(f'[WARN truncated sample] vision splice failed ({e}) — '
                  f'zeroing this micro-batch loss', flush=True)
            ignore = True
            image_mask = None

        pose_embeds = None
        if target_cam_embeds is not None:
            if self.cam_mapping is None:
                raise RuntimeError('target_cam_embeds present but cam_mapping is None.')
            pose_embeds = self.cam_mapping(target_cam_embeds.to(model_dtype)).squeeze(1)

        B, N, C = inputs_embeds.shape
        flat_embeds = inputs_embeds.reshape(B * N, C)
        flat_ids = input_ids.reshape(B * N)
        pose_replaced = False
        if pose_embeds is not None:
            pose_token_id = getattr(self.config, 'pose_token_id', None)
            assert pose_token_id is not None, 'config.pose_token_id not set'
            pose_mask = (flat_ids == pose_token_id)
            n_pose = int(pose_mask.sum().item())
            if n_pose > 0:
                assert n_pose == pose_embeds.shape[0], (
                    f'<POSE> count ({n_pose}) != target_cam_embeds count '
                    f'({pose_embeds.shape[0]}). Check sample ordering.'
                )
                flat_embeds[pose_mask] = flat_embeds[pose_mask] * 0.0 + pose_embeds.to(flat_embeds.dtype)
                pose_replaced = True
        inputs_embeds = flat_embeds.reshape(B, N, C)

        if self.cam_mapping is not None and not pose_replaced:
            cam_phantom = sum(
                p.sum() for p in self.cam_mapping.parameters() if p.requires_grad
            ) * 0.0
            inputs_embeds = inputs_embeds + cam_phantom

        visual_pos_masks = image_mask[..., 0] if image_mask is not None else None

        if position_ids is None:
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw,
                attention_mask=attention_mask,
            )
            self.model.rope_deltas = rope_deltas

        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=kwargs.get('use_cache', False),
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_image_embeds if image_mask is not None else None,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int) and logits_to_keep > 0
            else slice(None)
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        if _nan_probes and _nan_stage is None:
            if not torch.isfinite(hidden_states).all():
                _nan_stage = 'decoder_hidden_states'
            elif not torch.isfinite(logits).all():
                _nan_stage = 'lm_head_logits'
            if _nan_stage is not None:
                print(f'[NAN-PROBE] first non-finite at stage={_nan_stage} '
                      f'seq={tuple(input_ids.shape)} '
                      f'frames={post_decoder_num_frames.view(-1).tolist()}', flush=True)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            if (shift_labels == -100).all():
                ignore = True
                shift_labels = shift_labels * 0

            loss_fct = CrossEntropyLoss(reduction='none')
            shift_logits = shift_logits.view(-1, logits.shape[-1])
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            loss_weight = (labels != -100).sum(dim=-1).float()
            loss_weight = 1 / loss_weight.sqrt()
            loss_weight = torch.where(labels != -100, loss_weight.unsqueeze(1), 0.0)

            shift_weights = loss_weight[..., 1:].contiguous().view(-1).to(shift_logits.device)
            shift_weights_sum = shift_weights.sum()
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(shift_weights_sum, op=torch.distributed.ReduceOp.AVG)

            loss = loss * shift_weights
            loss = loss.sum() / shift_weights_sum

        self._last_post_decoder_semantic_loss = None
        self._last_post_decoder_semantic_accuracy = None
        self._last_post_decoder_semantic_miou = None
        semantic_loss_added = False
        need_post_decoder_semantic = (
            self.post_decoder_semantic_head is not None and gt_semantic_indices is not None
        )
        if need_post_decoder_semantic and labels is not None:
            post_semantic_loss = self._compute_post_decoder_semantic_loss(
                hidden_states, input_ids, gt_semantic_indices
            )
            if post_semantic_loss is not None:
                self._last_post_decoder_semantic_loss = post_semantic_loss.detach()
                weighted = self.post_decoder_semantic_loss_weight * post_semantic_loss
                loss = (loss + weighted) if loss is not None else weighted
                semantic_loss_added = True

        if self.post_decoder_semantic_head is not None and not semantic_loss_added:
            dummy = sum(p.sum() for p in self.post_decoder_semantic_head.parameters()) * 0.0
            loss = (loss + dummy) if loss is not None else dummy

        if ignore and loss is not None:
            print('[Debug] ignore curr loss')
            loss = loss * 0.0 + sum(
                p.sum() for p in self.parameters() if p.requires_grad
            ) * 0.0

        if loss is not None and not torch.isfinite(loss):
            print(
                f'[WARN non-finite loss] seq={tuple(input_ids.shape)} '
                f'frames={post_decoder_num_frames.view(-1).tolist()} '
                f'target={semantic_target_frame_idx.view(-1).tolist() if semantic_target_frame_idx is not None else None} '
                f'first_ids={input_ids[0, :8].tolist()} — replacing with phantom loss',
                flush=True,
            )
            self._dump_debug_batch(
                'nonfinite_loss',
                input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, labels=labels,
                pixel_values=pixel_values, image_grid_thw=image_grid_thw,
                geometry_pixel_values=geometry_pixel_values,
                geometry_patch_embeds=geometry_patch_embeds,
                geometry_camera_embeds=geometry_camera_embeds,
                gt_semantic_indices=gt_semantic_indices,
                semantic_target_frame_idx=semantic_target_frame_idx,
                post_decoder_num_frames=post_decoder_num_frames,
            )
            loss = sum(p.sum() for p in self.parameters() if p.requires_grad) * 0.0

        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )

    def _compute_post_decoder_semantic_loss(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.LongTensor,
        gt_semantic_indices: torch.LongTensor,
    ) -> Optional[torch.Tensor]:
        semantic_start_id = getattr(self.config, 'semantic_start_token_id', None)
        assert semantic_start_id is not None, 'config.semantic_start_token_id not set'

        seg_hidden_list = []
        gt_indices_list = []
        num_logical = gt_semantic_indices.shape[0]
        for b in range(min(num_logical, input_ids.shape[0])):
            sub_seq = input_ids[b]
            start_positions = (sub_seq == semantic_start_id).nonzero(as_tuple=False)
            if start_positions.numel() < 2:
                continue

            seg_start = start_positions[-1, 0].item() + 1
            seg_end = seg_start + NUM_SEG_TOKENS
            if seg_end > sub_seq.shape[0]:
                continue

            seg_hidden_list.append(hidden_states[b, seg_start:seg_end, :])
            gt_indices_list.append(gt_semantic_indices[b])

        if len(seg_hidden_list) == 0:
            dummy = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
            for p in self.post_decoder_semantic_head.parameters():
                if p.requires_grad:
                    dummy = dummy + p.sum() * 0.0
            return dummy

        seg_hidden = torch.stack(seg_hidden_list, dim=0)
        gt_indices = torch.stack(gt_indices_list, dim=0)

        logits = self.post_decoder_semantic_head(seg_hidden)

        if self.semantic_ce_weight_buf is not None:
            ce_weight = self.semantic_ce_weight_buf.to(device=logits.device, dtype=logits.dtype)
            loss_fct = CrossEntropyLoss(ignore_index=0, weight=ce_weight)
        else:
            loss_fct = CrossEntropyLoss(ignore_index=0)
        loss = loss_fct(
            logits.reshape(-1, logits.size(-1)),
            gt_indices.reshape(-1).to(logits.device),
        )

        with torch.no_grad():
            flat_preds = logits.reshape(-1, logits.size(-1)).argmax(dim=-1)
            flat_gt = gt_indices.reshape(-1).to(logits.device)
            valid = flat_gt != 0
            if valid.any():
                preds_v = flat_preds[valid]
                gt_v = flat_gt[valid]
                self._last_post_decoder_semantic_accuracy = (preds_v == gt_v).float().mean().detach()
                ious = []
                for c in range(1, logits.size(-1)):
                    pred_c = preds_v == c
                    gt_c = gt_v == c
                    union = (pred_c | gt_c).sum().float()
                    if union > 0:
                        ious.append(((pred_c & gt_c).sum().float() / union).item())
                self._last_post_decoder_semantic_miou = torch.tensor(
                    sum(ious) / len(ious) if ious else 0.0, device=logits.device
                )

        if not torch.isfinite(loss):
            print(f'[WARN semantic] non-finite loss={loss.item()}, skipping')
            return None
        return loss
