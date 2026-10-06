import logging
import os
import sys

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _ensure_vggt_importable():
    default = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', '..', 'third_party', 'vggt'))
    vggt_path = os.environ.get('VGGT_REPO_PATH', default)
    if vggt_path not in sys.path:
        sys.path.append(vggt_path)


def prepare_input(pixel_values):
    original_shape = pixel_values.shape
    if len(original_shape) == 3:
        pixel_values = pixel_values.unsqueeze(0)
        batch_size, num_frames = 1, 1
    elif len(original_shape) == 4:
        batch_size = 1
        num_frames = original_shape[0]
    elif len(original_shape) == 5:
        batch_size, num_frames = original_shape[0], original_shape[1]
        pixel_values = pixel_values.reshape(batch_size * num_frames, *original_shape[2:])
    else:
        raise ValueError(f"Unexpected input shape: {original_shape}. Expected 3D, 4D, or 5D tensor.")

    mean = torch.tensor([0.485, 0.456, 0.406],
                        device=pixel_values.device,
                        dtype=pixel_values.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225],
                       device=pixel_values.device,
                       dtype=pixel_values.dtype).view(1, 3, 1, 1)

    pixel_values = (pixel_values * std + mean).clamp(0, 1)

    C, H, W = pixel_values.shape[-3:]
    pixel_values = pixel_values.reshape(batch_size, num_frames, C, H, W)
    return pixel_values


class VGGT_Encoder(nn.Module):
    def __init__(self, weights_path='facebook/VGGT-1B',
                 output_point=False, output_depth=False, output_camera=False):
        super().__init__()
        _ensure_vggt_importable()
        from vggt.models.vggt import VGGT
        logger.info(f"Loading VGGT from: {weights_path}")
        self.vggt = VGGT.from_pretrained(weights_path)
        self.vggt.eval()
        self.output_point = output_point
        self.output_depth = output_depth
        self.output_camera = output_camera
        for param in self.vggt.parameters():
            param.requires_grad = False
        logger.info("VGGT loaded successfully, all parameters frozen")

    def forward(self, pixel_values):
        views = prepare_input(pixel_values=pixel_values)

        MULTI_LAYER_INDICES = [int(x) for x in os.environ.get('VGGT_MULTI_LAYER_INDICES', '4,11,17,23').split(',')]

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=views.dtype):
                aggregated_tokens_list, ps_idx = self.vggt.aggregator(views)

        aggregated_tokens_list = [i.to(pixel_values.dtype) for i in aggregated_tokens_list]

        spatial_feat_last = aggregated_tokens_list[-1]
        camera_token = spatial_feat_last[:, :, 0:1, :].flatten(0, 1)

        multi_layer_patches = []
        for layer_idx in MULTI_LAYER_INDICES:
            layer_feat = aggregated_tokens_list[layer_idx]
            patches = layer_feat[:, :, ps_idx:, :].flatten(0, 1)
            multi_layer_patches.append(patches)
        patch_tokens = torch.cat(multi_layer_patches, dim=-1)
        output_list = [camera_token, patch_tokens]

        if self.output_depth:
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=views.dtype):
                    vggt_depth, vggt_depth_conf = self.vggt.depth_head(
                        aggregated_tokens_list, images=views, patch_start_idx=ps_idx
                    )
            output_list.append(vggt_depth.to(pixel_values.dtype))
            output_list.append(vggt_depth_conf.to(pixel_values.dtype))
        else:
            output_list = output_list + [None, None]

        if self.output_point:
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=views.dtype):
                    pts3d_pred, pts3d_conf = self.vggt.point_head(
                        aggregated_tokens_list, images=views, patch_start_idx=ps_idx
                    )
            output_list.append(pts3d_pred)
            output_list.append(pts3d_conf)
        else:
            output_list = output_list + [None, None]

        if self.output_camera:
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=views.dtype):
                    pose_enc_list = self.vggt.camera_head(aggregated_tokens_list)
            pose_enc_list = [p.to(pixel_values.dtype) for p in pose_enc_list]
            output_list.append(pose_enc_list)
        else:
            output_list = output_list + [None]

        return tuple(output_list)


class VGGTSpatialTower(nn.Module):

    def __init__(self, weights_path='facebook/VGGT-1B',
                 output_point=False, output_depth=False, output_camera=False,
                 delay_load=True):
        super().__init__()
        self.is_loaded = False
        self.weights_path = weights_path
        self.output_point = output_point
        self.output_depth = output_depth
        self.output_camera = output_camera
        if not delay_load:
            self.load_model()

    def load_model(self):
        if self.is_loaded:
            logger.info(f"{self.weights_path} is already loaded, skipping.")
            return
        self.spatial_tower = VGGT_Encoder(
            weights_path=self.weights_path,
            output_point=self.output_point,
            output_depth=self.output_depth,
            output_camera=self.output_camera,
        )
        self.spatial_tower.eval()
        self.spatial_tower.requires_grad_(False)
        logger.info("VGGT spatial tower loaded, set to eval mode, and frozen")
        self.is_loaded = True

    def forward(self, pixel_values):
        return self.spatial_tower(pixel_values)

    @property
    def dtype(self):
        for p in self.spatial_tower.parameters():
            return p.dtype

    @property
    def device(self):
        for p in self.spatial_tower.parameters():
            return p.device
