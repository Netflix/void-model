import os
import sys
import glob
import json
import gc
import imageio
from loguru import logger
import inspect
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import cv2
from einops import rearrange, repeat
from PIL import Image
import mediapy as media
import skimage
import matplotlib

from videox_fun.data.dataset_image_video import get_random_mask


def filter_kwargs(cls, kwargs):
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs

def get_width_and_height_from_image_and_base_resolution(image, base_resolution):
    target_pixels = int(base_resolution) * int(base_resolution)
    original_width, original_height = Image.open(image).size
    ratio = (target_pixels / (original_width * original_height)) ** 0.5
    width_slider = round(original_width * ratio)
    height_slider = round(original_height * ratio)
    return height_slider, width_slider

def color_transfer(sc, dc):
    """
    Transfer color distribution from of sc, referred to dc.

    Args:
        sc (numpy.ndarray): input image to be transfered.
        dc (numpy.ndarray): reference image

    Returns:
        numpy.ndarray: Transferred color distribution on the sc.
    """

    def get_mean_and_std(img):
        x_mean, x_std = cv2.meanStdDev(img)
        x_mean = np.hstack(np.around(x_mean, 2))
        x_std = np.hstack(np.around(x_std, 2))
        return x_mean, x_std

    sc = cv2.cvtColor(sc, cv2.COLOR_RGB2LAB)
    s_mean, s_std = get_mean_and_std(sc)
    dc = cv2.cvtColor(dc, cv2.COLOR_RGB2LAB)
    t_mean, t_std = get_mean_and_std(dc)
    img_n = ((sc - s_mean) * (t_std / s_std)) + t_mean
    np.putmask(img_n, img_n > 255, 255)
    np.putmask(img_n, img_n < 0, 0)
    dst = cv2.cvtColor(cv2.convertScaleAbs(img_n), cv2.COLOR_LAB2RGB)
    return dst

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=12, imageio_backend=True, color_transfer_post_process=False):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).cpu().float().numpy().astype(np.uint8)
        outputs.append(Image.fromarray(x))

    if color_transfer_post_process:
        for i in range(1, len(outputs)):
            outputs[i] = Image.fromarray(color_transfer(np.uint8(outputs[i]), np.uint8(outputs[0])))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if imageio_backend:
        if path.endswith("mp4"):
            imageio.mimsave(path, outputs, fps=fps)
        else:
            imageio.mimsave(path, outputs, duration=(1000 * 1/fps))
    else:
        if path.endswith("mp4"):
            path = path.replace('.mp4', '.gif')
        outputs[0].save(
            path, format='GIF', append_images=outputs[1:], save_all=True,
            duration=round(1000 / fps), loop=0,
        )


def save_inout_row(input_video, input_mask, output_video, video_path, fps=16, visualize_masked_video=False, visualize_error=True):
    input_video = rearrange(input_video[0], "c t h w -> t h w c")
    input_mask = rearrange(input_mask[0], "c t h w -> t h w c")
    input_mask = repeat(input_mask, "t h w c -> t h w (repeat c)", repeat=3)
    input_mask = 1 - input_mask
    output_video = rearrange(output_video[0], "c t h w -> t h w c")
    min_len = min(len(input_video), len(output_video), len(input_mask))
    input_video = input_video[:min_len]
    input_mask = input_mask[:min_len]
    output_video = output_video[:min_len]

    row = [input_video.cpu().float().numpy(), input_mask.cpu().float().numpy(),]
    if visualize_masked_video:
        row += [(input_mask * input_video).cpu().float().numpy()]
    row += [output_video.cpu().float().numpy()]

    if visualize_error:
        err = torch.abs(input_video - output_video).mean(-1).cpu().float().numpy()
        vis_err = apply_colormap(err)
        row += [vis_err]

    row = np.concatenate(row, 2)
    media.write_video(video_path, row, fps=fps)


def get_image_to_video_latent(validation_image_start, validation_image_end, video_length, sample_size):
    if validation_image_start is not None and validation_image_end is not None:
        if type(validation_image_start) is str and os.path.isfile(validation_image_start):
            image_start = clip_image = Image.open(validation_image_start).convert("RGB")
            image_start = image_start.resize([sample_size[1], sample_size[0]])
            clip_image = clip_image.resize([sample_size[1], sample_size[0]])
        else:
            image_start = clip_image = validation_image_start
            image_start = [_image_start.resize([sample_size[1], sample_size[0]]) for _image_start in image_start]
            clip_image = [_clip_image.resize([sample_size[1], sample_size[0]]) for _clip_image in clip_image]

        if type(validation_image_end) is str and os.path.isfile(validation_image_end):
            image_end = Image.open(validation_image_end).convert("RGB")
            image_end = image_end.resize([sample_size[1], sample_size[0]])
        else:
            image_end = validation_image_end
            image_end = [_image_end.resize([sample_size[1], sample_size[0]]) for _image_end in image_end]

        if type(image_start) is list:
            clip_image = clip_image[0]
            start_video = torch.cat(
                [torch.from_numpy(np.array(_image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0) for _image_start in image_start], 
                dim=2
            )
            input_video = torch.tile(start_video[:, :, :1], [1, 1, video_length, 1, 1])
            input_video[:, :, :len(image_start)] = start_video
            
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, len(image_start):] = 255
        else:
            input_video = torch.tile(
                torch.from_numpy(np.array(image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0), 
                [1, 1, video_length, 1, 1]
            )
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, 1:] = 255

        if type(image_end) is list:
            image_end = [_image_end.resize(image_start[0].size if type(image_start) is list else image_start.size) for _image_end in image_end]
            end_video = torch.cat(
                [torch.from_numpy(np.array(_image_end)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0) for _image_end in image_end], 
                dim=2
            )
            input_video[:, :, -len(end_video):] = end_video
            
            input_video_mask[:, :, -len(image_end):] = 0
        else:
            image_end = image_end.resize(image_start[0].size if type(image_start) is list else image_start.size)
            input_video[:, :, -1:] = torch.from_numpy(np.array(image_end)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0)
            input_video_mask[:, :, -1:] = 0

        input_video = input_video / 255

    elif validation_image_start is not None:
        if type(validation_image_start) is str and os.path.isfile(validation_image_start):
            image_start = clip_image = Image.open(validation_image_start).convert("RGB")
            image_start = image_start.resize([sample_size[1], sample_size[0]])
            clip_image = clip_image.resize([sample_size[1], sample_size[0]])
        else:
            image_start = clip_image = validation_image_start
            image_start = [_image_start.resize([sample_size[1], sample_size[0]]) for _image_start in image_start]
            clip_image = [_clip_image.resize([sample_size[1], sample_size[0]]) for _clip_image in clip_image]
        image_end = None
        
        if type(image_start) is list:
            clip_image = clip_image[0]
            start_video = torch.cat(
                [torch.from_numpy(np.array(_image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0) for _image_start in image_start], 
                dim=2
            )
            input_video = torch.tile(start_video[:, :, :1], [1, 1, video_length, 1, 1])
            input_video[:, :, :len(image_start)] = start_video
            input_video = input_video / 255
            
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, len(image_start):] = 255
        else:
            input_video = torch.tile(
                torch.from_numpy(np.array(image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0), 
                [1, 1, video_length, 1, 1]
            ) / 255
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, 1:, ] = 255
    else:
        image_start = None
        image_end = None
        input_video = torch.zeros([1, 3, video_length, sample_size[0], sample_size[1]])
        input_video_mask = torch.ones([1, 1, video_length, sample_size[0], sample_size[1]]) * 255
        clip_image = None

    del image_start
    del image_end
    gc.collect()

    return  input_video, input_video_mask, clip_image

def get_video_to_video_latent(input_video_path, video_length, sample_size, fps=None, validation_video_mask=None, ref_image=None):
    if isinstance(input_video_path, str):
        input_video = media.read_video(input_video_path)
    else:
        input_video, input_video_mask = None, None

    input_video = torch.from_numpy(np.array(input_video))[:video_length]
    input_video = input_video.permute([3, 0, 1, 2]).float() / 255  # (c, t, h, w)
    input_video = F.interpolate(input_video, sample_size, mode='area').unsqueeze(0)  # (1, c, t, h, w)

    if validation_video_mask is not None:
        if (
            validation_video_mask.endswith(".jpg") or
            validation_video_mask.endswith(".jpeg") or
            validation_video_mask.endswith(".png")
            ):
            validation_video_mask = Image.open(validation_video_mask).convert('L').resize((sample_size[1], sample_size[0]))
            input_video_mask = np.where(np.array(validation_video_mask) < 240, 0, 255)
            
            input_video_mask = torch.from_numpy(np.array(input_video_mask)).unsqueeze(0).unsqueeze(-1).permute([3, 0, 1, 2]).unsqueeze(0)
            input_video_mask = torch.tile(input_video_mask, [1, 1, input_video.size()[2], 1, 1])
            input_video_mask = input_video_mask.to(input_video.device, input_video.dtype)
        elif validation_video_mask.endswith(".mp4"):
            validation_video_mask = media.read_video(validation_video_mask)[:video_length]
            if len(validation_video_mask.shape) == 4:  # (t, h, w, c)
                validation_video_mask = validation_video_mask[..., 0] # (t, h, w)
            input_video_mask = torch.from_numpy(validation_video_mask).unsqueeze(0)  # (1, t, h, w)
            input_video_mask = F.interpolate(input_video_mask.float(), sample_size, mode='area')
            input_video_mask = torch.where(input_video_mask < 240, 0, 255).unsqueeze(0)  # (1, 1, t, h, w)

            input_video_mask = dilate_video_mask(input_video_mask)

            input_video_mask = input_video_mask.to(input_video.device, input_video.dtype)

        else:
            raise NotImplementedError(f"Not supported validation_video_mask format {validation_video_mask}")

    if ref_image is not None:
        if isinstance(ref_image, str):
            clip_image = Image.open(ref_image).convert("RGB")
        else:
            clip_image = Image.fromarray(np.array(ref_image, np.uint8))
    else:
        clip_image = None

    if ref_image is not None:
        if isinstance(ref_image, str):
            ref_image = Image.open(ref_image).convert("RGB")
            ref_image = ref_image.resize((sample_size[1], sample_size[0]))
            ref_image = torch.from_numpy(np.array(ref_image))
            ref_image = ref_image.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255
        else:
            ref_image = torch.from_numpy(np.array(ref_image))
            ref_image = ref_image.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255
    return input_video, input_video_mask, ref_image, clip_image


def read_mask_video_binary(mask_path, sample_size, video_length, dilate_width=11):
    video_mask = media.read_video(mask_path)[:video_length]
    if len(video_mask.shape) == 4:  # (t, h, w, c)
        video_mask = video_mask[..., 0] # (t, h, w)
    video_mask = torch.from_numpy(video_mask).unsqueeze(0)  # (1, t, h, w)
    video_mask = F.interpolate(video_mask.float(), sample_size, mode='area')
    video_mask = torch.where(video_mask < 240, 0, 255).unsqueeze(0)  # (1, 1, t, h, w)
    if dilate_width > 0:
        video_mask = dilate_video_mask(video_mask, width=dilate_width)
    return video_mask


def temporal_padding(video, min_length=85, max_length=197, dim=2):
    length = video.size(dim)

    min_len = (length // 4) * 4 + 1
    if min_len < length:
        min_len += 4
    if (min_len // 4) % 2 == 0:
        min_len += 4
    target_length = min(min_len, max_length)
    target_length = max(min_length, target_length)

    logger.debug(f'video size: {video.shape}')
    if dim == 0:
        video = video[:target_length]
    elif dim == 1:
        video = video[:, :target_length]
    elif dim == 2:
        video = video[:, :, :target_length]
    elif dim == 3:
        video = video[:, :, :, :target_length]
    else:
        raise NotImplementedError
    logger.debug(f'making video length: {target_length}, padding length: {target_length - length}')
    while video.size(dim) < target_length:
        video_flipped = torch.flip(video, [dim])
        video = torch.cat([video, video_flipped], dim=dim)
        if dim == 0:
            video = video[:target_length]
        elif dim == 1:
            video = video[:, :target_length]
        elif dim == 2:
            video = video[:, :, :target_length]
        elif dim == 3:
            video = video[:, :, :, :target_length]
        else:
            raise NotImplementedError
    logger.debug(f'return video size: {video.shape}')
    return video


def get_video_mask_input(
        input_video_name,
        sample_size,
        keep_fg_ids=[-1],
        max_video_length=49,
        temporal_window_size=49,
        data_rootdir="datasets/test/",
        use_trimask=False,
        use_quadmask=False,
        use_fixed_bbox=False,
        dilate_width=11,
        apply_temporal_padding=True,
    ): 
    input_video_path = os.path.join(data_rootdir, input_video_name, "input_video.mp4")
    mask_paths = sorted(list(glob.glob(os.path.join(data_rootdir, input_video_name, 'mask_*.mp4'))))
    prompt = json.load(open(os.path.join(data_rootdir, input_video_name, "prompt.json")))['bg']
    
    input_video = media.read_video(input_video_path)
    clip_image = Image.fromarray(np.array(input_video[0]))

    input_video = torch.from_numpy(np.array(input_video))[:max_video_length]
    input_video = input_video.permute([3, 0, 1, 2]).float() / 255  # (c, t, h, w)
    input_video = F.interpolate(input_video, sample_size, mode='area').unsqueeze(0)  # (1, c, t, h, w)

    masks_to_remove = []
    masks_to_keep = []
    if mask_paths:
        for fg_id, mask_path in enumerate(mask_paths):
            if -1 in keep_fg_ids or fg_id not in keep_fg_ids:
                masks_to_remove.append(mask_path)
            else:
                masks_to_keep.append(mask_path)
        input_mask = None
        if use_trimask:
            for mask_path in masks_to_keep:
                mask_i = read_mask_video_binary(mask_path, sample_size, max_video_length, dilate_width=dilate_width)
                if input_mask is None:
                    input_mask = mask_i
                else:
                    input_mask = torch.where(mask_i > 127, 255, input_mask)
            if input_mask is not None:
                input_mask = torch.where(input_mask > 127, 0, 127)  # mask region --> 0 (keep), background --> 127 (neutral)

        for mask_path in masks_to_remove:
            mask_i = read_mask_video_binary(mask_path, sample_size, max_video_length, dilate_width=dilate_width)
            if input_mask is None:
                if use_trimask:
                    input_mask = torch.where(mask_i > 127, 255, 127)
                else:
                    input_mask = mask_i
            else:
                input_mask = torch.where(mask_i > 127, 255, input_mask)
    else:  # already has trimask/quadmask video ready
        # Look for mask files (can be trimask or quadmask)
        mask_files = sorted(list(glob.glob(os.path.join(data_rootdir, input_video_name, 'mask*.mp4'))))
        if not mask_files:
            mask_files = sorted(list(glob.glob(os.path.join(data_rootdir, input_video_name, 'quadmask_*.mp4'))))

        if (use_trimask or use_quadmask) and mask_files:
            input_mask = torch.from_numpy(media.read_video(mask_files[0])).float()[:max_video_length]
            if len(input_mask.shape) == 4: input_mask = input_mask[..., 0]
            input_mask = F.interpolate(input_mask.unsqueeze(0), sample_size, mode='area').unsqueeze(0)  # (1, 1, t, h, w)

            # Apply mask quantization based on mode
            if use_quadmask:
                # Quadmask mode: preserve 4 values [0, 63, 127, 255]
                input_mask = torch.where(input_mask <= 31, 0, input_mask)
                input_mask = torch.where((input_mask > 31) * (input_mask <= 95), 63, input_mask)
                input_mask = torch.where((input_mask > 95) * (input_mask <= 191), 127, input_mask)
                input_mask = torch.where(input_mask > 191, 255, input_mask)
                input_mask = 255 - input_mask
                logger.debug(f'[QUADMASK INFERENCE] Using 4-value quadmask: [0, 63, 127, 255]')
            else:
                # Trimask mode: 3 values [0, 127, 255]
                input_mask = torch.where(input_mask > 192, 255, input_mask)
                input_mask = torch.where((input_mask <= 192) * (input_mask >= 64), 128, input_mask)
                input_mask = torch.where(input_mask < 64, 0, input_mask)
                input_mask = 255 - input_mask
                logger.debug(f'[TRIMASK INFERENCE] Using 3-value trimask: [0, 127, 255]')
        else:
            logger.error(f'Masks not found in {os.path.join(data_rootdir, input_video_name)}')
            sys.exit(1)

    if use_fixed_bbox and not use_trimask:
        logger.debug('Using fixed bbox')
        input_mask = mask_to_fixed_bbox(input_mask)

    input_mask = input_mask.to(input_video.device, input_video.dtype)
    if apply_temporal_padding:
        input_video = temporal_padding(input_video, min_length=temporal_window_size, max_length=max_video_length)
        input_mask = temporal_padding(input_mask, min_length=temporal_window_size, max_length=max_video_length)
    input_mask = input_mask / 255.
    logger.debug('dataloading mask', input_mask.min(), input_mask.max(), input_mask.dtype, input_mask.shape)
    return input_video, input_mask, prompt, clip_image


def get_video_mask_validation(
        input_video_name,
        sample_size,
        max_video_length=49,
        temporal_window_size=49,
        data_rootdir="datasets/test/",
        use_trimask=False,
        use_fixed_bbox=False,
        dilate_width=11,
        caption_path="datasets/vidgen1m/VidGen_1M_video_caption.json",
    ):
    caption_list = json.load(open(caption_path, 'r'))
    prompt = None
    for caption_item in caption_list:
        if caption_item["vid"] == input_video_name.split('.')[0]:
            prompt = caption_item["caption"]
            break
    assert prompt is not None

    input_video_path = os.path.join(data_rootdir, input_video_name)
    input_video = media.read_video(input_video_path)
    input_video = torch.from_numpy(np.array(input_video))[:max_video_length]
    input_video = input_video.permute([3, 0, 1, 2]).float() / 255  # (c, t, h, w)
    input_video = F.interpolate(input_video, sample_size, mode='area').unsqueeze(0)  # (1, c, t, h, w)

    input_video = temporal_padding(input_video, min_length=temporal_window_size, max_length=max_video_length)
    input_mask = get_random_mask((input_video.size(2), input_video.size(1), input_video.size(3), input_video.size(4)))
    input_mask = input_mask.to(input_video.device, input_video.dtype)
    input_mask = input_mask.permute(1, 0, 2, 3).unsqueeze(0)
    return input_video, input_mask, prompt


def get_video(
        input_video_path,
        sample_size,
        max_video_length=49,
        temporal_window_size=49,
    ):
    input_video = media.read_video(input_video_path)
    input_video = torch.from_numpy(np.array(input_video))[:max_video_length]
    input_video = input_video.permute([3, 0, 1, 2]).float() / 255  # (c, t, h, w)
    input_video = F.interpolate(input_video, sample_size, mode='area').unsqueeze(0)  # (1, c, t, h, w)

    input_video = temporal_padding(input_video, min_length=temporal_window_size, max_length=max_video_length)
    return input_video


def dilate_video_mask(video_mask, width=11):
    
    is_tensor = torch.is_tensor(video_mask)
    if is_tensor:
        video_mask = video_mask[0, 0].numpy()  # (t, h, w)
    if video_mask.max() > 127:
        video_mask = video_mask.astype(np.uint8)
    elif video_mask.max() <= 1.0:
        video_mask = (video_mask * 255).astype(np.uint8)
    is_dim4 = len(video_mask.shape) == 4
    if is_dim4:
        video_mask = video_mask[..., -1]

    dilated_video_mask = []
    for mask in video_mask:
        dilated_mask = skimage.morphology.binary_dilation(mask, footprint=np.ones((width, width)))
        dilated_mask = np.where(dilated_mask, 255, 0)
        dilated_video_mask.append(dilated_mask)
    dilated_video_mask = np.stack(dilated_video_mask)

    if is_dim4:
        dilated_video_mask = dilated_video_mask[..., None]
    if is_tensor:
        dilated_video_mask = torch.from_numpy(dilated_video_mask).unsqueeze(0).unsqueeze(0)
    return dilated_video_mask


def erode_video_mask(video_mask, width=5):
    
    is_tensor = torch.is_tensor(video_mask)
    if is_tensor:
        video_mask = video_mask[0, 0].numpy()  # (t, h, w)
    if video_mask.max() > 127:
        video_mask = video_mask.astype(np.uint8)
    elif video_mask.max() <= 1.0:
        video_mask = (video_mask * 255).astype(np.uint8)
    is_dim4 = len(video_mask.shape) == 4
    if is_dim4:
        video_mask = video_mask[..., -1]

    eroded_video_mask = []
    for mask in video_mask:
        eroded_mask = skimage.morphology.binary_erosion(mask, footprint=np.ones((width, width)))
        eroded_mask = np.where(eroded_mask, 255, 0)
        eroded_video_mask.append(eroded_mask)
    eroded_video_mask = np.stack(eroded_video_mask)

    if is_dim4:
        eroded_video_mask = eroded_video_mask[..., None]
    if is_tensor:
        eroded_video_mask = torch.from_numpy(eroded_video_mask).unsqueeze(0).unsqueeze(0)
    return eroded_video_mask


def mask_to_bbox(video_mask):
    is_tensor = torch.is_tensor(video_mask)
    if is_tensor:
        video_mask = video_mask[0, 0].numpy()  # (t, h, w)
    if video_mask.max() > 127:
        video_mask = video_mask.astype(np.uint8)
    elif video_mask.max() <= 1.0:
        video_mask = (video_mask * 255).astype(np.uint8)
    is_dim4 = len(video_mask.shape) == 4
    if is_dim4:
        video_mask = video_mask[..., -1]

    bbox_masks = []
    for mask in video_mask:
        bbox_mask = np.zeros_like(mask)
        t, b, l, r = 0, mask.shape[0] - 1, 0, mask.shape[1] - 1
        while(mask[t].sum() == 0): t += 1
        while(mask[b].sum() == 0): b -= 1
        while(mask[:, l].sum() == 0): l += 1
        while(mask[:, r].sum() == 0): r -= 1
        bbox_mask[t:b, l:r] = 255
        bbox_masks.append(bbox_mask)
    bbox_masks = np.stack(bbox_masks)
    if is_dim4:
        bbox_masks = bbox_masks[..., None]
    if is_tensor:
        bbox_masks = torch.from_numpy(bbox_masks).unsqueeze(0).unsqueeze(0)
    return bbox_masks


def mask_to_fixed_bbox(video_mask):
    is_tensor = torch.is_tensor(video_mask)
    if is_tensor:
        video_mask = video_mask[0, 0].numpy()  # (t, h, w)
    if video_mask.max() > 127:
        video_mask = video_mask.astype(np.uint8)
    elif video_mask.max() <= 1.0:
        video_mask = (video_mask * 255).astype(np.uint8)
    is_dim4 = len(video_mask.shape) == 4
    if is_dim4:
        video_mask = video_mask[..., -1]

    bbox_masks = []

    # for mask in video_mask:
    mask = video_mask
    bbox_mask = np.zeros_like(mask)
    t, b, l, r = 0, mask.shape[1] - 1, 0, mask.shape[2] - 1
    while(mask[:, t].sum() == 0): t += 1
    while(mask[:, b].sum() == 0): b -= 1
    while(mask[:, :, l].sum() == 0): l += 1
    while(mask[:, :, r].sum() == 0): r -= 1
    bbox_mask[:, t:b, l:r] = 255
    # bbox_masks.append(bbox_mask)
    # bbox_masks = np.stack(bbox_masks)
    bbox_masks = bbox_mask
    if is_dim4:
        bbox_masks = bbox_masks[..., None]
    if is_tensor:
        bbox_masks = torch.from_numpy(bbox_masks).unsqueeze(0).unsqueeze(0)
    return bbox_masks


def apply_colormap(video):
    if len(video.shape) == 4:
        video = video.mean(-1)
    if video.max() >= 2.0:
        video = video.astype(float) / 255.

    video_colored = []
    cmap = matplotlib.colormaps['turbo']
    for frame in video:
        frame =  cmap(frame)[..., :3]
        video_colored.append(frame)
    video_colored = np.stack(video_colored)
    return video_colored

