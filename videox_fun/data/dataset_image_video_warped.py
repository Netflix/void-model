import csv
import io
import json
import math
import os
import glob
import random
from threading import Thread
import mediapy as media
import time

import albumentations
import cv2
import gc
import numpy as np
import torch
import torchvision.transforms as transforms
from scipy.special import binom

from func_timeout import func_timeout, FunctionTimedOut
from decord import VideoReader
from PIL import Image
from torch.utils.data import BatchSampler, Sampler
from torch.utils.data.dataset import Dataset
from contextlib import contextmanager
from videox_fun.utils.video_tuple_loader import load_video_mask_tuple_clip

VIDEO_READER_TIMEOUT = 20

bernstein = lambda n, k, t: binom(n,k)* t**k * (1.-t)**(n-k)

# codes from https://stackoverflow.com/questions/50731785/create-random-shape-contour-using-matplotlib
def bezier(points, num=200):
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve

class Segment():
    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1
        self.p2 = p2
        self.angle1 = angle1
        self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 100)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2-self.p1)**2))
        self.r = r*d
        self.p = np.zeros((4,2))
        self.p[0,:] = self.p1[:]
        self.p[3,:] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def calc_intermediate_points(self,r):
        self.p[1,:] = self.p1 + np.array(
            [self.r*np.cos(self.angle1), self.r*np.sin(self.angle1)])
        self.p[2,:] = self.p2 + np.array(
            [self.r*np.cos(self.angle2+np.pi), self.r*np.sin(self.angle2+np.pi)])
        self.curve = bezier(self.p,self.numpoints)


def get_curve(points, **kw):
    segments = []
    for i in range(len(points)-1):
        seg = Segment(points[i,:2], points[i+1,:2], points[i,2],points[i+1,2],**kw)
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve


def ccw_sort(p):
    d = p-np.mean(p,axis=0)
    s = np.arctan2(d[:,0], d[:,1])
    return p[np.argsort(s),:]


def get_bezier_curve(a, rad=0.2, edgy=0):
    """ given an array of points *a*, create a curve through
    those points. 
    *rad* is a number between 0 and 1 to steer the distance of
          control points.
    *edgy* is a parameter which controls how "edgy" the curve is,
           edgy=0 is smoothest."""
    p = np.arctan(edgy)/np.pi+.5
    a = ccw_sort(a)
    a = np.append(a, np.atleast_2d(a[0,:]), axis=0)
    d = np.diff(a, axis=0)
    ang = np.arctan2(d[:,1],d[:,0])
    f = lambda ang : (ang>=0)*ang + (ang<0)*(ang+2*np.pi)
    ang = f(ang)
    ang1 = ang
    ang2 = np.roll(ang,1)
    ang = p*ang1 + (1-p)*ang2 + (np.abs(ang2-ang1) > np.pi )*np.pi
    ang = np.append(ang, [ang[0]])
    a = np.append(a, np.atleast_2d(ang).T, axis=1)
    s, c = get_curve(a, r=rad, method="var")
    x,y = c.T
    return x,y, a


def get_random_points(n=5, scale=0.8, mindst=None, rec=0):
    """ create n random points in the unit square, which are *mindst*
    apart, then scale them."""
    mindst = mindst or .7/n
    a = np.random.rand(n,2)
    d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1)**2)
    if np.all(d >= mindst) or rec>=200:
        return a*scale
    else:
        return get_random_points(n=n, scale=scale, mindst=mindst, rec=rec+1)


def fill_mask(shape, x, y, fill_val=255):
    _, _, h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    mask = cv2.fillPoly(mask, [np.array([x, y], np.int32).T], fill_val)
    return mask


def random_shift(x, y, scale_range = [0.2, 0.7], trans_perturb_range=[-0.2, 0.2]):
    w_scale = np.random.uniform(scale_range[0], scale_range[1])
    h_scale = np.random.uniform(scale_range[0], scale_range[1])
    x_trans = np.random.uniform(0., 1. - w_scale)
    y_trans = np.random.uniform(0., 1. - h_scale)
    x_shifted = x * w_scale + x_trans + np.random.uniform(trans_perturb_range[0], trans_perturb_range[1])
    y_shifted = y * h_scale + y_trans + np.random.uniform(trans_perturb_range[0], trans_perturb_range[1])
    return x_shifted, y_shifted


def get_random_shape_mask(
        shape, n_pts_range=[3, 10], rad_range=[0.0, 1.0], edgy_range=[0.0, 0.1], n_keyframes_range=[2, 25],
        random_drop_range=[0.0, 0.2],
    ):
    f, _, h, w = shape

    n_pts = np.random.randint(n_pts_range[0], n_pts_range[1])
    n_keyframes = np.random.randint(n_keyframes_range[0], n_keyframes_range[1])
    keyframe_interval = f // (n_keyframes - 1)
    keyframe_indices = list(range(0, f, keyframe_interval))
    if len(keyframe_indices) == n_keyframes:
        keyframe_indices[-1] = f - 1
    else:
        keyframe_indices.append(f - 1)
    x_all_frames, y_all_frames = [], []
    for i, keyframe_index in enumerate(keyframe_indices):
        rad = np.random.uniform(rad_range[0], rad_range[1])
        edgy = np.random.uniform(edgy_range[0], edgy_range[1])
        x_kf, y_kf, _ = get_bezier_curve(get_random_points(n=n_pts), rad=rad, edgy=edgy)
        x_kf, y_kf = random_shift(x_kf, y_kf)
        if i == 0:
            x_all_frames.append(x_kf[None])
            y_all_frames.append(y_kf[None])
        else:
            x_interval = np.linspace(x_all_frames[-1][-1], x_kf, keyframe_index - keyframe_indices[i - 1] + 1)
            y_interval = np.linspace(y_all_frames[-1][-1], y_kf, keyframe_index - keyframe_indices[i - 1] + 1)
            x_all_frames.append(x_interval[1:])
            y_all_frames.append(y_interval[1:])
    x_all_frames = np.concatenate(x_all_frames, axis=0)
    y_all_frames = np.concatenate(y_all_frames, axis=0)

    masks = []
    for x, y in zip(x_all_frames, y_all_frames):
        x = np.round(x * w).astype(np.int32)
        y = np.round(y * h).astype(np.int32)
        mask = fill_mask(shape, x, y)
        masks.append(mask)
    masks = np.stack(masks, axis=0).astype(float) / 255.

    n_frames_random_drop = int(np.random.uniform(random_drop_range[0], random_drop_range[1]) * f)
    drop_index = np.random.randint(0, f - n_frames_random_drop)
    masks[drop_index:drop_index + n_frames_random_drop] = 0

    return masks  # (f, h, w), <float>[0, 1]


def get_random_mask(shape, mask_type_probs=[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.8]):
    f, c, h, w = shape

    if f != 1:
        mask_index = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], p=mask_type_probs)
    else:
        mask_index = np.random.choice([0, 1], p = [0.2, 0.8])
    mask = torch.zeros((f, 1, h, w), dtype=torch.uint8)

    if mask_index == 0:
        center_x = torch.randint(0, w, (1,)).item()
        center_y = torch.randint(0, h, (1,)).item()
        block_size_x = torch.randint(w // 4, w // 4 * 3, (1,)).item()
        block_size_y = torch.randint(h // 4, h // 4 * 3, (1,)).item()

        start_x = max(center_x - block_size_x // 2, 0)
        end_x = min(center_x + block_size_x // 2, w)
        start_y = max(center_y - block_size_y // 2, 0)
        end_y = min(center_y + block_size_y // 2, h)
        mask[:, :, start_y:end_y, start_x:end_x] = 1
    elif mask_index == 1:
        mask[:, :, :, :] = 1
    elif mask_index == 2:
        mask_frame_index = np.random.randint(1, 5)
        mask[mask_frame_index:, :, :, :] = 1
    elif mask_index == 3:
        mask_frame_index = np.random.randint(1, 5)
        mask[mask_frame_index:-mask_frame_index, :, :, :] = 1
    elif mask_index == 4:
        center_x = torch.randint(0, w, (1,)).item()
        center_y = torch.randint(0, h, (1,)).item()
        block_size_x = torch.randint(w // 4, w // 4 * 3, (1,)).item()
        block_size_y = torch.randint(h // 4, h // 4 * 3, (1,)).item()

        start_x = max(center_x - block_size_x // 2, 0)
        end_x = min(center_x + block_size_x // 2, w)
        start_y = max(center_y - block_size_y // 2, 0)
        end_y = min(center_y + block_size_y // 2, h)

        mask_frame_before = np.random.randint(0, f // 2)
        mask_frame_after = np.random.randint(f // 2, f)
        mask[mask_frame_before:mask_frame_after, :, start_y:end_y, start_x:end_x] = 1
    elif mask_index == 5:
        mask = torch.randint(0, 2, (f, 1, h, w), dtype=torch.uint8)
    elif mask_index == 6:
        num_frames_to_mask = random.randint(1, max(f // 2, 1))
        frames_to_mask = random.sample(range(f), num_frames_to_mask)

        for i in frames_to_mask:
            block_height = random.randint(1, h // 4)
            block_width = random.randint(1, w // 4)
            top_left_y = random.randint(0, h - block_height)
            top_left_x = random.randint(0, w - block_width)
            mask[i, 0, top_left_y:top_left_y + block_height, top_left_x:top_left_x + block_width] = 1
    elif mask_index == 7:
        center_x = torch.randint(0, w, (1,)).item()
        center_y = torch.randint(0, h, (1,)).item()
        a = torch.randint(min(w, h) // 8, min(w, h) // 4, (1,)).item()
        b = torch.randint(min(h, w) // 8, min(h, w) // 4, (1,)).item() 

        for i in range(h):
            for j in range(w):
                if ((i - center_y) ** 2) / (b ** 2) + ((j - center_x) ** 2) / (a ** 2) < 1:
                    mask[:, :, i, j] = 1
    elif mask_index == 8:
        center_x = torch.randint(0, w, (1,)).item()
        center_y = torch.randint(0, h, (1,)).item()
        radius = torch.randint(min(h, w) // 8, min(h, w) // 4, (1,)).item()
        for i in range(h):
            for j in range(w):
                if (i - center_y) ** 2 + (j - center_x) ** 2 < radius ** 2:
                    mask[:, :, i, j] = 1
    elif mask_index == 9:
        for idx in range(f):
            if np.random.rand() > 0.5:
                mask[idx, :, :, :] = 1
    else:
        num_objs = np.random.randint(1, 4)
        mask_npy = get_random_shape_mask(shape)
        for i in range(num_objs - 1):
            mask_npy += get_random_shape_mask(shape).clip(0, 1)

        mask = torch.from_numpy(mask_npy).unsqueeze(1)

    return mask.float()


def get_random_mask_multi(shape, mask_type_probs, range_num_masks=[1, 7]):
    num_masks = np.random.randint(range_num_masks[0], range_num_masks[1])
    masks = None
    for _ in range(num_masks):
        mask = get_random_mask(shape, mask_type_probs)
        if masks is None:
            masks = mask
        else:
            masks = (masks + mask).clip(0, 1)
    return masks


class ImageVideoSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """

    def __init__(self,
                 sampler: Sampler,
                 dataset: Dataset,
                 batch_size: int,
                 drop_last: bool = False
                ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # buckets for each aspect ratio
        self.bucket = {'image':[], 'video':[], 'video_mask_tuple':[]}

    def __iter__(self):
        for idx in self.sampler:
            content_type = self.dataset.dataset[idx].get('type', 'image')
            self.bucket[content_type].append(idx)

            # yield a batch of indices in the same aspect ratio group
            if len(self.bucket['video']) == self.batch_size:
                bucket = self.bucket['video']
                yield bucket[:]
                del bucket[:]
            elif len(self.bucket['video_mask_tuple']) == self.batch_size:
                bucket = self.bucket['video_mask_tuple']
                yield bucket[:]
                del bucket[:]
            elif len(self.bucket['image']) == self.batch_size:
                bucket = self.bucket['image']
                yield bucket[:]
                del bucket[:]


@contextmanager
def VideoReader_contextmanager(*args, **kwargs):
    vr = VideoReader(*args, **kwargs)
    try:
        yield vr
    finally:
        del vr
        gc.collect()


def get_video_reader_batch(video_reader, batch_index):
    frames = video_reader.get_batch(batch_index).asnumpy()
    return frames


def _read_video_from_dir(video_dir):
    frames = []
    frame_paths = sorted(list(glob.glob(os.path.join(video_dir, '*.png'))))

    if not frame_paths:
        raise ValueError(f"No PNG files found in directory: {video_dir}")

    for frame_path in frame_paths:
        frame = media.read_image(frame_path)
        frames.append(frame)

    if not frames:
        raise ValueError(f"Failed to read any frames from directory: {video_dir}")

    return np.stack(frames, axis=0)


def resize_frame(frame, target_short_side):
    h, w, _ = frame.shape
    if h < w:
        if target_short_side > h:
            return frame
        new_h = target_short_side
        new_w = int(target_short_side * w / h)
    else:
        if target_short_side > w:
            return frame
        new_w = target_short_side
        new_h = int(target_short_side * h / w)

    resized_frame = cv2.resize(frame, (new_w, new_h))
    return resized_frame


class ImageVideoDataset(Dataset):
    def __init__(
            self,
            ann_path, data_root=None,
            video_sample_size=512, video_sample_stride=4, video_sample_n_frames=16,
            image_sample_size=512,
            video_repeat=0,
            text_drop_ratio=0.1,
            enable_bucket=False,
            video_length_drop_start=0.0,
            video_length_drop_end=1.0,
            enable_inpaint=False,
            trimask_zeroout_removal=False,
            use_quadmask=False,
            ablation_binary_mask=False,
        ):
        # Loading annotations from files
        print(f"loading annotations from {ann_path} ...")
        if ann_path.endswith('.csv'):
            with open(ann_path, 'r') as csvfile:
                dataset = list(csv.DictReader(csvfile))
        elif ann_path.endswith('.json'):
            dataset = json.load(open(ann_path))
        else:
            raise ValueError(f"Unsupported annotation file format: {ann_path}. Only .csv and .json files are supported.")

        self.data_root = data_root

        # It's used to balance num of images and videos.
        self.dataset = []
        for data in dataset:
            if data.get('type', 'image') != 'video':
                self.dataset.append(data)
        if video_repeat > 0:
            for _ in range(video_repeat):
                for data in dataset:
                    if data.get('type', 'image') == 'video':
                        self.dataset.append(data)
        del dataset

        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        # TODO: enable bucket training
        self.enable_bucket = enable_bucket
        self.text_drop_ratio = text_drop_ratio
        self.enable_inpaint  = enable_inpaint
        self.trimask_zeroout_removal = trimask_zeroout_removal
        self.use_quadmask = use_quadmask
        self.ablation_binary_mask = ablation_binary_mask

        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end

        if self.use_quadmask:
            print(f"[QUADMASK MODE] Using 4-value quadmask: [0, 63, 127, 255]")
        if self.ablation_binary_mask:
            print(f"[ABLATION BINARY MASK] Remapping quadmask to binary: [0,63]→0, [127,255]→127")
        else:
            print(f"[TRIMASK MODE] Using 3-value trimask: [0, 127, 255]")

        # Video params
        self.video_sample_stride    = video_sample_stride
        self.video_sample_n_frames  = video_sample_n_frames
        self.video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.video_transforms = transforms.Compose(
            [
                transforms.Resize(min(self.video_sample_size)),
                transforms.CenterCrop(self.video_sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

        # Image params
        self.image_sample_size  = tuple(image_sample_size) if not isinstance(image_sample_size, int) else (image_sample_size, image_sample_size)
        self.image_transforms   = transforms.Compose([
            transforms.Resize(min(self.image_sample_size)),
            transforms.CenterCrop(self.image_sample_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        ])

        self.larger_side_of_image_and_video = max(min(self.image_sample_size), min(self.video_sample_size))

    def get_batch(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]

        if data_info.get('type', 'image') == 'video' and data_info.get('mask_path', None) is None:
            video_id, text = data_info['file_path'], data_info['text']

            if self.data_root is None:
                video_dir = video_id
            else:
                video_dir = os.path.join(self.data_root, video_id)

            with VideoReader_contextmanager(video_dir, num_threads=2) as video_reader:
                min_sample_n_frames = min(
                    self.video_sample_n_frames, 
                    int(len(video_reader) * (self.video_length_drop_end - self.video_length_drop_start) // self.video_sample_stride)
                )
                if min_sample_n_frames == 0:
                    raise ValueError(f"No Frames in video.")

                video_length = int(self.video_length_drop_end * len(video_reader))
                clip_length = min(video_length, (min_sample_n_frames - 1) * self.video_sample_stride + 1)
                start_idx   = random.randint(int(self.video_length_drop_start * video_length), video_length - clip_length) if video_length != clip_length else 0
                batch_index = np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)

                try:
                    sample_args = (video_reader, batch_index)
                    pixel_values = func_timeout(
                        VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                    )
                    resized_frames = []
                    for i in range(len(pixel_values)):
                        frame = pixel_values[i]
                        resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                        resized_frames.append(resized_frame)
                    pixel_values = np.array(resized_frames)
                except FunctionTimedOut:
                    raise ValueError(f"Read {idx} timeout.")
                except Exception as e:
                    raise ValueError(f"Failed to extract frames from video. Error is {e}.")

                if not self.enable_bucket:
                    pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
                    pixel_values = pixel_values / 255.
                    del video_reader
                else:
                    pixel_values = pixel_values

                if not self.enable_bucket:
                    pixel_values = self.video_transforms(pixel_values)

                # Random use no text generation
                if random.random() < self.text_drop_ratio:
                    text = ''
            return {
                'pixel_values': pixel_values, 
                'text': text, 
                'data_type': 'video',
            }
        elif data_info.get('type', 'image') == 'video' and data_info.get('mask_path', None) is not None:  # video with known mask
            video_path, text = data_info['file_path'], data_info['text']
            mask_video_path = video_path[:-4] + '_mask.mp4'
            with VideoReader_contextmanager(video_path, num_threads=2) as video_reader:
                min_sample_n_frames = min(
                    self.video_sample_n_frames, 
                    int(len(video_reader) * (self.video_length_drop_end - self.video_length_drop_start) // self.video_sample_stride)
                )
                if min_sample_n_frames == 0:
                    raise ValueError(f"No Frames in video.")

                video_length = int(self.video_length_drop_end * len(video_reader))
                clip_length = min(video_length, (min_sample_n_frames - 1) * self.video_sample_stride + 1)
                start_idx   = random.randint(int(self.video_length_drop_start * video_length), video_length - clip_length) if video_length != clip_length else 0
                batch_index = np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)

                try:
                    sample_args = (video_reader, batch_index)
                    pixel_values = func_timeout(
                        VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                    )
                    resized_frames = []
                    for i in range(len(pixel_values)):
                        frame = pixel_values[i]
                        resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                        resized_frames.append(resized_frame)
                    input_video = np.array(resized_frames)
                except FunctionTimedOut:
                    raise ValueError(f"Read {idx} timeout.")
                except Exception as e:
                    raise ValueError(f"Failed to extract frames from video. Error is {e}.")

            with VideoReader_contextmanager(mask_video_path, num_threads=2) as video_reader:
                try:
                    sample_args = (video_reader, batch_index)
                    mask_values = func_timeout(
                        VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                    )
                    resized_frames = []
                    for i in range(len(mask_values)):
                        frame = mask_values[i]
                        resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                        resized_frames.append(resized_frame)
                    mask_video = np.array(resized_frames)
                except FunctionTimedOut:
                    raise ValueError(f"Read {idx} timeout.")
                except Exception as e:
                    raise ValueError(f"Failed to extract frames from video. Error is {e}.")

            if len(mask_video.shape) == 3:
                mask_video = mask_video[..., None]
            if mask_video.shape[-1] == 3:
                mask_video = mask_video[..., :1]
            if len(mask_video.shape) != 4:
                raise ValueError(f"mask_video shape is {mask_video.shape}.")

            text = data_info['text']
            if not self.enable_bucket:
                input_video = torch.from_numpy(input_video).permute(0, 3, 1, 2).contiguous() / 255.
                mask_video = torch.from_numpy(mask_video).permute(0, 3, 1, 2).contiguous() / 255.

                pixel_values = torch.cat([input_video, mask_video], dim=1)
                pixel_values = self.video_transforms(pixel_values)
                input_video = pixel_values[:, :3]
                mask_video = pixel_values[:, 3:]

            # Random use no text generation
            if random.random() < self.text_drop_ratio:
                text = ''

            return {
                'pixel_values': input_video,
                'mask': mask_video,
                'text': text,
                'data_type': 'video',
            }

        elif data_info.get('type', 'image') == 'video_mask_tuple':  # object effect removal
            sample_dir = data_info['file_path'] if self.data_root is None else os.path.join(self.data_root, data_info['file_path'])
            try:
                input_video, target_video, mask_video, depth_video = load_video_mask_tuple_clip(
                    sample_dir=sample_dir,
                    video_sample_n_frames=self.video_sample_n_frames,
                    video_length_drop_start=self.video_length_drop_start,
                    video_length_drop_end=self.video_length_drop_end,
                    video_sample_stride=self.video_sample_stride,
                    target_short_side=self.larger_side_of_image_and_video,
                )
            except Exception as e:
                print(f"Error loading video_mask_tuple from {sample_dir}: {e}")
                import traceback
                traceback.print_exc()
                raise

            mask_video = 255 - mask_video  # will be flipped again in when feeding to model

            if len(mask_video.shape) == 3:
                mask_video = mask_video[..., None]
            if mask_video.shape[-1] == 3:
                mask_video = mask_video[..., :1]

            # Apply mask quantization after resize to preserve existing trimask / quadmask behavior.
            if self.ablation_binary_mask:
                # Ablation binary mask mode: remap [0, 63, 127, 255] to [0, 127]
                # Map 0 and 63 → 0
                # Map 127 and 255 → 127
                mask_video = np.where(mask_video <= 95, 0, mask_video)
                mask_video = np.where(mask_video > 95, 127, mask_video)
            elif self.use_quadmask:
                # Quadmask mode: preserve 4 values [0, 63, 127, 255]
                # Quantize to nearest quadmask value for robustness
                mask_video = np.where(mask_video <= 31, 0, mask_video)
                mask_video = np.where(np.logical_and(mask_video > 31, mask_video <= 95), 63, mask_video)
                mask_video = np.where(np.logical_and(mask_video > 95, mask_video <= 191), 127, mask_video)
                mask_video = np.where(mask_video > 191, 255, mask_video)
            else:
                # Trimask mode: 3 values [0, 127, 255]
                mask_video = np.where(np.logical_and(mask_video > 63, mask_video < 192), 127, mask_video)
                mask_video = np.where(mask_video >= 192, 255, mask_video)
                mask_video = np.where(mask_video <= 63, 0, mask_video)

            if len(mask_video.shape) == 3:
                mask_video = mask_video[..., None]
            if mask_video.shape[-1] == 3:
                mask_video = mask_video[..., :1]
            if len(mask_video.shape) != 4:
                raise ValueError(f"mask_video shape is {mask_video.shape}.")

            text = data_info['text']
            print(f"DEBUG DATASET: Converting to tensors (enable_bucket={self.enable_bucket})...")
            if not self.enable_bucket:
                print(f"DEBUG DATASET: Converting input_video to tensor...")
                input_video = torch.from_numpy(input_video).permute(0, 3, 1, 2).contiguous() / 255.
                print(f"DEBUG DATASET: Converting target_video to tensor...")
                target_video = torch.from_numpy(target_video).permute(0, 3, 1, 2).contiguous() / 255.
                print(f"DEBUG DATASET: Converting mask_video to tensor...")
                mask_video = torch.from_numpy(mask_video).permute(0, 3, 1, 2).contiguous() / 255.

                # Process depth video if available
                if depth_video is not None:
                    print(f"DEBUG DATASET: Processing depth_video...")
                    # IMPORTANT: Copy depth_video to ensure it's not memory-mapped
                    # Memory-mapped files can cause bus errors on GPU transfer
                    print(f"DEBUG DATASET: Copying depth_video to ensure not memory-mapped...")
                    depth_video = np.array(depth_video, copy=True)
                    print(f"DEBUG DATASET: depth_video copied, shape={depth_video.shape}")

                    # Ensure depth has correct shape
                    if len(depth_video.shape) == 3:
                        depth_video = depth_video[..., None]
                    if depth_video.shape[-1] == 3:
                        # Convert to grayscale if RGB
                        print(f"DEBUG DATASET: Converting depth to grayscale...")
                        depth_video = depth_video.mean(axis=-1, keepdims=True)
                    # Convert to tensor [F, 1, H, W] and normalize to [0, 1]
                    print(f"DEBUG DATASET: Converting depth to tensor...")
                    depth_video = torch.from_numpy(depth_video).permute(0, 3, 1, 2).contiguous().float() / 255.
                    # Ensure tensor is contiguous and owned
                    print(f"DEBUG DATASET: Cloning depth tensor...")
                    depth_video = depth_video.clone().contiguous()
                    print(f"DEBUG DATASET: depth_video final shape: {depth_video.shape}, is_contiguous: {depth_video.is_contiguous()}")

                # Apply transforms to each video separately (they expect 3 channels)
                print(f"DEBUG DATASET: Applying video transforms...")
                input_video = self.video_transforms(input_video)
                target_video = self.video_transforms(target_video)
                # Don't normalize mask since it's single channel
                print(f"DEBUG DATASET: Normalizing mask_video...")
                mask_video = mask_video * 2.0 - 1.0  # Scale to [-1, 1] like other channels
                print(f"DEBUG DATASET: All tensors ready (non-bucket mode)")

            else:
                # For bucket mode, keep as numpy until collate
                # Collate function expects [0, 255] range and will normalize
                print(f"DEBUG DATASET: Bucket mode - keeping as numpy in [0, 255] range...")
                print(f"DEBUG DATASET: All numpy arrays ready (bucket mode)")

            # Load warped noise - REQUIRED if specified in dataset
            warped_noise = None
            if 'warped_noise_path' in data_info:
                warped_noise_dir = data_info['warped_noise_path'] if self.data_root is None else os.path.join(self.data_root, data_info['warped_noise_path'])
                noise_path = os.path.join(warped_noise_dir, 'noises.npy')

                if not os.path.exists(noise_path):
                    raise FileNotFoundError(
                        f"Warped noise path specified in dataset but file not found: {noise_path}\n"
                        f"Make sure you've generated warped noise for all videos in the dataset."
                    )

                try:
                    warped_noise = np.load(noise_path)  # Shape: (T, C, H, W) in float16
                    warped_noise = torch.from_numpy(warped_noise).float()  # Convert to torch tensor
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load warped noise from {noise_path}: {e}\n"
                        f"The noise file may be corrupted. Try regenerating it."
                    )

            # Random use no text generation
            if random.random() < self.text_drop_ratio:
                text = ''

            if self.trimask_zeroout_removal:
                input_video = input_video * np.where(mask_video > 200, 0, 1).astype(input_video.dtype)

            result = {
                'pixel_values': target_video,
                'input_condition': input_video,
                'mask': mask_video,
                'text': text,
                'data_type': 'video_mask_tuple',
            }

            # Add depth maps if available
            if depth_video is not None:
                result['depth_maps'] = depth_video

            # Add warped noise to batch if available
            if warped_noise is not None:
                result['warped_noise'] = warped_noise

            return result

        else:
            image_path, text = data_info['file_path'], data_info['text']
            if self.data_root is not None:
                image_path = os.path.join(self.data_root, image_path)
            image = Image.open(image_path).convert('RGB')
            if not self.enable_bucket:
                image = self.image_transforms(image).unsqueeze(0)
            else:
                image = np.expand_dims(np.array(image), 0)
            if random.random() < self.text_drop_ratio:
                text = ''
            return {
                'pixel_values': image, 
                'text': text, 
                'data_type': 'image',
            }

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        data_type = data_info.get('type', 'image')
        while True:
            sample = {}
            try:
                data_info_local = self.dataset[idx % len(self.dataset)]
                data_type_local = data_info_local.get('type', 'image')
                if data_type_local != data_type:
                    raise ValueError("data_type_local != data_type")

                sample = self.get_batch(idx)
                sample["idx"] = idx

                if len(sample) > 0:
                    break
            except Exception as e:
                import traceback
                print(f"Error loading sample at index {idx}:")
                print(f"Data info: {self.dataset[idx % len(self.dataset)]}")
                print(f"Error: {e}")
                traceback.print_exc()
                idx = random.randint(0, self.length-1)

        if self.enable_inpaint and not self.enable_bucket:
            if "mask" not in sample:
                mask = get_random_mask_multi(sample["pixel_values"].size())
                sample["mask"] = mask
            else:
                mask = sample["mask"]

            if "input_condition" in sample:
                mask_pixel_values = sample["input_condition"]
            else:
                mask_pixel_values = sample["pixel_values"]
                mask_pixel_values = mask_pixel_values * (1 - mask) + torch.ones_like(mask_pixel_values) * -1 * mask

            sample["mask_pixel_values"] = mask_pixel_values

            clip_pixel_values = sample["pixel_values"][0].permute(1, 2, 0).contiguous()
            clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
            sample["clip_pixel_values"] = clip_pixel_values

            ref_pixel_values = sample["pixel_values"][0].unsqueeze(0)
            if (mask == 1).all():
                ref_pixel_values = torch.ones_like(ref_pixel_values) * -1
            sample["ref_pixel_values"] = ref_pixel_values

        return sample


class ImageVideoControlDataset(Dataset):
    def __init__(
            self,
            ann_path, data_root=None,
            video_sample_size=512, video_sample_stride=4, video_sample_n_frames=16,
            image_sample_size=512,
            video_repeat=0,
            text_drop_ratio=0.1,
            enable_bucket=False,
            video_length_drop_start=0.0, 
            video_length_drop_end=1.0,
            enable_inpaint=False,
    ):
        # Loading annotations from files
        print(f"loading annotations from {ann_path} ...")
        if ann_path.endswith('.csv'):
            with open(ann_path, 'r') as csvfile:
                dataset = list(csv.DictReader(csvfile))
        elif ann_path.endswith('.json'):
            dataset = json.load(open(ann_path))
        else:
            raise ValueError(f"Unsupported annotation file format: {ann_path}. Only .csv and .json files are supported.")

        self.data_root = data_root

        # It's used to balance num of images and videos.
        self.dataset = []
        for data in dataset:
            if data.get('type', 'image') != 'video':
                self.dataset.append(data)
        if video_repeat > 0:
            for _ in range(video_repeat):
                for data in dataset:
                    if data.get('type', 'image') == 'video':
                        self.dataset.append(data)
        del dataset

        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        # TODO: enable bucket training
        self.enable_bucket = enable_bucket
        self.text_drop_ratio = text_drop_ratio
        self.enable_inpaint  = enable_inpaint

        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end

        # Video params
        self.video_sample_stride    = video_sample_stride
        self.video_sample_n_frames  = video_sample_n_frames
        self.video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.video_transforms = transforms.Compose(
            [
                transforms.Resize(min(self.video_sample_size)),
                transforms.CenterCrop(self.video_sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

        # Image params
        self.image_sample_size  = tuple(image_sample_size) if not isinstance(image_sample_size, int) else (image_sample_size, image_sample_size)
        self.image_transforms   = transforms.Compose([
            transforms.Resize(min(self.image_sample_size)),
            transforms.CenterCrop(self.image_sample_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        ])

        self.larger_side_of_image_and_video = max(min(self.image_sample_size), min(self.video_sample_size))

    def get_batch(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        video_id, text = data_info['file_path'], data_info['text']

        if data_info.get('type', 'image')=='video':
            if self.data_root is None:
                video_dir = video_id
            else:
                video_dir = os.path.join(self.data_root, video_id)

            with VideoReader_contextmanager(video_dir, num_threads=2) as video_reader:
                min_sample_n_frames = min(
                    self.video_sample_n_frames, 
                    int(len(video_reader) * (self.video_length_drop_end - self.video_length_drop_start) // self.video_sample_stride)
                )
                if min_sample_n_frames == 0:
                    raise ValueError(f"No Frames in video.")

                video_length = int(self.video_length_drop_end * len(video_reader))
                clip_length = min(video_length, (min_sample_n_frames - 1) * self.video_sample_stride + 1)
                start_idx   = random.randint(int(self.video_length_drop_start * video_length), video_length - clip_length) if video_length != clip_length else 0
                batch_index = np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)

                try:
                    sample_args = (video_reader, batch_index)
                    pixel_values = func_timeout(
                        VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                    )
                    resized_frames = []
                    for i in range(len(pixel_values)):
                        frame = pixel_values[i]
                        resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                        resized_frames.append(resized_frame)
                    pixel_values = np.array(resized_frames)
                except FunctionTimedOut:
                    raise ValueError(f"Read {idx} timeout.")
                except Exception as e:
                    raise ValueError(f"Failed to extract frames from video. Error is {e}.")

                if not self.enable_bucket:
                    pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
                    pixel_values = pixel_values / 255.
                    del video_reader
                else:
                    pixel_values = pixel_values

                if not self.enable_bucket:
                    pixel_values = self.video_transforms(pixel_values)

                # Random use no text generation
                if random.random() < self.text_drop_ratio:
                    text = ''

            control_video_id = data_info['control_file_path']

            if self.data_root is None:
                control_video_id = control_video_id
            else:
                control_video_id = os.path.join(self.data_root, control_video_id)

            with VideoReader_contextmanager(control_video_id, num_threads=2) as control_video_reader:
                try:
                    sample_args = (control_video_reader, batch_index)
                    control_pixel_values = func_timeout(
                        VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                    )
                    resized_frames = []
                    for i in range(len(control_pixel_values)):
                        frame = control_pixel_values[i]
                        resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                        resized_frames.append(resized_frame)
                    control_pixel_values = np.array(resized_frames)
                except FunctionTimedOut:
                    raise ValueError(f"Read {idx} timeout.")
                except Exception as e:
                    raise ValueError(f"Failed to extract frames from video. Error is {e}.")

                if not self.enable_bucket:
                    control_pixel_values = torch.from_numpy(control_pixel_values).permute(0, 3, 1, 2).contiguous()
                    control_pixel_values = control_pixel_values / 255.
                    del control_video_reader
                else:
                    control_pixel_values = control_pixel_values

                if not self.enable_bucket:
                    control_pixel_values = self.video_transforms(control_pixel_values)
            return pixel_values, control_pixel_values, text, "video"
        else:
            image_path, text = data_info['file_path'], data_info['text']
            if self.data_root is not None:
                image_path = os.path.join(self.data_root, image_path)
            image = Image.open(image_path).convert('RGB')
            if not self.enable_bucket:
                image = self.image_transforms(image).unsqueeze(0)
            else:
                image = np.expand_dims(np.array(image), 0)

            if random.random() < self.text_drop_ratio:
                text = ''

            control_image_id = data_info['control_file_path']

            if self.data_root is None:
                control_image_id = control_image_id
            else:
                control_image_id = os.path.join(self.data_root, control_image_id)

            control_image = Image.open(control_image_id).convert('RGB')
            if not self.enable_bucket:
                control_image = self.image_transforms(control_image).unsqueeze(0)
            else:
                control_image = np.expand_dims(np.array(control_image), 0)
            return image, control_image, text, 'image'

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        data_type = data_info.get('type', 'image')
        while True:
            sample = {}
            try:
                data_info_local = self.dataset[idx % len(self.dataset)]
                data_type_local = data_info_local.get('type', 'image')
                if data_type_local != data_type:
                    raise ValueError("data_type_local != data_type")

                pixel_values, control_pixel_values, name, data_type = self.get_batch(idx)
                sample["pixel_values"] = pixel_values
                sample["control_pixel_values"] = control_pixel_values
                sample["text"] = name
                sample["data_type"] = data_type
                sample["idx"] = idx

                if len(sample) > 0:
                    break
            except Exception as e:
                print(e, self.dataset[idx % len(self.dataset)])
                idx = random.randint(0, self.length-1)

        if self.enable_inpaint and not self.enable_bucket:
            mask = get_random_mask(pixel_values.size())
            mask_pixel_values = pixel_values * (1 - mask) + torch.ones_like(pixel_values) * -1 * mask
            sample["mask_pixel_values"] = mask_pixel_values
            sample["mask"] = mask

            clip_pixel_values = sample["pixel_values"][0].permute(1, 2, 0).contiguous()
            clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
            sample["clip_pixel_values"] = clip_pixel_values

            ref_pixel_values = sample["pixel_values"][0].unsqueeze(0)
            if (mask == 1).all():
                ref_pixel_values = torch.ones_like(ref_pixel_values) * -1
            sample["ref_pixel_values"] = ref_pixel_values

        return sample
