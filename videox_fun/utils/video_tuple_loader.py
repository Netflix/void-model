import gc
import glob
import os
import random
from contextlib import contextmanager

import cv2
import mediapy as media
import numpy as np
from decord import VideoReader
from func_timeout import FunctionTimedOut, func_timeout

VIDEO_READER_TIMEOUT = 20
VIDEO_READER_NUM_THREADS = 2


@contextmanager
def _video_reader_contextmanager(*args, **kwargs):
    video_reader = VideoReader(*args, **kwargs)
    try:
        yield video_reader
    finally:
        del video_reader
        gc.collect()


def _get_video_reader_batch(video_reader, batch_index):
    return video_reader.get_batch(batch_index).asnumpy()


def _resize_frame(frame, target_short_side):
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

    return cv2.resize(frame, (new_w, new_h))


def _resize_clip(frames, target_short_side):
    return np.array([_resize_frame(frame, target_short_side) for frame in frames])


def _sample_batch_index(
    total_frames,
    video_sample_n_frames,
    video_length_drop_start,
    video_length_drop_end,
    video_sample_stride,
):
    min_sample_n_frames = min(
        video_sample_n_frames,
        int(total_frames * (video_length_drop_end - video_length_drop_start) // video_sample_stride),
    )
    if min_sample_n_frames == 0:
        raise ValueError("No frames available after sampling constraints.")

    video_length = int(video_length_drop_end * total_frames)
    clip_length = min(video_length, (min_sample_n_frames - 1) * video_sample_stride + 1)
    start_idx = (
        random.randint(int(video_length_drop_start * video_length), video_length - clip_length)
        if video_length != clip_length
        else 0
    )
    return np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)


def _read_resized_video_clip(video_path, batch_index, target_short_side):
    with _video_reader_contextmanager(video_path, num_threads=VIDEO_READER_NUM_THREADS) as video_reader:
        return _read_resized_video_reader_clip(video_reader, batch_index, target_short_side, video_path)


def _read_resized_video_reader_clip(video_reader, batch_index, target_short_side, source_name):
    try:
        frames = func_timeout(
            VIDEO_READER_TIMEOUT,
            _get_video_reader_batch,
            args=(video_reader, batch_index),
        )
        return _resize_clip(frames, target_short_side)
    except FunctionTimedOut as exc:
        raise ValueError(f"Read timeout while sampling frames from {source_name}.") from exc
    except Exception as exc:
        raise ValueError(f"Failed to extract frames from {source_name}. Error is {exc}.") from exc


def _get_frame_paths(frame_dir):
    frame_paths = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    if not frame_paths:
        raise ValueError(f"No PNG files found in directory: {frame_dir}")
    return frame_paths


def _read_resized_frame_dir_clip(frame_dir, batch_index, target_short_side):
    frame_paths = _get_frame_paths(frame_dir)
    try:
        selected_frames = [media.read_image(frame_paths[idx]) for idx in batch_index]
    except IndexError as exc:
        raise ValueError(
            f"Frame selection for {frame_dir} is out of range. Requested up to index "
            f"{int(np.max(batch_index))} but only found {len(frame_paths)} frames."
        ) from exc

    return _resize_clip(selected_frames, target_short_side)


def load_video_mask_tuple_clip(
    sample_dir,
    video_sample_n_frames,
    video_length_drop_start,
    video_length_drop_end,
    video_sample_stride,
    target_short_side,
):
    mp4_input_path = os.path.join(sample_dir, "rgb_full.mp4")

    if os.path.exists(mp4_input_path):
        with _video_reader_contextmanager(mp4_input_path, num_threads=VIDEO_READER_NUM_THREADS) as video_reader:
            batch_index = _sample_batch_index(
                total_frames=len(video_reader),
                video_sample_n_frames=video_sample_n_frames,
                video_length_drop_start=video_length_drop_start,
                video_length_drop_end=video_length_drop_end,
                video_sample_stride=video_sample_stride,
            )
            input_video = _read_resized_video_reader_clip(
                video_reader,
                batch_index,
                target_short_side,
                mp4_input_path,
            )

        target_video = _read_resized_video_clip(
            os.path.join(sample_dir, "rgb_removed.mp4"),
            batch_index,
            target_short_side,
        )
        mask_video = _read_resized_video_clip(
            os.path.join(sample_dir, "mask.mp4"),
            batch_index,
            target_short_side,
        )

        depth_video_path = os.path.join(sample_dir, "depth_removed.mp4")
        depth_video = None
        if os.path.exists(depth_video_path):
            depth_video = _read_resized_video_clip(depth_video_path, batch_index, target_short_side)
    else:
        input_dir = os.path.join(sample_dir, "input")
        input_frame_paths = _get_frame_paths(input_dir)
        batch_index = _sample_batch_index(
            total_frames=len(input_frame_paths),
            video_sample_n_frames=video_sample_n_frames,
            video_length_drop_start=video_length_drop_start,
            video_length_drop_end=video_length_drop_end,
            video_sample_stride=video_sample_stride,
        )
        input_video = _resize_clip(
            [media.read_image(input_frame_paths[idx]) for idx in batch_index],
            target_short_side,
        )
        target_video = _read_resized_frame_dir_clip(
            os.path.join(sample_dir, "bg"),
            batch_index,
            target_short_side,
        )
        mask_video = _read_resized_frame_dir_clip(
            os.path.join(sample_dir, "trimask"),
            batch_index,
            target_short_side,
        )
        depth_video = None

    return input_video, target_video, mask_video, depth_video
