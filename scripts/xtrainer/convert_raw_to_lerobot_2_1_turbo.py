#!/usr/bin/env python
"""Fast variant of ``convert_raw_to_lerobot_2_1.py`` producing the same LeRobot v2.1 tree.

Same raw layout, same CLI and same output as the original script. The differences are
purely about *how* the work is done:

* Raw JPEGs are decoded once into RAM. No PNG round trip (write / verify / re-read) is
  needed for video encoding or for image statistics.
* Frames are fed straight from memory to the same libx264/libx265/SVT-AV1 encoder and
  options that ``lerobot.datasets.video_utils.encode_video_frames`` would use, so the MP4
  files are bit-identical on the same machine with the same ``--encoder-threads``.
* Image statistics reproduce ``lerobot.datasets.compute_stats.sample_images`` on the
  in-memory frames (same sampling indices, same downsampling), so ``meta/*.json*`` match.
* Episodes are processed in parallel worker processes (``--workers``); the three cameras
  of an episode are encoded concurrently in threads. Episode/frame indices are assigned by
  the parent process in the original sorted order, so numbering never changes.
* PNGs are only written when they are part of the output (``--no-videos`` or
  ``--keep-images-for-video``).

The generated dataset can be passed directly to ``scripts/xtrainer/validate_dataset_v21.py``
and ``train_xvla.sh``.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import pickle
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path

import av
import cv2
import datasets
import numpy as np
from PIL import Image

from lerobot.configs import RGBEncoderConfig
from lerobot.datasets.compute_stats import (
    aggregate_stats,
    auto_downsample_height_width,
    compute_episode_stats,
    get_feature_stats,
    sample_indices,
)
from lerobot.datasets.feature_utils import get_hf_features_from_features
from lerobot.datasets.io_utils import embed_images
from lerobot.datasets.utils import serialize_dict
from lerobot.utils.constants import DEFAULT_FEATURES


CODEBASE_VERSION = "v2.1"
CHUNK_SIZE = 1000
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
# Same naming as ``encode_video_frames`` expects (``frame-000000.png``).
TEMP_IMAGE_PATH = "images/{image_key}/episode_{episode_index:06d}/frame-{frame_index:06d}.png"
TMP_DIR_NAME = ".turbo_tmp"
CAMERAS = {
    "observation.images.top": "topImg",
    "observation.images.left_wrist": "leftImg",
    "observation.images.right_wrist": "rightImg",
}
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


# --------------------------------------------------------------------------------------
# Small helpers shared with the original script
# --------------------------------------------------------------------------------------


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4, ensure_ascii=False)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _frame_id(path: Path) -> int | None:
    return int(path.stem) if path.stem.isdigit() else None


def _frame_ids(folder: Path, suffixes: tuple[str, ...]) -> list[int]:
    ids = {
        frame_id
        for suffix in suffixes
        for path in folder.glob(f"*{suffix}")
        if (frame_id := _frame_id(path)) is not None
    }
    return sorted(ids)


def _resolve_image_path(folder: Path, frame_id: int) -> Path:
    for suffix in IMAGE_SUFFIXES:
        path = folder / f"{frame_id}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"Image for frame {frame_id} not found in {folder}")


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Could not decode image: {path}")
    return np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _build_joint_names(dim: int) -> list[str]:
    if dim == 14:
        return (
            [f"left_joint{i}.pos" for i in range(1, 7)]
            + ["left_gripper.pos"]
            + [f"right_joint{i}.pos" for i in range(1, 7)]
            + ["right_gripper.pos"]
        )
    return [f"joint_{index}.pos" for index in range(dim)]


def _first_observation(episode_dirs: list[Path]) -> dict:
    for episode_dir in episode_dirs:
        for path in sorted((episode_dir / "observation").glob("*.pkl")):
            with path.open("rb") as file:
                payload = pickle.load(file)
            if "joint_positions" in payload and "control" in payload:
                return payload
    raise RuntimeError("No observation .pkl containing joint_positions and control was found.")


def _first_image_shape(episode_dirs: list[Path], raw_camera_dir: str) -> tuple[int, int, int]:
    for episode_dir in episode_dirs:
        image_dir = episode_dir / raw_camera_dir
        for path in sorted(image_dir.glob("*")) if image_dir.is_dir() else []:
            if path.suffix.lower() not in set(IMAGE_SUFFIXES):
                continue
            image = _read_rgb(path)
            return (int(image.shape[0]), int(image.shape[1]), 3)
    raise RuntimeError(f"No valid RGB image found for raw camera directory {raw_camera_dir!r}.")


def _build_features(
    action_dim: int, state_dim: int, image_shapes: dict[str, tuple[int, int, int]], use_videos: bool
) -> dict:
    if action_dim != state_dim:
        raise ValueError(f"Action/state dimensions differ: {action_dim} vs {state_dim}")
    names = _build_joint_names(action_dim)
    image_dtype = "video" if use_videos else "image"
    features = {
        "action": {"dtype": "float32", "shape": (action_dim,), "names": names},
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": names},
    }
    for key, shape in image_shapes.items():
        features[key] = {
            "dtype": image_dtype,
            "shape": shape,
            "names": ["height", "width", "channels"],
        }
    return features


def _episode_data_path(root: Path, episode_index: int) -> Path:
    return root / DATA_PATH.format(episode_chunk=episode_index // CHUNK_SIZE, episode_index=episode_index)


def _episode_video_path(root: Path, episode_index: int, camera_key: str) -> Path:
    return root / VIDEO_PATH.format(
        episode_chunk=episode_index // CHUNK_SIZE,
        episode_index=episode_index,
        video_key=camera_key,
    )


def _temporary_image_path(root: Path, episode_index: int, camera_key: str, frame_index: int) -> Path:
    return root / TEMP_IMAGE_PATH.format(
        image_key=camera_key, episode_index=episode_index, frame_index=frame_index
    )


def _write_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    Image.fromarray(rgb).save(temporary, format="PNG")
    temporary.replace(path)


def _write_verified_png(path: Path, rgb: np.ndarray, repair_retries: int) -> list[str]:
    messages: list[str] = []
    for attempt in range(repair_retries + 1):
        _write_png(path, rgb)
        decoded = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if decoded is not None:
            return messages
        if attempt < repair_retries:
            messages.append(f"[warn] Repair temporary image {attempt + 1}/{repair_retries}: {path}")
    raise OSError(f"Could not write a readable temporary image: {path}")


def _video_info(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {}
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    capture.release()
    codec = "".join(chr((fourcc >> (8 * index)) & 0xFF) for index in range(4)).strip()
    return {
        "video.height": height,
        "video.width": width,
        "video.codec": codec or "unknown",
        "video.pix_fmt": "unknown",
        "video.is_depth_map": False,
        "video.fps": int(round(fps)) if fps > 0 else 0,
        "video.channels": 3,
        "has_audio": False,
    }


# --------------------------------------------------------------------------------------
# Worker side: one raw episode -> decoded frames -> videos / PNGs / image stats
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkerConfig:
    action_dim: int
    state_dim: int
    fps: int
    vcodec: str
    use_videos: bool
    write_images: bool
    skip_bad_frames: bool
    min_frames: int
    encode_retries: int
    repair_retries: int
    encoder_threads: int | None
    quiet_encoder: bool
    decode_threads: int
    tmp_root: str


@dataclass(frozen=True)
class EpisodeJob:
    raw_episode: str
    frame_ids: tuple[int, ...]


@dataclass
class EpisodeResult:
    raw_name: str
    messages: list[str] = field(default_factory=list)
    skipped_bad: int = 0
    skip_reason: str | None = None
    actions: np.ndarray | None = None
    states: np.ndarray | None = None
    image_stats: dict[str, dict] = field(default_factory=dict)
    video_paths: dict[str, str] = field(default_factory=dict)
    image_dirs: dict[str, str] = field(default_factory=dict)


def _image_stats_from_frames(frames: list[np.ndarray]) -> dict:
    """Reproduce ``compute_episode_stats`` for an image/video feature on in-memory frames."""
    sampled_indices = sample_indices(len(frames))
    images = None
    for i, idx in enumerate(sampled_indices):
        img = np.transpose(frames[idx], (2, 0, 1))  # (H, W, C) -> (C, H, W), uint8
        img = auto_downsample_height_width(img)
        if images is None:
            images = np.empty((len(sampled_indices), *img.shape), dtype=img.dtype)
        images[i] = img
    stats = get_feature_stats(images, axis=(0, 2, 3), keepdims=True)
    return {k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in stats.items()}


def _encode_frames(
    frames: list[np.ndarray], video_path: Path, fps: int, encoder: RGBEncoderConfig, encoder_threads: int | None
) -> None:
    """Same encoder/options/pixel format as ``encode_video_frames``, fed from memory."""
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_options = encoder.get_codec_options(encoder_threads, as_strings=True)
    height, width = frames[0].shape[:2]
    with av.open(str(video_path), "w") as output:
        output_stream = output.add_stream(encoder.vcodec, fps, options=video_options)
        output_stream.pix_fmt = encoder.pix_fmt
        output_stream.width = width
        output_stream.height = height
        for rgb in frames:
            packet = output_stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24"))
            if packet:
                output.mux(packet)
        packet = output_stream.encode()
        if packet:
            output.mux(packet)
    if not video_path.exists():
        raise OSError(f"Video encoding did not work. File not found: {video_path}.")


def _encode_with_retries(
    frames: list[np.ndarray], video_path: Path, cfg: WorkerConfig, encoder: RGBEncoderConfig, label: str
) -> list[str]:
    messages: list[str] = []
    for attempt in range(cfg.encode_retries + 1):
        try:
            _encode_frames(frames, video_path, cfg.fps, encoder, cfg.encoder_threads)
            return messages
        except Exception as error:
            video_path.unlink(missing_ok=True)
            if attempt == cfg.encode_retries:
                raise RuntimeError(f"Video encoding failed for {label}: {error}") from error
            messages.append(f"[warn] Retry video encoding {attempt + 1}/{cfg.encode_retries}: {label}")
    return messages


def _process_episode(job: EpisodeJob, cfg: WorkerConfig) -> EpisodeResult:
    if cfg.quiet_encoder:
        logging.getLogger("libav").setLevel(logging.ERROR)
    cv2.setNumThreads(1)

    raw_episode = Path(job.raw_episode)
    observation_dir = raw_episode / "observation"
    camera_dirs = {key: raw_episode / name for key, name in CAMERAS.items()}
    result = EpisodeResult(raw_name=raw_episode.name)

    # 1) Observations (cheap, sequential) -------------------------------------------------
    candidates: list[tuple[int, np.ndarray, np.ndarray]] = []
    obs_errors: dict[int, Exception] = {}
    for raw_frame_id in job.frame_ids:
        try:
            with (observation_dir / f"{raw_frame_id}.pkl").open("rb") as file:
                observation = pickle.load(file)
            action = np.asarray(observation["control"], dtype=np.float32).reshape(-1)
            state = np.asarray(observation["joint_positions"], dtype=np.float32).reshape(-1)
            if action.shape != (cfg.action_dim,) or state.shape != (cfg.state_dim,):
                raise ValueError(
                    f"Expected action/state {cfg.action_dim}/{cfg.state_dim}; got {action.size}/{state.size}"
                )
            if not np.isfinite(action).all() or not np.isfinite(state).all():
                raise ValueError("Action or state contains NaN/Inf")
            candidates.append((raw_frame_id, action, state))
        except Exception as error:  # noqa: BLE001 - mirrors the original script
            obs_errors[raw_frame_id] = error

    # 2) Image decoding (parallel threads; cv2 releases the GIL) ---------------------------
    def decode(raw_frame_id: int) -> tuple[int, dict[str, np.ndarray] | None, Exception | None]:
        try:
            images = {
                key: _read_rgb(_resolve_image_path(camera_dir, raw_frame_id))
                for key, camera_dir in camera_dirs.items()
            }
            return raw_frame_id, images, None
        except Exception as error:  # noqa: BLE001
            return raw_frame_id, None, error

    decoded: dict[int, dict[str, np.ndarray]] = {}
    with ThreadPoolExecutor(max_workers=cfg.decode_threads) as pool:
        for raw_frame_id, images, error in pool.map(decode, [c[0] for c in candidates]):
            if error is not None:
                obs_errors[raw_frame_id] = error
            else:
                decoded[raw_frame_id] = images

    # 3) Merge in original frame order, applying the skip/fail semantics -------------------
    actions: list[np.ndarray] = []
    states: list[np.ndarray] = []
    frames: dict[str, list[np.ndarray]] = {key: [] for key in CAMERAS}
    valid = {c[0]: c for c in candidates}
    for raw_frame_id in job.frame_ids:
        error = obs_errors.get(raw_frame_id)
        if error is None and raw_frame_id in decoded:
            _, action, state = valid[raw_frame_id]
            actions.append(action)
            states.append(state)
            for key in CAMERAS:
                frames[key].append(decoded[raw_frame_id][key])
            continue
        result.skipped_bad += 1
        if not cfg.skip_bad_frames:
            raise RuntimeError(f"Invalid frame {raw_episode.name}/{raw_frame_id}") from error
        result.messages.append(f"[warn] Skip invalid frame {raw_episode.name}/{raw_frame_id}: {error}")
    decoded.clear()

    if len(actions) < cfg.min_frames:
        result.skip_reason = f"[skip] Too few valid frames after filtering ({len(actions)}): {raw_episode.name}"
        return result

    result.actions = np.stack(actions).astype(np.float32)
    result.states = np.stack(states).astype(np.float32)
    tmp_dir = Path(cfg.tmp_root) / raw_episode.name

    # 4) PNGs only if they are part of the output ------------------------------------------
    if cfg.write_images:
        with ThreadPoolExecutor(max_workers=cfg.decode_threads) as pool:
            futures = []
            for key, camera_frames in frames.items():
                image_dir = tmp_dir / "images" / key
                result.image_dirs[key] = str(image_dir)
                for frame_index, rgb in enumerate(camera_frames):
                    path = image_dir / f"frame-{frame_index:06d}.png"
                    futures.append(pool.submit(_write_verified_png, path, rgb, cfg.repair_retries))
            for future in futures:
                result.messages.extend(future.result())

    # 5) Image statistics (identical to compute_episode_stats on the PNGs) ------------------
    for key, camera_frames in frames.items():
        result.image_stats[key] = _image_stats_from_frames(camera_frames)

    # 6) Video encoding, the three cameras concurrently -------------------------------------
    if cfg.use_videos:
        encoder = RGBEncoderConfig(vcodec=cfg.vcodec)
        with ThreadPoolExecutor(max_workers=len(CAMERAS)) as pool:
            futures = {}
            for key, camera_frames in frames.items():
                video_path = tmp_dir / f"{key}.mp4"
                result.video_paths[key] = str(video_path)
                futures[key] = pool.submit(
                    _encode_with_retries, camera_frames, video_path, cfg, encoder, f"{raw_episode.name}/{key}"
                )
            for key in CAMERAS:  # deterministic message order
                result.messages.extend(futures[key].result())
    return result


# --------------------------------------------------------------------------------------
# Parent side: scheduling, index assignment, parquet + metadata
# --------------------------------------------------------------------------------------


def _iter_results(jobs: list[EpisodeJob], cfg: WorkerConfig, workers: int, max_restarts: int):
    """Yield one ``EpisodeResult`` per job, in job order.

    With ``workers == 0`` everything runs in this process. Otherwise a native crash inside a
    worker (FFmpeg/codec) breaks the pool; the pool is rebuilt and the remaining jobs are
    resubmitted, at most ``max_restarts`` times for the job that was being waited on.
    """
    if workers == 0:
        for job in jobs:
            yield _process_episode(job, cfg)
        return

    next_index = 0
    restarts = [0] * len(jobs)
    context = mp.get_context("spawn")
    while next_index < len(jobs):
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=context)
        futures = []
        try:
            futures = [pool.submit(_process_episode, job, cfg) for job in jobs[next_index:]]
            try:
                for future in futures:
                    result = future.result()
                    next_index += 1
                    yield result
            except BrokenProcessPool:
                restarts[next_index] += 1
                if restarts[next_index] > max_restarts:
                    raise RuntimeError(
                        f"Worker pool crashed repeatedly while processing {jobs[next_index].raw_episode}."
                    ) from None
                print(
                    f"[warn] Worker pool crashed; restarting from {Path(jobs[next_index].raw_episode).name} "
                    f"({restarts[next_index]}/{max_restarts})"
                )
        finally:
            # Do not wait for queued episodes when the parent stops early (error / crash restart).
            for future in futures:
                future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)


def convert(args: argparse.Namespace) -> None:
    raw_root = Path(args.raw_root)
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw root does not exist or is not a directory: {raw_root}")
    episode_dirs = sorted(path for path in raw_root.iterdir() if path.is_dir())
    if not episode_dirs:
        raise RuntimeError(f"No episode directories found under {raw_root}")

    first_observation = _first_observation(episode_dirs)
    action_dim = int(np.asarray(first_observation["control"]).reshape(-1).shape[0])
    state_dim = int(np.asarray(first_observation["joint_positions"]).reshape(-1).shape[0])
    image_shapes = {key: _first_image_shape(episode_dirs, name) for key, name in CAMERAS.items()}
    features = {**_build_features(action_dim, state_dim, image_shapes, args.use_videos), **DEFAULT_FEATURES}
    hf_features = get_hf_features_from_features(features)

    output_root = Path(args.output_root)
    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite_output:
            raise FileExistsError(
                f"Output root is non-empty: {output_root}. Pass --overwrite-output to replace it."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    tmp_root = output_root / TMP_DIR_NAME
    tmp_root.mkdir(parents=True, exist_ok=True)

    # Cheap, sequential pre-checks (no decoding) keep the original skip messages and order.
    entries: list[EpisodeJob | str] = []
    for raw_episode in episode_dirs:
        observation_dir = raw_episode / "observation"
        camera_dirs = {key: raw_episode / name for key, name in CAMERAS.items()}
        if not observation_dir.is_dir() or not all(path.is_dir() for path in camera_dirs.values()):
            entries.append(f"[skip] Missing observation or camera directory: {raw_episode}")
            continue
        ids = set(_frame_ids(observation_dir, (".pkl",)))
        for camera_dir in camera_dirs.values():
            ids &= set(_frame_ids(camera_dir, IMAGE_SUFFIXES))
        frame_ids = sorted(ids)[args.skip_first_frames :]
        if len(frame_ids) < args.min_frames:
            entries.append(f"[skip] Too few aligned frames ({len(frame_ids)}): {raw_episode.name}")
            continue
        entries.append(EpisodeJob(raw_episode=str(raw_episode), frame_ids=tuple(frame_ids)))

    jobs = [entry for entry in entries if isinstance(entry, EpisodeJob)]
    cfg = WorkerConfig(
        action_dim=action_dim,
        state_dim=state_dim,
        fps=args.fps,
        vcodec=args.vcodec,
        use_videos=args.use_videos,
        write_images=(not args.use_videos) or args.keep_images_for_video,
        skip_bad_frames=args.skip_bad_frames,
        min_frames=args.min_frames,
        encode_retries=args.encode_retries,
        repair_retries=args.repair_retries,
        encoder_threads=args.encoder_threads,
        quiet_encoder=args.quiet_encoder,
        decode_threads=args.decode_threads,
        tmp_root=str(tmp_root),
    )
    print(
        f"[info] {len(jobs)} episodes queued; workers={args.workers}, decode_threads={args.decode_threads}, "
        f"encoder_threads={args.encoder_threads or 'auto'}, vcodec={args.vcodec}"
    )

    all_stats: list[dict] = []
    episode_rows: list[dict] = []
    episode_stats_rows: list[dict] = []
    total_frames = 0
    total_videos = 0
    saved_episodes = 0

    try:
        results = _iter_results(jobs, cfg, args.workers, args.encode_retries)
        for entry in entries:
            if isinstance(entry, str):
                print(entry)
                continue
            result = next(results)
            for message in result.messages:
                print(message)
            if result.skip_reason:
                print(result.skip_reason)
                _discard_tmp(tmp_root / result.raw_name)
                continue

            episode_index = saved_episodes
            length = int(result.actions.shape[0])

            image_paths: dict[str, list[str]] = {}
            for camera_key in CAMERAS:
                tmp_image_dir = result.image_dirs.get(camera_key)
                if tmp_image_dir is None:
                    continue
                final_dir = _temporary_image_path(output_root, episode_index, camera_key, 0).parent
                final_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(tmp_image_dir, final_dir)
                image_paths[camera_key] = [
                    str(_temporary_image_path(output_root, episode_index, camera_key, i)) for i in range(length)
                ]

            episode_buffer = {
                "action": result.actions,
                "observation.state": result.states,
                "frame_index": np.arange(length, dtype=np.int64),
                "timestamp": np.arange(length, dtype=np.float32) / float(args.fps),
                "episode_index": np.full(length, episode_index, dtype=np.int64),
                "index": np.arange(total_frames, total_frames + length, dtype=np.int64),
                "task_index": np.zeros(length, dtype=np.int64),
                **image_paths,
            }
            parquet_payload = {key: episode_buffer[key] for key in hf_features}
            episode_dataset = datasets.Dataset.from_dict(parquet_payload, features=hf_features, split="train")
            if not args.use_videos:
                episode_dataset = embed_images(episode_dataset)
            data_path = _episode_data_path(output_root, episode_index)
            data_path.parent.mkdir(parents=True, exist_ok=True)
            episode_dataset.to_parquet(data_path)

            numeric_buffer = {key: value for key, value in episode_buffer.items() if key not in CAMERAS}
            stats = compute_episode_stats(numeric_buffer, features)
            for camera_key in CAMERAS:  # same key order as the original episode buffer
                stats[camera_key] = result.image_stats[camera_key]
            all_stats.append(stats)
            episode_rows.append({"episode_index": episode_index, "tasks": [args.task], "length": length})
            episode_stats_rows.append({"episode_index": episode_index, "stats": serialize_dict(stats)})

            if args.use_videos:
                for camera_key in CAMERAS:
                    video_path = _episode_video_path(output_root, episode_index, camera_key)
                    video_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(result.video_paths[camera_key], video_path)
                    features[camera_key]["info"] = _video_info(video_path)
                    total_videos += 1
            _discard_tmp(tmp_root / result.raw_name)

            total_frames += length
            saved_episodes += 1
            print(
                f"[ok] Saved episode {saved_episodes}: {result.raw_name} "
                f"({length} frames; skipped={result.skipped_bad})"
            )
    finally:
        _discard_tmp(tmp_root)

    if not saved_episodes:
        raise RuntimeError("No episodes were converted; inspect the raw layout and frame files.")

    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": args.robot_type,
        "total_episodes": saved_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_videos,
        "total_chunks": (saved_episodes + CHUNK_SIZE - 1) // CHUNK_SIZE,
        "chunks_size": CHUNK_SIZE,
        "fps": args.fps,
        "splits": {"train": f"0:{saved_episodes}"},
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH if args.use_videos else None,
        "features": features,
    }
    _write_json(output_root / "meta/info.json", info)
    _write_json(output_root / "meta/stats.json", serialize_dict(aggregate_stats(all_stats)))
    _write_jsonl(output_root / "meta/tasks.jsonl", [{"task_index": 0, "task": args.task}])
    _write_jsonl(output_root / "meta/episodes.jsonl", episode_rows)
    _write_jsonl(output_root / "meta/episodes_stats.jsonl", episode_stats_rows)
    print(f"Done. Converted {saved_episodes} episodes to: {output_root}")


def _discard_tmp(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw Dobot X-trainer recordings to LeRobot v2.1 (parallel, in-memory variant)."
    )
    parser.add_argument("--raw-root", "--raw_root", dest="raw_root", required=True, help="Raw collect_data directory.")
    parser.add_argument(
        "--output-root",
        "--output_root",
        dest="output_root",
        required=True,
        help="Destination LeRobot Dataset v2.1 directory.",
    )
    parser.add_argument("--repo-id", "--repo_id", dest="repo_id", default="local/dobot_xtrainer_converted_v21")
    parser.add_argument("--task", default="Insert the test tube on the desktop into the rack.")
    parser.add_argument("--robot-type", "--robot_type", dest="robot_type", default="dobot_xtrainer_bimanual")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--use-videos",
        "--use_videos",
        dest="use_videos",
        action="store_true",
        help="Encode camera frames as MP4 (default).",
    )
    parser.add_argument(
        "--no-videos", "--no_videos", dest="use_videos", action="store_false", help="Embed images in Parquet instead."
    )
    parser.set_defaults(use_videos=True)
    parser.add_argument("--vcodec", default="h264", choices=["h264", "hevc", "libsvtav1"])
    parser.add_argument("--repair-retries", "--repair_retries", dest="repair_retries", type=int, default=2)
    parser.add_argument("--encode-retries", "--encode_retries", dest="encode_retries", type=int, default=1)
    parser.add_argument(
        "--encode-in-subprocess",
        "--encode_in_subprocess",
        dest="encode_in_subprocess",
        action="store_true",
        help="Accepted for CLI compatibility; encoding always runs inside the worker processes.",
    )
    parser.add_argument("--encode-in-process", "--encode_in_process", dest="encode_in_subprocess", action="store_false")
    parser.set_defaults(encode_in_subprocess=True)
    parser.add_argument("--skip-first-frames", "--skip_first_frames", dest="skip_first_frames", type=int, default=0)
    parser.add_argument("--min-frames", "--min_frames", dest="min_frames", type=int, default=10)
    parser.add_argument(
        "--skip-bad-frames",
        "--skip_bad_frames",
        dest="skip_bad_frames",
        action="store_true",
        help="Skip corrupt frames (default).",
    )
    parser.add_argument("--fail-on-bad-frames", "--fail_on_bad_frames", dest="skip_bad_frames", action="store_false")
    parser.set_defaults(skip_bad_frames=True)
    parser.add_argument(
        "--keep-images-for-video",
        "--keep_images_for_video",
        dest="keep_images_for_video",
        action="store_true",
        help="Also write the per-frame PNGs next to the MP4s (as the original script would leave them).",
    )
    parser.add_argument(
        "--quiet-encoder",
        "--quiet_encoder",
        dest="quiet_encoder",
        action="store_true",
        help="Hide encoder output (default).",
    )
    parser.add_argument("--verbose-encoder", "--verbose_encoder", dest="quiet_encoder", action="store_false")
    parser.set_defaults(quiet_encoder=True)
    parser.add_argument(
        "--overwrite-output",
        "--overwrite_output",
        dest="overwrite_output",
        action="store_true",
        help="Replace a non-empty output directory.",
    )
    # Turbo-only knobs. Defaults keep the output identical to the original script.
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="Episode-level worker processes. 0 runs everything in this process (debugging).",
    )
    parser.add_argument(
        "--decode-threads",
        "--decode_threads",
        dest="decode_threads",
        type=int,
        default=4,
        help="JPEG decode / PNG write threads per worker.",
    )
    parser.add_argument(
        "--encoder-threads",
        "--encoder_threads",
        dest="encoder_threads",
        type=int,
        default=None,
        help="Threads per video encoder. Default lets the codec decide (same as the original script). "
        "libx264 output only stays bit-identical for the same thread count; set e.g. 2 for throughput "
        "when running many workers.",
    )
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.min_frames <= 0:
        parser.error("--min-frames must be positive")
    if args.skip_first_frames < 0:
        parser.error("--skip-first-frames must be non-negative")
    if args.encode_retries < 0:
        parser.error("--encode-retries must be non-negative")
    if args.repair_retries < 0:
        parser.error("--repair-retries must be non-negative")
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    if args.decode_threads <= 0:
        parser.error("--decode-threads must be positive")
    if args.encoder_threads is not None and args.encoder_threads <= 0:
        parser.error("--encoder-threads must be positive")
    return args


if __name__ == "__main__":
    convert(parse_args())
