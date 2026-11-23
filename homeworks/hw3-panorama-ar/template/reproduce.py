# -*- coding: utf-8 -*-
"""Рендер собственных серий HW3 и расчёт чисел для REPORT.md.

Камера в этой среде недоступна, поэтому кадры — плоская сцена и сцена с
параллаксом, собранные NumPy/OpenCV при seed=4. Команды пайплайна те же,
что для фотографий.
"""
import json
import platform
import sys
from pathlib import Path

import cv2
import numpy as np

import panorama as hw

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUT = ROOT / 'out'
SEED = 4


def texture(width, height, seed):
    rng = np.random.default_rng(seed)
    image = np.zeros((height, width, 3), np.uint8)
    image[:] = rng.integers(30, 80, size=3, dtype=np.uint8)
    yy, xx = np.mgrid[:height, :width]
    image[..., 0] = np.clip(image[..., 0] + (40 * xx / width).astype(np.uint8), 0, 255)
    image[..., 1] = np.clip(image[..., 1] + (50 * yy / height).astype(np.uint8), 0, 255)
    for _ in range(80):
        center = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        radius = int(rng.integers(8, 40))
        color = tuple(int(v) for v in rng.integers(0, 255, size=3))
        cv2.circle(image, center, radius, color, -1)
    for _ in range(40):
        p1 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        p2 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        color = tuple(int(v) for v in rng.integers(0, 255, size=3))
        cv2.line(image, p1, p2, color, int(rng.integers(1, 4)))
    # Мелкий шахматный растр даёт ложные соответствия: SIFT сходится, но сдвиг кратен клетке.
    for index in range(30):
        origin = (int(rng.integers(10, width - 180)), int(rng.integers(40, height - 40)))
        cv2.putText(image, 'P%d-%d' % (seed, index), origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, 'IU CV HW3', (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (240, 240, 240), 3)
    return image


def crop_homography(canvas, angle_deg, center, size):
    width, height = size
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    angle = np.deg2rad(angle_deg)
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    to_origin = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], float)
    to_canvas = np.array([[1, 0, center[0]], [0, 1, center[1]], [0, 0, 1]], float)
    frame_to_canvas = to_canvas @ rotation @ to_origin
    frame = cv2.warpPerspective(canvas, np.linalg.inv(frame_to_canvas), (width, height),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return frame


def save_series(folder, frames):
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, frame in enumerate(frames):
        path = folder / ('frame%d.png' % index)
        hw.write_image(path, frame)
        paths.append(path)
    return paths


def pair_table(images):
    panorama, reports = hw.stitch(images, detector='sift', ratio=0.75, seed=SEED)
    return panorama, reports


def stitcher_result(images):
    tried = {}
    for mode, name in ((cv2.Stitcher_PANORAMA, 'panorama'), (cv2.Stitcher_SCANS, 'scans')):
        status, image = cv2.Stitcher_create(mode).stitch(images)
        tried[name] = int(status)
        if status == cv2.Stitcher_OK:
            return tried, image
    return tried, None


def corner_motion(reference, frames, alpha):
    previous = None
    tracks, found = [], 0
    for frame in frames:
        try:
            src, dst = hw.match_points(reference, frame, 'sift', 0.75)
            homography, _ = hw.ransac_homography(src, dst, iterations=150, seed=SEED)
            previous = hw.smooth_homography(previous, homography, hw.image_corners(reference), alpha)
            tracks.append(hw.project(hw.image_corners(reference), previous))
            found += 1
        except ValueError:
            previous = None
            tracks.append(None)
    steps = []
    for left, right in zip(tracks, tracks[1:]):
        if left is not None and right is not None:
            steps.append(np.linalg.norm(right - left, axis=1))
    if not steps:
        return dict(found=found, frames=len(frames), rms_px=None)
    stacked = np.concatenate(steps)
    return dict(found=found, frames=len(frames), rms_px=float(np.sqrt(np.mean(stacked ** 2))),
                samples=int(stacked.size))


def render_ar(reference, poster_box):
    rng = np.random.default_rng(SEED)
    height, width = 480, 640
    frames = []
    x0, y0, pw, ph = poster_box
    for index in range(24):
        scene = np.full((height, width, 3), 28, np.uint8)
        cv2.rectangle(scene, (20, 20), (width - 20, height - 20), (40, 42, 48), 2)
        if index >= 20:
            frames.append(scene)
            continue
        if index < 12:
            dx, dy, angle = float(rng.normal(0, 0.6)), float(rng.normal(0, 0.6)), float(rng.normal(0, 0.15))
        else:
            dx, dy, angle = (index - 12) * 5.0, (index - 12) * 1.5, 0.0
        matrix = np.array([[1, 0, x0 + dx], [0, 1, y0 + dy], [0, 0, 1]], float)
        if abs(angle) > 1e-6:
            center = np.array([[1, 0, -(pw - 1) / 2], [0, 1, -(ph - 1) / 2], [0, 0, 1]], float)
            back = np.array([[1, 0, (pw - 1) / 2 + x0 + dx], [0, 1, (ph - 1) / 2 + y0 + dy], [0, 0, 1]], float)
            rad = np.deg2rad(angle)
            rotation = np.array([[np.cos(rad), -np.sin(rad), 0], [np.sin(rad), np.cos(rad), 0], [0, 0, 1]], float)
            matrix = back @ rotation @ center
        placed = cv2.warpPerspective(reference, matrix, (width, height))
        mask = cv2.warpPerspective(np.full(reference.shape[:2], 255, np.uint8), matrix, (width, height))
        scene[mask > 0] = placed[mask > 0]
        noise = rng.normal(0, 2.0, scene.shape)
        frames.append(np.clip(scene.astype(np.float32) + noise, 0, 255).astype(np.uint8))
    return frames


def write_video(path, frames, fps=15):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (frames[0].shape[1], frames[0].shape[0]))
    if not writer.isOpened():
        raise RuntimeError('Не удалось записать ' + str(path))
    for frame in frames:
        writer.write(frame)
    writer.release()


def dlt_controls():
    rows = []
    samples = ROOT.parent / 'sample_data'
    for case in ('case1', 'case2', 'case3'):
        data = json.loads((samples / case / 'correspondences.json').read_text(encoding='utf-8'))
        src, dst = np.float64(data['src']), np.float64(data['dst'])
        mine = hw.dlt(src, dst)
        library, _ = cv2.findHomography(src, dst, method=0)
        delta = float(np.linalg.norm(mine / mine[2, 2] - library / library[2, 2]))
        rows.append(dict(case=case, frobenius=delta))
    return rows


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    canvas = texture(2200, 1100, SEED)
    hw.write_image(DATA / 'scene-plane.png', canvas)
    size = (520, 380)
    centers = [(430, 520), (700, 500), (980, 540), (1260, 510)]
    angles = [-6.0, -2.0, 3.0, 7.0]
    planar = [crop_homography(canvas, angle, center, size) for angle, center in zip(angles, centers)]
    planar_paths = save_series(DATA / 'planar', planar)
    panorama, reports = pair_table(planar)
    hw.write_image(OUT / 'panorama-planar.png', panorama)
    status, stitched = stitcher_result(planar)
    if stitched is not None:
        hw.write_image(OUT / 'panorama-planar-stitcher.png', stitched)

    background = texture(1600, 500, SEED + 1)
    foreground = texture(360, 220, SEED + 2)
    parallax = []
    for index, shift in enumerate((0, 36, 72, 108)):
        view = background[:, 80 + shift:80 + shift + 520].copy()
        fg = cv2.resize(foreground, (220, 140))
        x, y = 40 + shift * 2, 160
        view[y:y + fg.shape[0], x:x + fg.shape[1]] = fg
        parallax.append(view)
    parallax_paths = save_series(DATA / 'parallax', parallax)
    try:
        parallax_image, parallax_reports = pair_table(parallax)
        hw.write_image(OUT / 'panorama-parallax.png', parallax_image)
        parallax_error = None
    except ValueError as exc:
        parallax_reports, parallax_error = [], str(exc)
    parallax_status, parallax_stitched = stitcher_result(parallax)
    if parallax_stitched is not None:
        hw.write_image(OUT / 'panorama-parallax-stitcher.png', parallax_stitched)

    reference = canvas[180:180 + 240, 200:200 + 320].copy()
    cv2.rectangle(reference, (8, 8), (311, 231), (0, 220, 255), 4)
    texture_image = np.zeros_like(reference)
    texture_image[:] = (40, 90, 200)
    cv2.putText(texture_image, 'AR', (90, 140), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 6)
    hw.write_image(DATA / 'reference.png', reference)
    hw.write_image(DATA / 'texture.png', texture_image)
    ar_frames = render_ar(reference, (160, 120, reference.shape[1], reference.shape[0]))
    write_video(DATA / 'ar-input.mp4', ar_frames)
    stationary = ar_frames[:12]
    motion = {}
    for alpha in (1.0, 0.4):
        motion['alpha_%s' % alpha] = corner_motion(reference, stationary, alpha)
        stats = hw.ar_video(reference, texture_image, str(DATA / 'ar-input.mp4'),
                            str(OUT / ('ar-alpha-%s.mp4' % alpha)), alpha=alpha)
        motion['alpha_%s' % alpha]['pipeline'] = stats
    summary = dict(
        seed=SEED,
        python=sys.version.split()[0],
        opencv=cv2.__version__,
        platform=platform.platform(),
        processor=platform.processor(),
        planar=dict(paths=[str(p.relative_to(ROOT)) for p in planar_paths], shape=list(panorama.shape),
                    angles_deg=angles, reports=reports, stitcher_status=status,
                    stitcher_shape=None if stitched is None else list(stitched.shape)),
        parallax=dict(paths=[str(p.relative_to(ROOT)) for p in parallax_paths], reports=parallax_reports,
                      error=parallax_error, stitcher_status=parallax_status,
                      stitcher_shape=None if parallax_stitched is None else list(parallax_stitched.shape)),
        dlt=dlt_controls(),
        ar=motion,
        detector='sift', ratio=0.75, ransac_threshold_px=3.0, ransac_iterations=500,
    )
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
