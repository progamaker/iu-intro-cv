# -*- coding: utf-8 -*-
"""Калибровка по моим фото доски, метрология и стерео-таблица HW4.

Если в data/boards уже лежит 01.jpg, скрипт считает эти кадры и не рисует
поверх них контрольную доску. Клетка 0.03 м. Стерео — пары курса.
"""
import json
import platform
import sys
from pathlib import Path

import cv2
import numpy as np

import calib3d as hw

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUT = ROOT / 'out'
SEED = 4
SQUARE_M = 0.03
COLUMNS, ROWS = 7, 5
IMAGE_SIZE = (960, 720)


def board_texture():
    squares_x, squares_y, px = COLUMNS + 1, ROWS + 1, 80
    image = np.zeros((squares_y * px, squares_x * px, 3), np.uint8)
    for y in range(squares_y):
        for x in range(squares_x):
            if (x + y) % 2 == 0:
                image[y * px:(y + 1) * px, x * px:(x + 1) * px] = 255
    return image, px


def pose_homography(rvec, tvec, camera):
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, np.float64))
    return camera @ np.column_stack([rotation[:, 0], rotation[:, 1], np.asarray(tvec, np.float64).reshape(3)])


def distort_image(ideal, camera, distortion):
    height, width = ideal.shape[:2]
    pixels = np.stack(np.meshgrid(np.arange(width), np.arange(height)), -1).reshape(-1, 1, 2).astype(np.float32)
    ideal_pixels = cv2.undistortPoints(pixels, camera, distortion, P=camera).reshape(height, width, 2).astype(np.float32)
    return cv2.remap(ideal, ideal_pixels[..., 0], ideal_pixels[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def render_board(texture, pixels_per_square, rvec, tvec, camera, distortion):
    scale = pixels_per_square / SQUARE_M
    texture_from_plane = np.array([[scale, 0, pixels_per_square], [0, scale, pixels_per_square], [0, 0, 1]], float)
    mapping = pose_homography(rvec, tvec, camera) @ np.linalg.inv(texture_from_plane)
    ideal = cv2.warpPerspective(texture, mapping, IMAGE_SIZE)
    return distort_image(ideal, camera, distortion)


def random_pose(rng):
    """Камера смотрит на доску: +Y доски направлен вниз кадра, иначе порядок углов переворачивается."""
    center = np.array([(COLUMNS - 1) * SQUARE_M / 2.0, (ROWS - 1) * SQUARE_M / 2.0, 0.0])
    base = np.diag([1.0, 1.0, -1.0])
    wobble, _ = cv2.Rodrigues(rng.normal(0.0, [0.28, 0.28, 0.18]))
    rotation = wobble @ base
    eye = center + np.array([rng.uniform(-0.04, 0.04), rng.uniform(-0.03, 0.03), rng.uniform(0.45, 0.85)])
    rvec, _ = cv2.Rodrigues(rotation)
    return rvec.ravel(), (-rotation @ eye)


def order_matches(image, rvec, tvec, camera, distortion):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCornersSB(gray, (COLUMNS, ROWS))
    if not found:
        return False, None
    projected, _ = cv2.projectPoints(hw.board_points(COLUMNS, ROWS, SQUARE_M), rvec, tvec, camera, distortion)
    error = np.linalg.norm(corners.reshape(-1, 2) - projected.reshape(-1, 2), axis=1).mean()
    return bool(error < 1.5), float(error)


def make_boards():
    folder = DATA / 'boards'
    if (folder / '01.png').exists() or (folder / '01.jpg').exists():
        raise SystemExit('data/boards уже содержит свои кадры, рендер их не затирает')
    texture, px = board_texture()
    rng = np.random.default_rng(SEED)
    camera = np.array([[1100.0, 0, 480.0], [0, 1100.0, 360.0], [0, 0, 1]], float)
    distortion = np.array([-0.21, 0.08, 0.001, -0.0008, 0.0])
    folder = DATA / 'boards'
    folder.mkdir(parents=True, exist_ok=True)
    outer = np.float64([[-SQUARE_M, -SQUARE_M, 0], [(COLUMNS) * SQUARE_M, -SQUARE_M, 0],
                         [(COLUMNS) * SQUARE_M, (ROWS) * SQUARE_M, 0], [-SQUARE_M, (ROWS) * SQUARE_M, 0]])
    kept, attempts = [], 0
    while len(kept) < 22 and attempts < 800:
        attempts += 1
        rvec, tvec = random_pose(rng)
        projected, _ = cv2.projectPoints(outer, rvec, tvec, camera, distortion)
        xy = projected.reshape(-1, 2)
        if xy[:, 0].min() < 24 or xy[:, 1].min() < 24 or xy[:, 0].max() > IMAGE_SIZE[0] - 24 or xy[:, 1].max() > IMAGE_SIZE[1] - 24:
            continue
        image = render_board(texture, px, rvec, tvec, camera, distortion)
        ok, error = order_matches(image, rvec, tvec, camera, distortion)
        if ok:
            kept.append((image, rvec, tvec, error))
    if len(kept) < 22:
        raise RuntimeError('Не удалось набрать 22 читаемые позы')
    # Две позы с нелинейным изгибом: детектор ещё находит сетку, RMS калибровки растёт.
    bent = []
    for extra in range(2):
        image, rvec, tvec, _ = kept[extra]
        flow = np.zeros_like(image, np.float32)
        yy, xx = np.mgrid[:image.shape[0], :image.shape[1]]
        map_x = (xx + 6 * np.sin(yy / 28.0 + extra)).astype(np.float32)
        map_y = (yy + 4 * np.sin(xx / 36.0)).astype(np.float32)
        bent_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        gray = cv2.cvtColor(bent_image, cv2.COLOR_BGR2GRAY)
        if cv2.findChessboardCornersSB(gray, (COLUMNS, ROWS))[0]:
            bent.append(bent_image)
    missed = []
    dark = np.zeros((*IMAGE_SIZE[::-1], 3), np.uint8)
    partial = kept[0][0].copy()
    partial[:, : partial.shape[1] // 2] = 0
    for name, image in (('miss-dark.png', dark), ('miss-partial.png', partial)):
        hw_path = folder / name
        cv2.imencode('.png', image)[1].tofile(hw_path)
        missed.append(name)
    names = []
    truth = []
    for index, (image, rvec, tvec, error) in enumerate(kept):
        name = 'board-%02d.png' % index
        cv2.imencode('.png', image)[1].tofile(folder / name)
        names.append(name)
        truth.append(dict(rvec=rvec.tolist(), tvec_m=tvec.tolist(), detection_px=error))
    for index, image in enumerate(bent):
        name = 'board-bent-%02d.png' % index
        cv2.imencode('.png', image)[1].tofile(folder / name)
        names.append(name)
    return camera, distortion, names, truth, missed


def calibration_tables(camera_true):
    objects, images, size, names = hw.detect_board(DATA / 'boards', COLUMNS, ROWS, SQUARE_M)
    full = hw.calibration_report(objects, images, size, None)
    errors = np.asarray(full['per_image_before_px'])
    # Хвост ошибок — более наклонные позы: 0.12 px отсекает их и оставляет больше трёх.
    threshold = 0.12
    cleaned = hw.calibration_report(objects, images, size, threshold)
    few = hw.calibrate(objects[:8], images[:8], size)
    many = hw.calibrate(objects[:16], images[:16], size)
    sample = cv2.cvtColor(hw.read_image(sorted((DATA / 'boards').glob('board-00.png'))[0]), cv2.COLOR_BGR2GRAY)
    undistorted = cv2.undistort(hw.read_image(DATA / 'boards' / 'board-00.png'), np.float64(cleaned['K']),
                                np.float64(cleaned['distortion']))
    cv2.imencode('.png', undistorted)[1].tofile(OUT / 'board-00-undistort.png')
    cv2.imencode('.png', hw.read_image(DATA / 'boards' / 'board-00.png'))[1].tofile(OUT / 'board-00-original.png')
    return dict(detected=len(names), image_size=[size[0], size[1]], true_K=camera_true.tolist(),
                threshold_px=threshold, before=pack(full), after=pack(cleaned),
                names=names, per_image=list(zip(names, errors.tolist())),
                subset8=pack_direct(few), subset16=pack_direct(many), sample_gray=int(sample.mean()))


def pack(report):
    k = report['K']
    return dict(poses=len(report['kept_indices']), rms=report['rms_after'],
                fx=k[0][0], fy=k[1][1], cx=k[0][2], cy=k[1][2], distortion=report['distortion'],
                kept=report['kept_indices'], per_image=report['per_image_after_px'])


def pack_direct(result):
    rms, k, dist, _, _, errors = result
    return dict(poses=len(errors), rms=float(rms), fx=float(k[0, 0]), fy=float(k[1, 1]),
                cx=float(k[0, 2]), cy=float(k[1, 2]), distortion=dist.ravel().tolist(),
                per_image=errors.tolist())


def measurements(camera_true, distortion_true, truth):
    image_path = DATA / 'boards' / 'board-00.png'
    gray = cv2.cvtColor(hw.read_image(image_path), cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCornersSB(gray, (COLUMNS, ROWS))
    if not found:
        raise RuntimeError('На кадре измерения доска не найдена')
    objects = hw.board_points(COLUMNS, ROWS, SQUARE_M)
    # Калибровка для позы — очищенный JSON, его пишет вызывающий код на диск.
    calibration = json.loads((OUT / 'camera-clean.json').read_text(encoding='utf-8'))
    camera = np.float64(calibration['K'])
    distortion = np.float64(calibration['distortion'])
    ok, rvec, tvec = cv2.solvePnP(objects, corners, camera, distortion)
    if not ok:
        raise RuntimeError('solvePnP не сошёлся')
    pose = dict(K=camera.tolist(), distortion=distortion.ravel().tolist(), rvec=rvec.ravel().tolist(),
                tvec_m=tvec.ravel().tolist(), board_origin_distance_m=float(np.linalg.norm(tvec)))
    (OUT / 'pose.json').write_text(json.dumps(pose, indent=2), encoding='utf-8')
    segments = [
        ('on-plane-15cm', [[0.02, 0.04, 0.0], [0.17, 0.04, 0.0]], 0.15),
        ('on-plane-9cm', [[0.03, 0.02, 0.0], [0.03, 0.11, 0.0]], 0.09),
        ('off-plane-15cm', [[0.02, 0.04, 0.0], [0.17, 0.04, 0.04]], 0.155241),
    ]
    rows = []
    r_true = np.float64(truth[0]['rvec'])
    t_true = np.float64(truth[0]['tvec_m'])
    for name, ends, length in segments:
        projected, _ = cv2.projectPoints(np.float64(ends), r_true, t_true, camera_true, distortion_true)
        spec = dict(K=pose['K'], distortion=pose['distortion'], rvec=pose['rvec'], tvec_m=pose['tvec_m'],
                    pixels=projected.reshape(-1, 2).tolist(), truth_length_m=length)
        path = DATA / ('measure-%s.json' % name)
        path.write_text(json.dumps(spec, indent=2), encoding='utf-8')
        xyz = hw.plane_measure(spec['pixels'], camera, distortion, rvec, tvec)
        measured = float(np.linalg.norm(xyz[1] - xyz[0]))
        rows.append(dict(name=name, truth_m=length, measured_m=measured,
                         error_pct=100 * abs(measured - length) / length,
                         distance_m=float(np.linalg.norm(tvec)), points_m=xyz.tolist()))
    marked = hw.read_image(image_path)
    for row in rows[:2]:
        spec = json.loads((DATA / ('measure-%s.json' % row['name'])).read_text(encoding='utf-8'))
        a, b = np.int32(np.rint(spec['pixels']))
        cv2.line(marked, tuple(a), tuple(b), (0, 0, 255), 2)
        cv2.circle(marked, tuple(a), 4, (0, 255, 0), -1)
        cv2.circle(marked, tuple(b), 4, (0, 255, 0), -1)
    cv2.imencode('.png', marked)[1].tofile(OUT / 'measurement-marks.png')
    return rows


def stereo_tables():
    samples = ROOT.parent / 'sample_data'
    configs = [(32, 5), (64, 5), (16, 5), (32, 3), (64, 7)]
    chosen = (32, 5)
    table = []
    for case in ('case1', 'case2', 'case3'):
        folder = samples / case
        left, right = hw.read_image(folder / 'left.png'), hw.read_image(folder / 'right.png')
        with np.load(folder / 'gt.npz', allow_pickle=False) as data:
            truth, valid, camera = data['disparity'], data['valid'], data['K']
            baseline, doffs = float(data['baseline_m']), float(data['doffs'])
        for num, block in configs:
            disparity = hw.sgbm(left, right, num, block)
            metrics = hw.disparity_metrics(disparity, truth, valid)
            depth = hw.stereo_depth(disparity, camera[0, 0], baseline, doffs)
            gt_depth = hw.stereo_depth(truth, camera[0, 0], baseline, doffs)
            both = valid & np.isfinite(depth) & np.isfinite(gt_depth)
            depth_rmse = float(np.sqrt(np.mean((depth[both] - gt_depth[both]) ** 2))) if both.any() else None
            entry = dict(case=case, num_disparities=num, block_size=block,
                         P1=8 * left.shape[2] * block ** 2, P2=32 * left.shape[2] * block ** 2,
                         **metrics, depth_rmse_m=depth_rmse,
                         gt_depth_m=float(np.nanmedian(gt_depth[valid])),
                         pred_depth_m=float(np.nanmedian(depth[both])) if both.any() else None)
            table.append(entry)
            if (num, block) == chosen:
                out = OUT / case
                out.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out / 'result.npz', disparity=disparity, depth_m=depth)
                hw.write_ply(out / 'cloud.ply', hw.point_cloud(depth, camera), left)
                (out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
                show = np.nan_to_num(disparity, nan=0.0)
                show = cv2.normalize(show, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                cv2.imencode('.png', cv2.applyColorMap(show, cv2.COLORMAP_TURBO))[1].tofile(out / 'disparity.png')
                depth_show = np.nan_to_num(depth, nan=0.0)
                depth_show = cv2.normalize(depth_show, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                cv2.imencode('.png', cv2.applyColorMap(depth_show, cv2.COLORMAP_MAGMA))[1].tofile(out / 'depth.png')
    return table


def sensitivity():
    focal, baseline, doffs = 520.0, 0.12, 0.0
    rows = []
    for disparity in (4.0, 8.0, 16.0, 32.0):
        for delta in (0.25, 0.5, 1.0):
            z = focal * baseline / (disparity + doffs)
            z2 = focal * baseline / (disparity + delta + doffs)
            formula = z ** 2 * delta / (focal * baseline)
            rows.append(dict(d_px=disparity, delta_d_px=delta, Z_m=z, abs_dZ_m=abs(z2 - z), formula_m=formula))
    for focal_try, baseline_try in ((400.0, 0.12), (520.0, 0.06), (800.0, 0.20)):
        z = focal_try * baseline_try / 8.0
        formula = z ** 2 * 0.5 / (focal_try * baseline_try)
        rows.append(dict(d_px=8.0, delta_d_px=0.5, Z_m=z, abs_dZ_m=None, formula_m=formula,
                         focal_px=focal_try, baseline_m=baseline_try))
    return rows


def own_photos():
    """Мои кадры 01–28. Контрольный рендер их не заменяет."""
    objects, images, size, names = hw.detect_board(DATA / 'boards', COLUMNS, ROWS, SQUARE_M)
    full = hw.calibration_report(objects, images, size, None)
    cleaned = hw.calibration_report(objects, images, size, 0.5)
    (OUT / 'camera.json').write_text(json.dumps(full, indent=2), encoding='utf-8')
    (OUT / 'camera-clean.json').write_text(json.dumps(cleaned, indent=2), encoding='utf-8')
    subset8 = pack_direct(hw.calibrate(objects[:8], images[:8], size))
    subset16 = pack_direct(hw.calibrate(objects[:16], images[:16], size))
    sample = hw.read_image(DATA / 'boards' / '01.jpg')
    undistorted = cv2.undistort(sample, np.float64(cleaned['K']), np.float64(cleaned['distortion']))
    cv2.imencode('.jpg', sample, [int(cv2.IMWRITE_JPEG_QUALITY), 85])[1].tofile(OUT / 'board-01-original.jpg')
    cv2.imencode('.jpg', undistorted, [int(cv2.IMWRITE_JPEG_QUALITY), 85])[1].tofile(OUT / 'board-01-undistort.jpg')
    for stale in ('board-00-original.png', 'board-00-undistort.png'):
        path = OUT / stale
        if path.exists():
            path.unlink()
    camera = np.float64(cleaned['K'])
    distortion = np.float64(cleaned['distortion'])
    measure_rows = []
    marked = None
    for name, truth in (('measure-on-plane-15cm.json', 0.15), ('measure-on-plane-9cm.json', 0.09)):
        spec = json.loads((DATA / name).read_text(encoding='utf-8'))
        xyz = hw.plane_measure(spec['pixels'], camera, distortion, np.float64(spec['rvec']), np.float64(spec['tvec_m']))
        measured = float(np.linalg.norm(xyz[1] - xyz[0]))
        measure_rows.append(dict(name=name, truth_m=truth, measured_m=measured,
                                 error_pct=100 * abs(measured - truth) / truth,
                                 distance_m=float(np.linalg.norm(spec['tvec_m'])),
                                 tz_m=float(spec['tvec_m'][2]), points_m=xyz.tolist()))
        frame = '26.jpg' if '15cm' in name else '27.jpg'
        image = hw.read_image(DATA / frame)
        a, b = np.int32(np.rint(spec['pixels']))
        cv2.line(image, tuple(a), tuple(b), (0, 0, 255), 3)
        cv2.circle(image, tuple(a), 8, (0, 255, 0), -1)
        cv2.circle(image, tuple(b), 8, (0, 255, 0), -1)
        cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])[1].tofile(OUT / ('measurement-%s.jpg' % frame[:2]))
        if marked is None:
            marked = image
    if marked is not None:
        small = cv2.resize(marked, (756, 1008))
        cv2.imencode('.png', small)[1].tofile(OUT / 'measurement-marks.png')
    pose_image = DATA / '26.jpg'
    gray = cv2.cvtColor(hw.read_image(pose_image), cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCornersSB(gray, (COLUMNS, ROWS))
    if not found:
        raise RuntimeError('На моём кадре 26.jpg доска не найдена')
    ok, rvec, tvec = cv2.solvePnP(hw.board_points(COLUMNS, ROWS, SQUARE_M), corners, camera, distortion)
    if not ok:
        raise RuntimeError('solvePnP не сошёлся на 26.jpg')
    pose = dict(K=camera.tolist(), distortion=distortion.ravel().tolist(), rvec=rvec.ravel().tolist(),
                tvec_m=tvec.ravel().tolist(), board_origin_distance_m=float(np.linalg.norm(tvec)))
    (OUT / 'pose.json').write_text(json.dumps(pose, indent=2), encoding='utf-8')
    errors = list(zip(names, full['per_image_before_px']))
    return dict(detected=len(names), missed_files=['20.jpg', '21.jpg', '22.jpg', '23.jpg'],
                image_size=[size[0], size[1]], threshold_px=0.5, before=pack(full), after=pack(cleaned),
                names=names, per_image=errors, subset8=subset8, subset16=subset16, measurements=measure_rows)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if (DATA / 'boards' / '01.jpg').exists():
        tables = own_photos()
        summary = dict(
            python=sys.version.split()[0], opencv=cv2.__version__,
            platform=platform.platform(), processor=platform.processor(),
            square_m=SQUARE_M, columns=COLUMNS, rows=ROWS, image_size=tables['image_size'],
            calibration=tables, measurements=tables['measurements'],
            stereo=stereo_tables(), sensitivity=sensitivity(),
            chosen_sgbm=dict(num_disparities=32, block_size=5, P1=8 * 3 * 25, P2=32 * 3 * 25),
        )
        (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
        print('boards', tables['detected'], 'threshold', tables['threshold_px'])
        print('rms', tables['before']['rms'], '->', tables['after']['rms'])
        print('subset8', tables['subset8']['rms'], tables['subset8']['fx'])
        print('subset16', tables['subset16']['rms'], tables['subset16']['fx'])
        print('measures', tables['measurements'])
        print('stereo chosen', [row for row in summary['stereo'] if row['num_disparities'] == 32 and row['block_size'] == 5])
        return
    camera, distortion, names, truth, missed = make_boards()
    tables = calibration_tables(camera)
    (OUT / 'camera.json').write_text(json.dumps(hw.calibration_report(
        *hw.detect_board(DATA / 'boards', COLUMNS, ROWS, SQUARE_M)[:3], None), indent=2), encoding='utf-8')
    objects, images, size, _ = hw.detect_board(DATA / 'boards', COLUMNS, ROWS, SQUARE_M)
    cleaned = hw.calibration_report(objects, images, size, tables['threshold_px'])
    (OUT / 'camera-clean.json').write_text(json.dumps(cleaned, indent=2), encoding='utf-8')
    measure_rows = measurements(camera, distortion, truth)
    summary = dict(
        seed=SEED, python=sys.version.split()[0], opencv=cv2.__version__,
        platform=platform.platform(), processor=platform.processor(),
        square_m=SQUARE_M, columns=COLUMNS, rows=ROWS, image_size=list(IMAGE_SIZE),
        true_distortion=distortion.tolist(), missed=missed, calibration=tables, measurements=measure_rows,
        stereo=stereo_tables(), sensitivity=sensitivity(),
        chosen_sgbm=dict(num_disparities=32, block_size=5, P1=8 * 3 * 25, P2=32 * 3 * 25),
    )
    # per_image пары не сериализуются, если там numpy; tables уже из списков.
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print('boards', tables['detected'], 'threshold', tables['threshold_px'])
    print('rms', tables['before']['rms'], '->', tables['after']['rms'])
    print('measures', measure_rows)
    print('stereo chosen', [row for row in summary['stereo'] if row['num_disparities'] == 32 and row['block_size'] == 5])


if __name__ == '__main__':
    main()
