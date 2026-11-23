# -*- coding: utf-8 -*-
"""ДЗ: шаблон. Заполните TODO; команды и контракт — в README.md."""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def read_image(path):
    image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Не удалось прочитать изображение: ' + str(path))
    return image


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError('Ошибка кодирования: ' + str(path))
    encoded.tofile(path)


def normalize_points(points):
    """Нормализация Хартли: центроид 0, среднее расстояние sqrt(2)."""
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    center = pts.mean(axis=0)
    centered = pts - center
    mean_dist = np.linalg.norm(centered, axis=1).mean()
    if not np.isfinite(mean_dist) or mean_dist < 1e-12:
        raise ValueError('Точки вырождены: нулевой разброс')
    scale = np.sqrt(2.0) / mean_dist
    transform = np.array([
        [scale, 0.0, -scale * center[0]],
        [0.0, scale, -scale * center[1]],
        [0.0, 0.0, 1.0],
    ], np.float64)
    homogeneous = np.column_stack([pts, np.ones(len(pts))])
    normalized = (transform @ homogeneous.T).T
    return normalized[:, :2], transform


def dlt(src, dst):
    """Нормализованный DLT; H отображает src в dst, H[2,2]=1."""
    src = np.asarray(src, np.float64).reshape(-1, 2)
    dst = np.asarray(dst, np.float64).reshape(-1, 2)
    if len(src) < 4 or len(src) != len(dst):
        raise ValueError('Для DLT нужно не меньше четырёх пар')
    src_n, t_src = normalize_points(src)
    dst_n, t_dst = normalize_points(dst)
    rows = []
    for (x, y), (u, v) in zip(src_n, dst_n):
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, -u])
    design = np.asarray(rows, np.float64)
    _, singular, vt = np.linalg.svd(design)
    # У коллинеарных и прочих вырожденных наборов нуль-пространство толще одной оси.
    if singular[-2] <= 1e-8 * singular[0]:
        raise ValueError('Вырожденная конфигурация точек')
    homography_n = vt[-1].reshape(3, 3)
    homography = np.linalg.inv(t_dst) @ homography_n @ t_src
    if abs(homography[2, 2]) < 1e-12:
        raise ValueError('Вырожденная гомография')
    return homography / homography[2, 2]


def project(points, H):
    points = np.asarray(points, np.float64).reshape(-1, 2)
    q = np.column_stack([points, np.ones(len(points))]) @ np.asarray(H).T
    valid = np.abs(q[:, 2]) > 1e-12
    out = np.full((len(points), 2), np.inf)
    out[valid] = q[valid, :2] / q[valid, 2:3]
    return out


def ransac_homography(src, dst, threshold=3.0, iterations=500, seed=0):
    """Свой цикл RANSAC, свой DLT, маска инлаеров в исходном порядке."""
    src = np.asarray(src, np.float64).reshape(-1, 2)
    dst = np.asarray(dst, np.float64).reshape(-1, 2)
    if len(src) < 4 or len(src) != len(dst):
        raise ValueError('Для RANSAC нужно не меньше четырёх пар')
    rng = np.random.default_rng(seed)
    best_mask, best_count = None, 3
    for _ in range(iterations):
        sample = rng.choice(len(src), 4, replace=False)
        try:
            candidate = dlt(src[sample], dst[sample])
        except (ValueError, np.linalg.LinAlgError):
            continue
        error = np.linalg.norm(project(src, candidate) - dst, axis=1)
        mask = np.isfinite(error) & (error < threshold)
        count = int(mask.sum())
        if count > best_count:
            best_mask, best_count = mask, count
    if best_mask is None:
        raise ValueError('RANSAC не нашёл гомографию')
    homography = dlt(src[best_mask], dst[best_mask])
    error = np.linalg.norm(project(src, homography) - dst, axis=1)
    mask = np.isfinite(error) & (error < threshold)
    if int(mask.sum()) >= 4 and int(mask.sum()) != int(best_mask.sum()):
        homography = dlt(src[mask], dst[mask])
    return homography, mask


def match_points(image1, image2, detector='sift', ratio=0.75):
    """SIFT/ORB и ratio test; вернуть две согласованные Nx2 матрицы."""
    def gray(image):
        if image.ndim == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if detector == 'sift':
        extractor = cv2.SIFT_create()
        norm = cv2.NORM_L2
    elif detector == 'orb':
        extractor = cv2.ORB_create(nfeatures=4000)
        norm = cv2.NORM_HAMMING
    else:
        raise ValueError('Детектор должен быть sift или orb')
    keys1, desc1 = extractor.detectAndCompute(gray(image1), None)
    keys2, desc2 = extractor.detectAndCompute(gray(image2), None)
    if desc1 is None or desc2 is None or len(keys1) < 2 or len(keys2) < 2:
        raise ValueError('Недостаточно признаков')
    pairs = cv2.BFMatcher(norm).knnMatch(desc1, desc2, k=2)
    src, dst_pts = [], []
    for pair in pairs:
        if len(pair) != 2:
            continue
        best, second = pair
        if best.distance < ratio * second.distance:
            src.append(keys1[best.queryIdx].pt)
            dst_pts.append(keys2[best.trainIdx].pt)
    if len(src) < 4:
        raise ValueError('Недостаточно соответствий после ratio test')
    return np.float64(src), np.float64(dst_pts)


def image_corners(image):
    h, w = image.shape[:2]
    return np.float64([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])


def feather_blend(images, transforms):
    """H_i: кадр i → система первого кадра; чёрный цвет не означает пустоту."""
    transforms = [np.asarray(matrix, np.float64) for matrix in transforms]
    corners = np.vstack([project(image_corners(image), matrix) for image, matrix in zip(images, transforms)])
    if not np.isfinite(corners).all():
        raise ValueError('Преобразование уводит углы кадра в бесконечность')
    min_xy = np.floor(corners.min(axis=0) + 1e-9)
    max_xy = np.ceil(corners.max(axis=0) - 1e-9)
    width = int(max_xy[0] - min_xy[0] + 1)
    height = int(max_xy[1] - min_xy[1] + 1)
    shift = np.array([[1.0, 0.0, -min_xy[0]], [0.0, 1.0, -min_xy[1]], [0.0, 0.0, 1.0]], np.float64)
    accumulator = np.zeros((height, width, 3), np.float64)
    weight_sum = np.zeros((height, width), np.float64)
    for image, matrix in zip(images, transforms):
        mapping = shift @ matrix
        warped = cv2.warpPerspective(image, mapping, (width, height), flags=cv2.INTER_LINEAR)
        coverage = cv2.warpPerspective(np.full(image.shape[:2], 255, np.uint8), mapping, (width, height),
                                       flags=cv2.INTER_NEAREST)
        inside = coverage > 0
        if not inside.any():
            continue
        # Вес — расстояние до границы покрытия, а не яркость: чёрный кадр тоже занимает холст.
        weight = cv2.distanceTransform(inside.astype(np.uint8), cv2.DIST_L2, 3)
        weight = np.where(inside, np.maximum(weight, 1e-6), 0.0)
        accumulator += warped.astype(np.float64) * weight[..., None]
        weight_sum += weight
    panorama = np.zeros_like(accumulator)
    covered = weight_sum > 0
    panorama[covered] = accumulator[covered] / weight_sum[covered, None]
    return np.clip(np.rint(panorama), 0, 255).astype(np.uint8), shift


def smooth_homography(previous, current, corners, alpha=0.4):
    """EMA проекций углов: alpha=1 означает текущий кадр без сглаживания."""
    current = np.asarray(current, np.float64)
    corners = np.asarray(corners, np.float64).reshape(-1, 2)
    if previous is None or alpha >= 1.0:
        return current / current[2, 2]
    blended = alpha * project(corners, current) + (1.0 - alpha) * project(corners, previous)
    return dlt(corners, blended)


def overlay(frame, texture, H, opacity=0.8):
    """H переводит углы texture в координаты кадра."""
    height, width = frame.shape[:2]
    warped = cv2.warpPerspective(texture, H, (width, height), flags=cv2.INTER_NEAREST)
    coverage = cv2.warpPerspective(np.full(texture.shape[:2], 255, np.uint8), H, (width, height),
                                   flags=cv2.INTER_NEAREST)
    alpha = (coverage > 0).astype(np.float32) * np.float32(opacity)
    composed = frame.astype(np.float32) * (1.0 - alpha[..., None]) + warped.astype(np.float32) * alpha[..., None]
    return np.clip(np.rint(composed), 0, 255).astype(np.uint8)


def stitch(images, detector='sift', ratio=0.75, seed=0):
    transforms, reports = [np.eye(3)], []
    for i in range(1, len(images)):
        a, b = match_points(images[i], images[i-1], detector, ratio)
        H, mask = ransac_homography(a, b, seed=seed)
        transforms.append(transforms[-1] @ H)
        error = np.linalg.norm(project(a[mask], H) - b[mask], axis=1).mean()
        reports.append(dict(pair=[i, i-1], matches=len(a), inlier_ratio=float(mask.mean()),
                            reprojection_px=float(error)))
    panorama, _ = feather_blend(images, transforms)
    return panorama, reports


def ar_video(reference, texture, source, output=None, detector='sift', alpha=0.4, display=False):
    source = int(source) if str(source).isdigit() else str(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError('Не удалось открыть видео/камеру')
    writer, previous, count, tracked = None, None, 0, 0
    started = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            count += 1
            try:
                a, b = match_points(reference, frame, detector)
                H, _ = ransac_homography(a, b, iterations=150)
                previous = smooth_homography(previous, H, image_corners(reference), alpha)
                # Текстура масштабируется в размер опорного изображения.
                tex = cv2.resize(texture, (reference.shape[1], reference.shape[0]))
                frame = overlay(frame, tex, previous)
                tracked += 1
            except ValueError:
                previous = None  # При потере цели старую H не рисуем.
            if output:
                if writer is None:
                    Path(output).parent.mkdir(parents=True, exist_ok=True)
                    fps = cap.get(cv2.CAP_PROP_FPS) or 15
                    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'),
                                             fps, (frame.shape[1], frame.shape[0]))
                    if not writer.isOpened():
                        raise ValueError('Не удалось открыть выходное видео')
                writer.write(frame)
            if display:
                cv2.imshow('AR: Esc = выход', frame)
                if cv2.waitKey(1) == 27:
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()
    elapsed = time.perf_counter() - started
    if not count:
        raise ValueError('Видео не содержит кадров')
    return dict(frames=count, tracked=tracked, fps=count / elapsed)


def main():
    parser = argparse.ArgumentParser(description='HW3: панорама и AR')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('panorama')
    p.add_argument('images', nargs='+')
    p.add_argument('-o', '--output', required=True)
    p.add_argument('--detector', choices=['sift', 'orb'], default='sift')
    p.add_argument('--seed', type=int, default=0)
    p = sub.add_parser('dlt-check')
    p.add_argument('correspondences', help='JSON: src, dst (точные пары)')
    p = sub.add_parser('ar')
    p.add_argument('reference')
    p.add_argument('texture')
    p.add_argument('--source', required=True)
    p.add_argument('-o', '--output')
    p.add_argument('--alpha', type=float, default=0.4)
    p.add_argument('--display', action='store_true')
    args = parser.parse_args()
    if args.command == 'panorama':
        image, report = stitch([read_image(p) for p in args.images], args.detector, seed=args.seed)
        write_image(args.output, image)
        result = dict(shape=list(image.shape), pairs=report)
    elif args.command == 'dlt-check':
        data = json.loads(Path(args.correspondences).read_text(encoding='utf-8'))
        src, dst = np.float64(data['src']), np.float64(data['dst'])
        mine = dlt(src, dst)
        library, _ = cv2.findHomography(src, dst, method=0)
        delta = float(np.linalg.norm(mine / mine[2, 2] - library / library[2, 2]))
        result = dict(frobenius=delta, passed=delta < 1e-6)
    else:
        result = ar_video(read_image(args.reference), read_image(args.texture), args.source,
                          args.output, alpha=args.alpha, display=args.display)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
