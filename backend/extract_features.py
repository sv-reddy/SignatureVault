"""
extract_features.py
===================
High-performance 4-channel signature feature extractor for SignVault.

HOW IT WORKS
------------
Reads unified images from DATA/unification_Data/ and writes (4, 384, 384)
uint8 .npy tensors to DATA/process_data/.

Pipeline per image:

  Phase 1 – Illumination Normalization
    • BGR → LAB, extract L channel (luminance only)
    • Shadow removal: L / GaussBlur(L, σ=max_dim/4) – cancels lighting gradients
    • Adaptive Gaussian Threshold (blockSize=41, C=10) → binary ink mask
    • ROI crop (15 px padding) + letterbox resize to 384×384

  Phase 2 – Multimodal 4-Channel Extraction
    Ch-0  Shape          : inverted binary ink mask  (ink=255, bg=0)
    Ch-1  Pseudo-Pressure: Gaussian blur σ=1.0 of Ch-0 → stroke density
    Ch-2  Stroke Angle   : arctan2(Sobel-Y, Sobel-X), mapped to [0, 255]
    Ch-3  Skeleton       : Zhang-Suen / morphological thinning → 1-px paths

  Phase 3 – Storage
    Stack → (4, 384, 384) uint8 .npy  (~4× smaller than float32)
    Naming: {DATASET}_{USER_ID}_{G|F}_{COUNT}.npy

GPU priority: cv2.cuda → CuPy → PyTorch-CUDA → SciPy (CPU fallback)
Parallelism : ProcessPoolExecutor; capped to 3 workers when GPU is active
              (each spawned process creates its own CUDA context ~200 MB VRAM)

HOW TO RUN
----------
    python extract_features.py

No command-line arguments. Skips images that already have a .npy output.
DATA paths resolved relative to this script's parent directory.
"""

import logging
import math
import os
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter as _scipy_gauss
from scipy.ndimage import sobel as _scipy_sobel


from tqdm import tqdm as _tqdm_cls

def _tqdm(it, **kw):
    return _tqdm_cls(it, **kw)


TARGET          = 384          
ROI_PAD         = 15           
ADAPT_BLOCK     = 41           
ADAPT_C         = 10           
PRESSURE_SIGMA  = 1.0          
IMAGE_EXTS      = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

_SCRIPT_DIR = Path(__file__).resolve().parent
_DATA_ROOT  = _SCRIPT_DIR.parent / "DATA"
INPUT_DIR   = _DATA_ROOT / "unification_Data"
OUTPUT_DIR  = _DATA_ROOT / "process_data"
LOG_FILE    = OUTPUT_DIR / "feature_extraction.log"

_BACKEND: str | None = None


def _backend() -> str:
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND

    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            cv2.cuda.setDevice(0)
            _probe = cv2.cuda_GpuMat()
            _probe.upload(np.zeros((4, 4), dtype=np.float32))
            del _probe
            _BACKEND = "cv2cuda"
            return _BACKEND
    except (AttributeError, cv2.error, Exception):
        pass

    try:
        import cupy as cp
        cp.zeros(1)
        _BACKEND = "cupy"
        return _BACKEND
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            torch.zeros(1, device="cuda")
            _BACKEND = "torch"
            return _BACKEND
    except Exception:
        pass

    _BACKEND = "cpu"
    return _BACKEND



def gpu_gaussian(arr: np.ndarray, sigma: float) -> np.ndarray:
    b     = _backend()
    arr_f = arr.astype(np.float32)

    if b == "cv2cuda":
        try:
            ks = int(6 * sigma + 1) | 1
            ks = min(ks, 1025)
            gpu_mat = cv2.cuda_GpuMat()
            gpu_mat.upload(arr_f)
            gauss = cv2.cuda.createGaussianFilter(
                cv2.CV_32F, cv2.CV_32F, (ks, ks), sigma
            )
            return gauss.apply(gpu_mat).download()
        except Exception:
            pass

    if b in ("cv2cuda", "cupy"):
        try:
            import cupy as cp
            import cupyx.scipy.ndimage as cpnd
            return cp.asnumpy(
                cpnd.gaussian_filter(cp.asarray(arr_f), sigma=sigma)
            )
        except Exception:
            pass

    if b in ("cv2cuda", "cupy", "torch"):
        try:
            return _torch_gaussian(arr_f, sigma)
        except Exception:
            pass

    return _scipy_gauss(arr_f, sigma=sigma)


def _torch_gaussian(arr_f: np.ndarray, sigma: float) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    ks = int(6 * sigma + 1) | 1
    ks = min(ks, 511)
    x  = torch.arange(ks, dtype=torch.float32) - ks // 2
    g  = torch.exp(-x ** 2 / (2.0 * sigma ** 2))
    g  /= g.sum()
    k  = (g.unsqueeze(1) @ g.unsqueeze(0)).view(1, 1, ks, ks).cuda()

    t  = torch.from_numpy(arr_f).cuda().unsqueeze(0).unsqueeze(0)
    p  = ks // 2
    out = torch.nn.functional.conv2d(
        torch.nn.functional.pad(t, [p, p, p, p], mode="reflect"), k
    )
    return out.squeeze().cpu().numpy()


def gpu_sobel_angle(arr: np.ndarray) -> np.ndarray:
    b     = _backend()
    arr_f = arr.astype(np.float32)


    if b == "cv2cuda":
        try:
            gpu_mat = cv2.cuda_GpuMat()
            gpu_mat.upload(arr_f)
            sx_filt = cv2.cuda.createSobelFilter(cv2.CV_32F, cv2.CV_32F, 1, 0, ksize=3)
            sy_filt = cv2.cuda.createSobelFilter(cv2.CV_32F, cv2.CV_32F, 0, 1, ksize=3)
            gx = sx_filt.apply(gpu_mat).download()
            gy = sy_filt.apply(gpu_mat).download()
            angle = np.arctan2(gy, gx)
            return ((angle + np.pi) / (2.0 * np.pi) * 255.0).astype(np.float32)
        except Exception:
            pass

    if b in ("cv2cuda", "cupy"):
        try:
            import cupy as cp
            import cupyx.scipy.ndimage as cpnd
            g  = cp.asarray(arr_f)
            gx = cpnd.sobel(g, axis=1).astype(cp.float32)
            gy = cpnd.sobel(g, axis=0).astype(cp.float32)
            angle = cp.arctan2(gy, gx)
            return cp.asnumpy(
                (angle + cp.pi) / (2.0 * cp.pi) * 255.0
            ).astype(np.float32)
        except Exception:
            pass

    if b in ("cv2cuda", "cupy", "torch"):
        try:
            import torch
            import torch.nn.functional as F
            kx = torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
            ).cuda().view(1, 1, 3, 3)
            ky = torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
            ).cuda().view(1, 1, 3, 3)
            t  = torch.from_numpy(arr_f).cuda().unsqueeze(0).unsqueeze(0)
            tp = F.pad(t, [1, 1, 1, 1], mode="reflect")
            gx = F.conv2d(tp, kx).squeeze().cpu().numpy()
            gy = F.conv2d(tp, ky).squeeze().cpu().numpy()
            angle = np.arctan2(gy, gx)
            return ((angle + np.pi) / (2.0 * np.pi) * 255.0).astype(np.float32)
        except Exception:
            pass

    gx    = _scipy_sobel(arr_f, axis=1)
    gy    = _scipy_sobel(arr_f, axis=0)
    angle = np.arctan2(gy.astype(np.float32), gx.astype(np.float32))
    return ((angle + np.pi) / (2.0 * np.pi) * 255.0).astype(np.float32)



def skeletonize(ink: np.ndarray) -> np.ndarray:
    try:
        from skimage.morphology import skeletonize as _sk_thin
        skel_bool = _sk_thin(ink > 127)
        return (skel_bool.astype(np.uint8) * 255)
    except ImportError:
        pass


    try:
        thinned = cv2.ximgproc.thinning(
            ink, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
        )
        return thinned
    except AttributeError:
        pass


    return _morph_skeleton(ink)


def _morph_skeleton(ink: np.ndarray) -> np.ndarray:
    skel   = np.zeros_like(ink)
    temp   = ink.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    prev_nonzero = -1

    while True:
        eroded       = cv2.erode(temp, kernel)
        dilated_back = cv2.dilate(eroded, kernel)
        diff         = cv2.subtract(temp, dilated_back)
        cv2.bitwise_or(skel, diff, skel)
        temp         = eroded
        nz           = cv2.countNonZero(temp)
        if nz == 0 or nz == prev_nonzero:
            break
        prev_nonzero = nz

    return skel



def _read_image(path: Path) -> np.ndarray:
    try:
        from PIL import Image
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with Image.open(path) as pil_img:
                img_rgb = pil_img.convert("RGB")
                return cv2.cvtColor(np.array(img_rgb, dtype=np.uint8),
                                    cv2.COLOR_RGB2BGR)
    except Exception as e:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            return img
        raise IOError(f"Cannot read image via PIL or OpenCV: {path} ({e})")



def _letterbox(mask: np.ndarray, size: int = TARGET) -> np.ndarray:
    h, w = mask.shape[:2]
    if h == 0 or w == 0:
        return np.full((size, size), 255, dtype=np.uint8)

    scale   = size / max(h, w)
    new_h   = max(1, int(round(h * scale)))
    new_w   = max(1, int(round(w * scale)))
    interp  = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(mask, (new_w, new_h), interpolation=interp)
    _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

    canvas  = np.full((size, size), 255, dtype=np.uint8)
    y0      = (size - new_h) // 2
    x0      = (size - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _remove_form_lines(ink_mask: np.ndarray) -> np.ndarray:
    h, w = ink_mask.shape[:2]

    horiz_kernel_len = max(31, w // 3)
    vert_kernel_len = max(31, h // 3)

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_kernel_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_kernel_len))

    horiz_lines = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, h_kernel)
    vert_lines = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, v_kernel)
    lines = cv2.bitwise_or(horiz_lines, vert_lines)

    edges = cv2.Canny(ink_mask, 50, 150)
    hough = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=max(40, min(h, w) // 2),
        maxLineGap=8,
    )
    if hough is not None:
        min_h_len = max(40, int(w * 0.55))
        min_v_len = max(40, int(h * 0.55))
        horiz_tol = max(2, h // 80)
        vert_tol = max(2, w // 80)
        for seg in hough[:, 0, :]:
            x1, y1, x2, y2 = [int(v) for v in seg]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dx >= min_h_len and dy <= horiz_tol:
                cv2.line(lines, (x1, y1), (x2, y2), 255, 2)
            elif dy >= min_v_len and dx <= vert_tol:
                cv2.line(lines, (x1, y1), (x2, y2), 255, 2)

    lines = cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    return cv2.bitwise_and(ink_mask, cv2.bitwise_not(lines))


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]


    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L   = lab[:, :, 0].astype(np.float32)

    sigma_div = max(h, w) / 4.0
    L_blur    = gpu_gaussian(L, sigma_div)
    L_norm    = np.clip(L / (L_blur + 1e-6) * 128.0, 0.0, 255.0).astype(np.uint8)

    binary = cv2.adaptiveThreshold(
        L_norm, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        ADAPT_BLOCK, ADAPT_C,
    )

    ink_px = _remove_form_lines(255 - binary)
    binary_clean = 255 - ink_px

    ys, xs = np.where(ink_px > 0)
    if len(ys) == 0:
        cropped = binary_clean
    else:
        y1 = max(0, int(ys.min()) - ROI_PAD)
        y2 = min(h, int(ys.max()) + ROI_PAD + 1)
        x1 = max(0, int(xs.min()) - ROI_PAD)
        x2 = min(w, int(xs.max()) + ROI_PAD + 1)
        cropped = binary_clean[y1:y2, x1:x2]


    return _letterbox(cropped, TARGET)



def extract_channels(binary_mask: np.ndarray) -> np.ndarray:
    ch0 = (255 - binary_mask)

    ch1 = np.clip(
        gpu_gaussian(ch0.astype(np.float32), PRESSURE_SIGMA), 0.0, 255.0
    ).astype(np.uint8)

    ch2 = np.clip(
        gpu_sobel_angle(ch0.astype(np.float32)), 0.0, 255.0
    ).astype(np.uint8)

    ch3 = skeletonize(ch0)

    return np.stack([ch0, ch1, ch2, ch3], axis=0).astype(np.uint8)



def _worker(args: tuple) -> tuple:
    src_str, dst_str = args
    src, dst = Path(src_str), Path(dst_str)
    try:
        img    = _read_image(src)
        mask   = preprocess(img)
        tensor = extract_channels(mask)
        np.save(str(dst), tensor)
        return (src.name, True, None)
    except Exception:
        return (src.name, False, traceback.format_exc())



def _detect_gpu() -> str:
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            return "cv2cuda"
    except (AttributeError, Exception):
        pass
    try:
        import cupy  # noqa: F401
        return "cupy"
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            return "torch"
    except Exception:
        pass
    return "cpu"



def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    log = logging.getLogger(__name__)

    if not INPUT_DIR.exists():
        log.error("Input directory not found: %s", INPUT_DIR)
        raise SystemExit(1)

    src_files = sorted(
        p for p in INPUT_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    if not src_files:
        log.error("No images found in %s", INPUT_DIR)
        raise SystemExit(1)

    tasks: list[tuple[str, str]] = []
    skipped = 0
    for src in src_files:
        dst = OUTPUT_DIR / (src.stem + ".npy")
        if dst.exists():
            skipped += 1
        else:
            tasks.append((str(src), str(dst)))

    gpu_tag = _detect_gpu()
    cpu_cnt = os.cpu_count() or 4

    if gpu_tag != "cpu":
        n_workers   = min(cpu_cnt, 3)
        worker_note = "(capped to 3 – preserves VRAM on RTX 3050 6 GB)"
    else:
        n_workers   = cpu_cnt
        worker_note = "(all available CPU cores)"

    log.info("=" * 66)
    log.info("  SignVault  –  Feature Extraction  (4-channel, uint8)")
    log.info("=" * 66)
    log.info("Input dir    : %s", INPUT_DIR)
    log.info("Output dir   : %s", OUTPUT_DIR)
    log.info("Channel map  : Ch0=Shape | Ch1=Pressure | Ch2=Angle | Ch3=Skeleton")
    log.info("Output dtype : uint8  (4, 384, 384)  ")
    log.info("Total images : %d  |  already done (skipped): %d",
             len(src_files), skipped)
    log.info("To process   : %d", len(tasks))
    log.info("GPU backend  : %s", gpu_tag.upper())
    log.info("Workers      : %d  %s", n_workers, worker_note)
    log.info("-" * 66)

    if not tasks:
        log.info("Nothing to do – all outputs already exist.")
        return

    ok_count = err_count = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_worker, t): t[0] for t in tasks}
        for fut in _tqdm(
            as_completed(futures),
            total=len(tasks),
            desc="Extracting features",
            unit="img",
        ):
            fname, ok, err = fut.result()
            if ok:
                ok_count += 1
            else:
                err_count += 1
                log.warning("FAILED  %s\n%s", fname, err.strip())

    log.info("=" * 66)
    log.info("Complete   ✓ %d extracted    ✗ %d failed", ok_count, err_count)
    log.info("Output dir : %s", OUTPUT_DIR)
    log.info("Log file   : %s", LOG_FILE)


if __name__ == "__main__":
    main()
