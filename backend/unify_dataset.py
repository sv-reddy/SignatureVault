"""
unify_dataset.py
================
Unifies CEDAR, BHSig260 (Hindi & Bengali), GPDS, ICDAR2011, and Independent
signature datasets into a single flat directory: DATA/unification_Data/

HOW IT WORKS
------------
Scans each dataset's source directories, renames files to a collision-free
global UID convention, and copies them under DATA/unification_Data/:

    Source                          UID range     Writers
    CEDAR/full_org|forg             101 – 155      55      writers 1-55
    BHSig260/Hindi                  201 – 360     160      folders 001-160
    BHSig260/Bengali                401 – 500     100      folders 001-100
    GPDS/genuine|forge              601 – 750     150      writers 001-150
    ICDAR2011/{id,id_forg}          801 – 899      64      folders 001-069 (flat)
    Independent/{id,id_forg}       1001 – 1223    223      folders 001-223

Output filename convention:
    {DATASET}_{USER_ID:03d}_{G|F}_{COUNT:03d}.{ext}
        e.g.  CEDAR_101_G_001.png   HS_201_F_030.tif   GPDS_601_G_016.jpg   ICDAR_849_G_011.png   IND_1001_F_004.jpg

Outputs written to DATA/unification_Data/:
    manifest.json      – per-filename metadata (dataset, script, status, UID)
    unification.log    – copy log + warnings for missing/short-count writers

HOW TO RUN
----------
    python unify_dataset.py

"""

import json
import logging
import re
import shutil
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT  = SCRIPT_DIR.parent / "DATA"

OUT_DIR   = DATA_ROOT / "unification_Data"
LOG_FILE  = OUT_DIR / "unification.log"
MANIFEST  = OUT_DIR / "manifest.json"

EXPECTED_GENUINE: dict[str, int] = {
    "CEDAR": 24,
    "HS":    24,
    "BS":    24,
    "GPDS":  16,
    "ICDAR": 12,
    "IND":   10,
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}



def setup_logging() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(levelname)-8s %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def _copy(src: Path, dst: Path) -> bool:
    if not src.exists():
        logging.warning("MISSING FILE  : %s", src)
        return False
    shutil.copy2(src, dst)
    return True


def _rel(path: Path) -> str:
    try:
        return path.relative_to(DATA_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _iter_image_files(folder: Path) -> list[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: p.name.lower(),
    )



def process_cedar(manifest: dict) -> None:
    org_dir  = DATA_ROOT / "CEDAR" / "full_org"
    forg_dir = DATA_ROOT / "CEDAR" / "full_forg"

    _GEN_RE  = re.compile(r"^original_(\d+)_(\d+)\.png$",   re.IGNORECASE)
    _FORG_RE = re.compile(r"^forgeries_(\d+)_(\d+)\.png$",  re.IGNORECASE)


    gen_by_writer: dict[int, list[tuple[int, Path]]] = defaultdict(list)
    for f in org_dir.iterdir():
        m = _GEN_RE.match(f.name)
        if m:
            gen_by_writer[int(m.group(1))].append((int(m.group(2)), f))


    forg_by_writer: dict[int, list[tuple[int, Path]]] = defaultdict(list)
    for f in forg_dir.iterdir():
        m = _FORG_RE.match(f.name)
        if m:
            forg_by_writer[int(m.group(1))].append((int(m.group(2)), f))

    if not gen_by_writer:
        logging.error("CEDAR: no genuine images found in %s", org_dir)
        return

    total = 0
    for writer_num in sorted(gen_by_writer):
        uid     = 100 + writer_num
        uid_str = str(uid).zfill(3)

        gen_files = sorted(gen_by_writer[writer_num], key=lambda t: t[0])
        if len(gen_files) < EXPECTED_GENUINE["CEDAR"]:
            logging.warning(
                "CEDAR writer %03d (UID %s): expected %d genuine, found %d",
                writer_num, uid_str, EXPECTED_GENUINE["CEDAR"], len(gen_files),
            )
        for cnt, (_, src) in enumerate(gen_files, start=1):
            dst_name = f"CEDAR_{uid_str}_G_{cnt:03d}.png"
            if _copy(src, OUT_DIR / dst_name):
                manifest[dst_name] = {
                    "original_path": _rel(src),
                    "dataset":       "CEDAR",
                    "script":        "Latin",
                    "status":        "genuine",
                    "user_id":       uid_str,
                }
                total += 1


        forg_files = sorted(forg_by_writer.get(writer_num, []), key=lambda t: t[0])
        if not forg_files:
            logging.warning("CEDAR writer %03d: no forgery files found", writer_num)
        for cnt, (_, src) in enumerate(forg_files, start=1):
            dst_name = f"CEDAR_{uid_str}_F_{cnt:03d}.png"
            if _copy(src, OUT_DIR / dst_name):
                manifest[dst_name] = {
                    "original_path": _rel(src),
                    "dataset":       "CEDAR",
                    "script":        "Latin",
                    "status":        "forgery",
                    "user_id":       uid_str,
                }
                total += 1

    writer_count = len(gen_by_writer)
    logging.info(
        "CEDAR : %3d writers processed → UIDs 101-%d  |  %d files copied",
        writer_count, 100 + max(gen_by_writer), total,
    )


def process_bhsig_hindi(manifest: dict) -> None:
    hindi_root = DATA_ROOT / "BHSig260" / "Hindi"
    folders = sorted(
        [d for d in hindi_root.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    if not folders:
        logging.error("HS: no writer folders found in %s", hindi_root)
        return

    total = 0
    for folder in folders:
        folder_idx = int(folder.name)       
        uid        = 200 + folder_idx       
        uid_str    = str(uid).zfill(3)

        _GEN_RE  = re.compile(
            rf"^H-S-0*{folder_idx}-G-(\d+)\.(tif|png|jpg|jpeg)$", re.IGNORECASE
        )
        _FORG_RE = re.compile(
            rf"^H-S-0*{folder_idx}-F-(\d+)\.(tif|png|jpg|jpeg)$", re.IGNORECASE
        )

        gen_files: list[tuple[int, Path]] = []
        forg_files: list[tuple[int, Path]] = []
        for f in folder.iterdir():
            mg = _GEN_RE.match(f.name)
            mf = _FORG_RE.match(f.name)
            if mg:
                gen_files.append((int(mg.group(1)), f))
            elif mf:
                forg_files.append((int(mf.group(1)), f))

        gen_files.sort(key=lambda t: t[0])
        forg_files.sort(key=lambda t: t[0])

        if len(gen_files) < EXPECTED_GENUINE["HS"]:
            logging.warning(
                "HS   folder %s (UID %s): expected %d genuine, found %d",
                folder.name, uid_str, EXPECTED_GENUINE["HS"], len(gen_files),
            )
        if not forg_files:
            logging.warning("HS   folder %s: no forgery files found", folder.name)

        for cnt, (_, src) in enumerate(gen_files, start=1):
            dst_name = f"HS_{uid_str}_G_{cnt:03d}.tif"
            if _copy(src, OUT_DIR / dst_name):
                manifest[dst_name] = {
                    "original_path": _rel(src),
                    "dataset":       "BHSig260_Hindi",
                    "script":        "Hindi",
                    "status":        "genuine",
                    "user_id":       uid_str,
                }
                total += 1

        for cnt, (_, src) in enumerate(forg_files, start=1):
            dst_name = f"HS_{uid_str}_F_{cnt:03d}.tif"
            if _copy(src, OUT_DIR / dst_name):
                manifest[dst_name] = {
                    "original_path": _rel(src),
                    "dataset":       "BHSig260_Hindi",
                    "script":        "Hindi",
                    "status":        "forgery",
                    "user_id":       uid_str,
                }
                total += 1

    logging.info(
        "HS    : %3d writers processed → UIDs 201-360  |  %d files copied",
        len(folders), total,
    )

def process_bhsig_bengali(manifest: dict) -> None:
    bengali_root = DATA_ROOT / "BHSig260" / "Bengali"
    folders = sorted(
        [d for d in bengali_root.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    if not folders:
        logging.error("BS: no writer folders found in %s", bengali_root)
        return

    total = 0
    for folder in folders:
        folder_idx = int(folder.name)
        uid        = 400 + folder_idx
        uid_str    = str(uid).zfill(3)

        _GEN_RE  = re.compile(
            rf"^B-S-0*{folder_idx}-G-(\d+)\.(tif|png|jpg|jpeg)$", re.IGNORECASE
        )
        _FORG_RE = re.compile(
            rf"^B-S-0*{folder_idx}-F-(\d+)\.(tif|png|jpg|jpeg)$", re.IGNORECASE
        )

        gen_files: list[tuple[int, Path]] = []
        forg_files: list[tuple[int, Path]] = []
        for f in folder.iterdir():
            mg = _GEN_RE.match(f.name)
            mf = _FORG_RE.match(f.name)
            if mg:
                gen_files.append((int(mg.group(1)), f))
            elif mf:
                forg_files.append((int(mf.group(1)), f))

        gen_files.sort(key=lambda t: t[0])
        forg_files.sort(key=lambda t: t[0])

        if len(gen_files) < EXPECTED_GENUINE["BS"]:
            logging.warning(
                "BS   folder %s (UID %s): expected %d genuine, found %d",
                folder.name, uid_str, EXPECTED_GENUINE["BS"], len(gen_files),
            )
        if not forg_files:
            logging.warning("BS   folder %s: no forgery files found", folder.name)

        for cnt, (_, src) in enumerate(gen_files, start=1):
            dst_name = f"BS_{uid_str}_G_{cnt:03d}.tif"
            if _copy(src, OUT_DIR / dst_name):
                manifest[dst_name] = {
                    "original_path": _rel(src),
                    "dataset":       "BHSig260_Bengali",
                    "script":        "Bengali",
                    "status":        "genuine",
                    "user_id":       uid_str,
                }
                total += 1

        for cnt, (_, src) in enumerate(forg_files, start=1):
            dst_name = f"BS_{uid_str}_F_{cnt:03d}.tif"
            if _copy(src, OUT_DIR / dst_name):
                manifest[dst_name] = {
                    "original_path": _rel(src),
                    "dataset":       "BHSig260_Bengali",
                    "script":        "Bengali",
                    "status":        "forgery",
                    "user_id":       uid_str,
                }
                total += 1

    logging.info(
        "BS    : %3d writers processed → UIDs 401-500  |  %d files copied",
        len(folders), total,
    )


def process_gpds(manifest: dict) -> None:
    gen_dir  = DATA_ROOT / "GPDS" / "genuine"
    forg_dir = DATA_ROOT / "GPDS" / "forge"

    _GEN_RE  = re.compile(r"^c-(\d{3})-(\d{2}) \(Copy\)\.jpg$",  re.IGNORECASE)
    _FORG_RE = re.compile(r"^cf-(\d{3})-(\d{2}) \(Copy\)\.jpg$", re.IGNORECASE)


    gen_by_writer: dict[int, list[tuple[int, Path]]] = defaultdict(list)
    for f in gen_dir.iterdir():
        m = _GEN_RE.match(f.name)
        if m:
            gen_by_writer[int(m.group(1))].append((int(m.group(2)), f))


    forg_by_writer: dict[int, list[tuple[int, Path]]] = defaultdict(list)
    for f in forg_dir.iterdir():
        m = _FORG_RE.match(f.name)
        if m:
            forg_by_writer[int(m.group(1))].append((int(m.group(2)), f))

    if not gen_by_writer:
        logging.error("GPDS: no genuine images found in %s", gen_dir)
        return

    total = 0
    for writer_num in sorted(gen_by_writer):
        uid     = 600 + writer_num
        uid_str = str(uid).zfill(3)
        w_str   = str(writer_num).zfill(3)

        gen_files = sorted(gen_by_writer[writer_num], key=lambda t: t[0])
        if len(gen_files) < EXPECTED_GENUINE["GPDS"]:
            logging.warning(
                "GPDS writer %s (UID %s): expected %d genuine, found %d",
                w_str, uid_str, EXPECTED_GENUINE["GPDS"], len(gen_files),
            )
        for cnt, (_, src) in enumerate(gen_files, start=1):
            dst_name = f"GPDS_{uid_str}_G_{cnt:03d}.jpg"
            if _copy(src, OUT_DIR / dst_name):
                manifest[dst_name] = {
                    "original_path": _rel(src),
                    "dataset":       "GPDS",
                    "script":        "Latin",
                    "status":        "genuine",
                    "user_id":       uid_str,
                }
                total += 1


        forg_files = sorted(forg_by_writer.get(writer_num, []), key=lambda t: t[0])
        if not forg_files:
            logging.warning("GPDS writer %s: no forgery files found", w_str)
        for cnt, (_, src) in enumerate(forg_files, start=1):
            dst_name = f"GPDS_{uid_str}_F_{cnt:03d}.jpg"
            if _copy(src, OUT_DIR / dst_name):
                manifest[dst_name] = {
                    "original_path": _rel(src),
                    "dataset":       "GPDS",
                    "script":        "Latin",
                    "status":        "forgery",
                    "user_id":       uid_str,
                }
                total += 1

    writer_count = len(gen_by_writer)
    logging.info(
        "GPDS  : %3d writers processed → UIDs 601-750  |  %d files copied",
        writer_count, total,
    )


def process_icdar2011(manifest: dict) -> None:
    icdar_root = DATA_ROOT / "ICDAR2011"

    if not icdar_root.exists():
        logging.error("ICDAR: root not found at %s", icdar_root)
        return

    gen_by_writer: dict[int, list[Path]] = defaultdict(list)
    forg_by_writer: dict[int, list[Path]] = defaultdict(list)

    # Flat layout: DATA/ICDAR2011/{001,002,...}/ and DATA/ICDAR2011/{001_forg,...}/
    for folder in sorted([d for d in icdar_root.iterdir() if d.is_dir()], key=lambda d: d.name.lower()):
        if folder.name.isdigit():
            gen_by_writer[int(folder.name)].extend(_iter_image_files(folder))
        elif folder.name.endswith("_forg") and folder.name[:-5].isdigit():
            forg_by_writer[int(folder.name[:-5])].extend(_iter_image_files(folder))

    if not gen_by_writer:
        logging.error("ICDAR: no genuine writer folders found in %s", icdar_root)
        return

    total = 0
    for writer_num in sorted(gen_by_writer):
        uid = 800 + writer_num
        uid_str = str(uid).zfill(3)

        gen_files = sorted(gen_by_writer[writer_num], key=lambda p: p.name.lower())
        if len(gen_files) < EXPECTED_GENUINE["ICDAR"]:
            logging.warning(
                "ICDAR writer %03d (UID %s): expected %d genuine, found %d",
                writer_num, uid_str, EXPECTED_GENUINE["ICDAR"], len(gen_files),
            )
        for cnt, src in enumerate(gen_files, start=1):
            ext = src.suffix.lower() or ".png"
            dst_name = f"ICDAR_{uid_str}_G_{cnt:03d}{ext}"
            if _copy(src, OUT_DIR / dst_name):
                manifest[dst_name] = {
                    "original_path": _rel(src),
                    "dataset":       "ICDAR2011",
                    "script":        "Latin",
                    "status":        "genuine",
                    "user_id":       uid_str,
                }
                total += 1

        forg_files = sorted(forg_by_writer.get(writer_num, []), key=lambda p: p.name.lower())
        if not forg_files:
            logging.warning("ICDAR writer %03d: no forgery files found", writer_num)
        for cnt, src in enumerate(forg_files, start=1):
            ext = src.suffix.lower() or ".png"
            dst_name = f"ICDAR_{uid_str}_F_{cnt:03d}{ext}"
            if _copy(src, OUT_DIR / dst_name):
                manifest[dst_name] = {
                    "original_path": _rel(src),
                    "dataset":       "ICDAR2011",
                    "script":        "Latin",
                    "status":        "forgery",
                    "user_id":       uid_str,
                }
                total += 1

    writer_count = len(gen_by_writer)
    logging.info(
        "ICDAR : %3d writers processed → UIDs %d-%d  |  %d files copied",
        writer_count, 800 + min(gen_by_writer), 800 + max(gen_by_writer), total,
    )


def process_independent(manifest: dict) -> None:
    ind_root = DATA_ROOT / "Independent"
    if not ind_root.exists():
        logging.error("IND: root not found at %s", ind_root)
        return

    folders = sorted(
        [d for d in ind_root.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    if not folders:
        logging.error("IND: no genuine writer folders found in %s", ind_root)
        return

    total = 0
    for folder in folders:
        writer_num = int(folder.name)
        uid = 1000 + writer_num
        uid_str = str(uid).zfill(4)

        gen_files = _iter_image_files(folder)
        forg_folder = ind_root / f"{folder.name}_forg"
        forg_files = _iter_image_files(forg_folder) if forg_folder.exists() else []

        if len(gen_files) < EXPECTED_GENUINE["IND"]:
            logging.warning(
                "IND writer %03d (UID %s): expected %d genuine, found %d",
                writer_num, uid_str, EXPECTED_GENUINE["IND"], len(gen_files),
            )
        if not forg_files:
            logging.warning("IND writer %03d: no forgery files found", writer_num)

        for cnt, src in enumerate(gen_files, start=1):
            ext = src.suffix.lower() or ".jpg"
            dst_name = f"IND_{uid_str}_G_{cnt:03d}{ext}"
            if _copy(src, OUT_DIR / dst_name):
                manifest[dst_name] = {
                    "original_path": _rel(src),
                    "dataset":       "Independent",
                    "script":        "Latin",
                    "status":        "genuine",
                    "user_id":       uid_str,
                }
                total += 1

        for cnt, src in enumerate(forg_files, start=1):
            ext = src.suffix.lower() or ".jpg"
            dst_name = f"IND_{uid_str}_F_{cnt:03d}{ext}"
            if _copy(src, OUT_DIR / dst_name):
                manifest[dst_name] = {
                    "original_path": _rel(src),
                    "dataset":       "Independent",
                    "script":        "Latin",
                    "status":        "forgery",
                    "user_id":       uid_str,
                }
                total += 1

    logging.info(
        "IND   : %3d writers processed → UIDs 1001-%d  |  %d files copied",
        len(folders), 1000 + max(int(f.name) for f in folders), total,
    )

def main() -> None:
    setup_logging()
    logging.info("=" * 60)
    logging.info("  Dataset Unification  –  SignVault")
    logging.info("=" * 60)
    logging.info("DATA root   : %s", DATA_ROOT)
    logging.info("Output dir  : %s", OUT_DIR)

    required_dirs = {
        "CEDAR/full_org":         DATA_ROOT / "CEDAR"    / "full_org",
        "CEDAR/full_forg":        DATA_ROOT / "CEDAR"    / "full_forg",
        "BHSig260/Hindi":         DATA_ROOT / "BHSig260" / "Hindi",
        "BHSig260/Bengali":       DATA_ROOT / "BHSig260" / "Bengali",
        "GPDS/genuine":           DATA_ROOT / "GPDS"     / "genuine",
        "GPDS/forge":             DATA_ROOT / "GPDS"     / "forge",
        "ICDAR2011":              DATA_ROOT / "ICDAR2011",
        "Independent":            DATA_ROOT / "Independent",
    }
    missing = [label for label, path in required_dirs.items() if not path.exists()]
    if missing:
        for label in missing:
            logging.error("Required directory missing: %s", label)
        raise SystemExit(1)

    manifest: dict = {}

    process_cedar(manifest)
    process_bhsig_hindi(manifest)
    process_bhsig_bengali(manifest)
    process_gpds(manifest)
    process_icdar2011(manifest)
    process_independent(manifest)

    if not manifest:
        logging.error(
            "No images were copied. Verify source dataset directories contain images."
        )
        raise SystemExit(1)


    MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logging.info("-" * 60)
    logging.info("Manifest    : %s  (%d entries)", MANIFEST.name, len(manifest))
    logging.info("Log         : %s", LOG_FILE.name)
    logging.info("=" * 60)
    logging.info("Unification complete.")


if __name__ == "__main__":
    main()
