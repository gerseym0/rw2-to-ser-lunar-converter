#!/usr/bin/env python3
"""
rw2_to_ser.py  —  Panasonic S5 RW2 → 16-bit SER (raw Bayer, без дебайеризации)
================================================================================
Конвертирует папку с RW2-кадрами в единый SER-файл.
Опционально: детектирует диск Луны и делает кроп с отступом.

Зависимости:
    pip install rawpy numpy scipy

Использование:
    python rw2_to_ser.py                        # настройки из секции ниже
    python rw2_to_ser.py  <папка>  <выход.ser>  # через аргументы

В AutoStakkert3:
    File → Open → .ser
    Color → Force Bayer RGGB
    Advanced Settings → Drizzle 1.5x  (= автоматически Bayer Drizzle)
"""

import sys
import os
import glob
import struct
import numpy as np
import rawpy
from datetime import datetime, timezone

# ╔══════════════════════════════════════════════════════════════╗
# ║                      НАСТРОЙКИ                              ║
# ╚══════════════════════════════════════════════════════════════╝

INPUT_FOLDER  = r"C:\Users\ibatu\OneDrive\Desktop\ser"
OUTPUT_FILE   = r"C:\Users\ibatu\OneDrive\Desktop\ser\moon_raw_bayer.ser"

# ── Кроп ──────────────────────────────────────────────────────
# True  = авто-детекция Луны + кроп
# False = без кропа (полный кадр)
ENABLE_CROP         = True

# Отступ от края диска до края кадра (в пикселях).
# Рекомендуется 200-400 px — достаточно для AS3 MAP alignment.
CROP_PADDING_PX     = 50

# Сколько кадров анализировать для нахождения позиции Луны.
# Больше = надёжнее, но медленнее. 3-5 обычно достаточно.
CROP_ANALYSIS_FRAMES = 3

# ── Обработка ─────────────────────────────────────────────────
FORCE_BLACK_LEVEL   = None   # None = из метаданных файла
FORCE_COLOR_ID      = None   # None = авто (BAYER_RGGB для S5)
WRITE_TRAILER       = True

# ── Метаданные SER ────────────────────────────────────────────
SER_OBSERVER        = ""
SER_INSTRUMENT      = "Panasonic S5"
SER_TELESCOPE       = ""

# ╚══════════════════════════════════════════════════════════════╝


# ─── SER ColorID ──────────────────────────────────────────────
SER_MONO       = 0
SER_BAYER_RGGB = 8
SER_BAYER_GRBG = 9
SER_BAYER_GBRG = 10
SER_BAYER_BGGR = 11

COLOR_NAMES = {
    0: "MONO", 8: "BAYER_RGGB", 9: "BAYER_GRBG",
    10: "BAYER_GBRG", 11: "BAYER_BGGR", 16: "RGB", 18: "BGR",
}

BAYER_TO_SER_ID = {
    (0, 1, 3, 2): SER_BAYER_RGGB,   # Panasonic S5
    (0, 1, 2, 3): SER_BAYER_RGGB,
    (1, 0, 3, 2): SER_BAYER_GRBG,
    (1, 0, 2, 3): SER_BAYER_GRBG,
    (3, 2, 0, 1): SER_BAYER_GBRG,
    (2, 3, 0, 1): SER_BAYER_GBRG,
    (3, 2, 1, 0): SER_BAYER_BGGR,
    (2, 3, 1, 0): SER_BAYER_BGGR,
}


# ══════════════════════════════════════════════════════════════
# ДЕТЕКЦИЯ ЛУНЫ
# ══════════════════════════════════════════════════════════════

def detect_moon_in_raw(raw_visible: np.ndarray,
                        black_level: int = 511,
                        downsample: int = 8) -> tuple:
    """
    Найти диск Луны в сыром байеровском кадре.

    Алгоритм:
      1. Уменьшаем изображение в downsample раз усреднением блоков
         → байер-паттерн усредняется, получаем чистый сигнал
      2. Вычитаем black_level, нормируем
      3. Пороговая бинаризация → самый крупный яркий объект = Луна
      4. Подбираем окружность по маске (min enclosing circle через моменты)

    Возвращает (cx, cy, radius) в полном разрешении или None при ошибке.
    """
    try:
        from scipy import ndimage
    except ImportError:
        print("  [!] scipy не установлен: pip install scipy")
        print("      Детекция Луны недоступна, кроп пропущен.")
        return None

    h, w = raw_visible.shape
    ds = downsample

    # ── 1. Даунсэмплинг ───────────────────────────────────────
    h_trim = (h // ds) * ds
    w_trim = (w // ds) * ds
    block = raw_visible[:h_trim, :w_trim].astype(np.float32)
    small = block.reshape(h_trim // ds, ds, w_trim // ds, ds).mean(axis=(1, 3))

    # ── 2. Вычитание BL и нормировка ─────────────────────────
    small -= black_level
    np.clip(small, 0, None, out=small)
    vmax = small.max()
    if vmax < 1e-6:
        return None          # пустой кадр
    small /= vmax

    # ── 3. Бинаризация ────────────────────────────────────────
    threshold = 0.05
    binary = (small > threshold)
    binary = ndimage.binary_fill_holes(binary)

    # ── 4. Крупнейший объект ──────────────────────────────────
    labeled, n = ndimage.label(binary)
    if n == 0:
        return None
    sizes = ndimage.sum(binary, labeled, range(1, n + 1))
    moon_label = int(np.argmax(sizes)) + 1
    moon_mask = (labeled == moon_label)

    # ── 5. Параметры диска через моменты изображения ─────────
    ys, xs = np.where(moon_mask)
    if len(xs) == 0:
        return None

    cx_ds = xs.mean()
    cy_ds = ys.mean()

    dist = np.sqrt((xs - cx_ds)**2 + (ys - cy_ds)**2)
    radius_ds = dist.max()

    # ── 6. Масштаб обратно в полное разрешение ───────────────
    cx     = cx_ds     * ds
    cy     = cy_ds     * ds
    radius = radius_ds * ds

    return float(cx), float(cy), float(radius)


def compute_crop_box(cx: float, cy: float, radius: float,
                     img_w: int, img_h: int,
                     padding_px: int) -> tuple:
    """
    Вычислить прямоугольник кропа.

    КРИТИЧНО: x1 и y1 ДОЛЖНЫ быть чётными — иначе байер-паттерн
    сдвинется на 1 пиксель и AS3 получит неверную цветовую матрицу.
    Ширина и высота также должны быть чётными.

    Возвращает (x1, y1, crop_w, crop_h).
    """
    half = int(radius) + padding_px

    x1 = int(cx) - half
    y1 = int(cy) - half
    x2 = int(cx) + half
    y2 = int(cy) + half

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    # ─── БАЙЕР-ВЫРАВНИВАНИЕ ───────────────────────────────────
    x1 = (x1 // 2) * 2
    y1 = (y1 // 2) * 2
    x2 = (x2 // 2) * 2
    y2 = (y2 // 2) * 2

    crop_w = x2 - x1
    crop_h = y2 - y1

    if crop_w <= 0 or crop_h <= 0:
        return None

    return x1, y1, crop_w, crop_h


def analyze_frames_for_crop(rw2_files: list,
                             black_levels: list,
                             raw_pattern: np.ndarray,
                             img_w: int, img_h: int,
                             n_frames: int,
                             padding_px: int) -> tuple:
    """
    Проанализировать первые n_frames кадров, найти позицию Луны
    и вернуть единый стабильный прямоугольник кропа.
    """
    from scipy import ndimage  # noqa: проверка импорта

    bl_avg = int(np.mean(black_levels))

    candidates = min(n_frames, len(rw2_files))
    indices = np.linspace(0, len(rw2_files) - 1, candidates, dtype=int).tolist()

    results = []
    print(f"\n  Детекция Луны (анализ {candidates} кадров):")

    for idx in indices:
        fname = os.path.basename(rw2_files[idx])
        print(f"    [{idx+1:3d}] {fname}  ", end="", flush=True)

        with rawpy.imread(rw2_files[idx]) as raw:
            vis = raw.raw_image_visible.copy()

        det = detect_moon_in_raw(vis, black_level=bl_avg)
        if det is None:
            print("-> не найдена")
            continue

        cx, cy, radius = det
        print(f"-> центр=({cx:.0f}, {cy:.0f})  радиус={radius:.0f} px")
        results.append((cx, cy, radius))

    if not results:
        print("  [!] Луна не обнаружена ни в одном кадре — кроп отключён")
        return None

    cxs     = np.median([r[0] for r in results])
    cys     = np.median([r[1] for r in results])
    radii   = np.median([r[2] for r in results])

    print(f"\n  Медиана: центр=({cxs:.0f}, {cys:.0f})  радиус={radii:.0f} px")
    print(f"  Отступ от края диска: {padding_px} px")

    box = compute_crop_box(cxs, cys, radii, img_w, img_h, padding_px)
    if box is None:
        print("  [!] Ошибка вычисления кропа — кроп отключён")
        return None

    x1, y1, cw, ch = box
    print(f"  Кроп: x={x1}, y={y1},  {cw} x {ch} px")
    x1_even = x1 % 2 == 0
    y1_even = y1 % 2 == 0
    print(f"  (x1 чётное={x1_even}, y1 чётное={y1_even}) <- байер OK")

    return box


# ══════════════════════════════════════════════════════════════
# SER ФОРМАТ
# ══════════════════════════════════════════════════════════════

def detect_color_id(raw_pattern: np.ndarray) -> int:
    key = tuple(int(raw_pattern[r, c]) for r in range(2) for c in range(2))
    color_id = BAYER_TO_SER_ID.get(key)
    if color_id is None:
        print(f"  [!] Неизвестный байер-паттерн {key} -> MONO(0)")
        return SER_MONO
    return color_id


def to_ser_ticks(dt: datetime) -> int:
    """Конвертировать datetime в SER-таймштамп (.NET ticks с 0001-01-01)."""
    DOTNET_EPOCH_TICKS = 621_355_968_000_000_000
    unix_sec = dt.replace(tzinfo=timezone.utc).timestamp()
    return int(DOTNET_EPOCH_TICKS + unix_sec * 10_000_000)


def build_ser_header(width: int, height: int, frame_count: int,
                     color_id: int, bit_depth: int,
                     observer: str, instrument: str, telescope: str,
                     timestamp_ticks: int) -> bytes:
    """
    178-байтный заголовок SER v3.
    """
    def pad40(s: str) -> bytes:
        b = s.encode("ascii", errors="replace")[:40]
        return b + b"\x00" * (40 - len(b))

    header = struct.pack(
        "<14s i i i i i i i 40s 40s 40s Q Q",
        b"LUCAM-RECORDER",
        0,               # LuID
        color_id,
        0,               # LittleEndian=0 -> LE (де-факто инвертировано)
        width,
        height,
        bit_depth,
        frame_count,
        pad40(observer),
        pad40(instrument),
        pad40(telescope),
        timestamp_ticks,
        timestamp_ticks,
    )
    assert len(header) == 178, f"BUG: header={len(header)} bytes"
    return header


# ══════════════════════════════════════════════════════════════
# ОБРАБОТКА КАДРА
# ══════════════════════════════════════════════════════════════

def process_frame(raw_visible: np.ndarray,
                  black_levels: list,
                  raw_pattern: np.ndarray,
                  crop_box: tuple = None) -> np.ndarray:
    """
    Вычесть BL (per-channel Bayer-aware), масштабировать до 16 бит,
    опционально кропнуть.
    """
    data = raw_visible.astype(np.int32)

    for row in range(2):
        for col in range(2):
            ch_idx = int(raw_pattern[row, col])
            data[row::2, col::2] -= black_levels[ch_idx]

    np.clip(data, 0, None, out=data)

    # 14-бит -> 16-бит (× 4)
    data *= 4
    np.clip(data, 0, 65535, out=data)

    frame = data.astype(np.uint16)

    if crop_box is not None:
        x1, y1, cw, ch = crop_box
        frame = frame[y1:y1 + ch, x1:x1 + cw]

    return frame


# ══════════════════════════════════════════════════════════════
# ВЕРИФИКАЦИЯ
# ══════════════════════════════════════════════════════════════

def verify_ser(ser_path: str, expected_w: int, expected_h: int,
               expected_frames: int) -> None:
    print(f"\n{'─'*65}")
    print(f"Верификация: {ser_path}")

    with open(ser_path, "rb") as f:
        raw_hdr = f.read(178)

    (file_id, lu_id, color_id, le_flag,
     img_w, img_h, bit_depth, frame_cnt,
     _obs, instrument, _tel,
     dt, dt_utc) = struct.unpack("<14s i i i i i i i 40s 40s 40s Q Q", raw_hdr)

    instr_str = instrument.rstrip(b"\x00").decode("ascii", errors="replace")

    print(f"  FileID      : {file_id}")
    print(f"  ColorID     : {color_id} = {COLOR_NAMES.get(color_id, '?')}")
    print(f"  LittleEndian: {le_flag}  (0=LE OK)")
    print(f"  Width       : {img_w}  (ожидалось {expected_w})")
    print(f"  Height      : {img_h}  (ожидалось {expected_h})")
    print(f"  BitDepth    : {bit_depth}")
    print(f"  FrameCount  : {frame_cnt}  (ожидалось {expected_frames})")
    print(f"  Instrument  : {instr_str}")

    ok = (file_id == b"LUCAM-RECORDER" and
          img_w == expected_w and
          img_h == expected_h and
          frame_cnt == expected_frames and
          bit_depth == 16)
    print(f"  Статус      : {'OK' if ok else 'ОШИБКА'}")


# ══════════════════════════════════════════════════════════════
# ОСНОВНАЯ ФУНКЦИЯ
# ══════════════════════════════════════════════════════════════

def run_conversion(input_folder: str, output_file: str,
                   enable_crop: bool = True,
                   crop_padding_px: int = 300,
                   crop_analysis_frames: int = 3,
                   force_black_level=None,
                   force_color_id=None,
                   write_trailer: bool = True,
                   observer: str = "",
                   instrument: str = "Panasonic S5",
                   telescope: str = "") -> None:

    SEP = "=" * 65

    # ── 1. Поиск файлов ──────────────────────────────────────
    rw2_files = sorted(
        glob.glob(os.path.join(input_folder, "*.RW2")) +
        glob.glob(os.path.join(input_folder, "*.rw2"))
    )
    seen, uniq = set(), []
    for f in rw2_files:
        k = f.lower()
        if k not in seen:
            seen.add(k); uniq.append(f)
    rw2_files = uniq

    if not rw2_files:
        print(f"ОШИБКА: RW2-файлы не найдены: {input_folder}")
        sys.exit(1)

    print(SEP)
    print(f"Входная папка:  {input_folder}")
    print(f"Найдено файлов: {len(rw2_files)}")
    for f in rw2_files:
        print(f"  {os.path.basename(f)}")

    # ── 2. Параметры из первого файла ────────────────────────
    print(f"\n{SEP}")
    print(f"Параметры: {os.path.basename(rw2_files[0])}")

    with rawpy.imread(rw2_files[0]) as raw:
        raw_pattern = raw.raw_pattern.copy()
        bl_meta     = list(raw.black_level_per_channel)
        white_level = raw.white_level
        h0, w0      = raw.raw_image_visible.shape

    print(f"  Полный кадр           : {w0} x {h0} px")
    print(f"  raw_pattern           : {raw_pattern.tolist()}")
    print(f"  Черный уровень        : {bl_meta}")
    print(f"  Белый уровень         : {white_level}")

    black_levels = [force_black_level] * 4 if force_black_level else bl_meta
    color_id     = force_color_id if force_color_id is not None else detect_color_id(raw_pattern)

    print(f"  ColorID               : {color_id} = {COLOR_NAMES.get(color_id, '?')}")

    # ── 3. Детекция Луны и кроп ──────────────────────────────
    crop_box = None
    out_w, out_h = w0, h0

    if enable_crop:
        print(f"\n{SEP}")
        print(f"ДЕТЕКЦИЯ ЛУНЫ")
        try:
            import scipy  # noqa: проверка наличия
            crop_box = analyze_frames_for_crop(
                rw2_files, black_levels, raw_pattern,
                w0, h0,
                n_frames=crop_analysis_frames,
                padding_px=crop_padding_px,
            )
        except ImportError:
            print("  [!] scipy не установлен (pip install scipy) — кроп пропущен")
            crop_box = None

        if crop_box:
            _, _, out_w, out_h = crop_box
        else:
            print("  Кроп отключён — используется полный кадр")
    else:
        print(f"\n  Кроп: отключён (ENABLE_CROP=False)")

    # ── 4. Информация о выходном файле ───────────────────────
    print(f"\n{SEP}")
    print(f"Параметры выходного SER:")
    print(f"  Разрешение кадра      : {out_w} x {out_h} px")
    frame_bytes = out_w * out_h * 2
    total_mb = (178 + len(rw2_files) * frame_bytes) / 1_048_576
    print(f"  Размер кадра          : {frame_bytes / 1_048_576:.1f} МБ")
    print(f"  Ожидаемый размер SER  : {total_mb:.0f} МБ")

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    # ── 5. Запись SER ────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"Запись: {output_file}\n")

    ts = to_ser_ticks(datetime.now(timezone.utc))
    frame_timestamps = []

    with open(output_file, "wb") as f:
        f.write(build_ser_header(
            out_w, out_h, len(rw2_files),
            color_id, 16,
            observer, instrument, telescope, ts
        ))

        for i, rw2_path in enumerate(rw2_files, 1):
            fname = os.path.basename(rw2_path)
            print(f"  [{i:3d}/{len(rw2_files)}] {fname}", end="  ", flush=True)

            with rawpy.imread(rw2_path) as raw:
                vis = raw.raw_image_visible.copy()

            frame = process_frame(vis, black_levels, raw_pattern, crop_box)
            f.write(frame.tobytes())

            frame_timestamps.append(to_ser_ticks(datetime.now(timezone.utc)))

            print(f"min={frame.min():5d}  max={frame.max():5d}  "
                  f"mean={frame.mean():7.1f}  size={frame.shape[1]}x{frame.shape[0]}")

        if write_trailer:
            for ts_f in frame_timestamps:
                f.write(struct.pack("<Q", ts_f))

    actual_mb = os.path.getsize(output_file) / 1_048_576

    print(f"\n{SEP}")
    print(f"ГОТОВО!")
    print(f"  Файл          : {output_file}")
    print(f"  Кадров        : {len(rw2_files)}")
    print(f"  Разрешение    : {out_w} x {out_h} px")
    print(f"  Бит/пиксель   : 16")
    print(f"  ColorID       : {color_id} ({COLOR_NAMES.get(color_id, '?')})")
    print(f"  Размер файла  : {actual_mb:.1f} МБ")
    if crop_box:
        x1, y1, cw, ch = crop_box
        print(f"  Кроп          : x={x1}, y={y1},  {cw}x{ch} px  "
              f"(x1%2={x1%2}, y1%2={y1%2})")
    print(SEP)
    print()
    print("В AutoStakkert3:")
    print("  Color  -> Force Bayer RGGB")
    print("  Drizzle -> 1.5x  (= Bayer Drizzle автоматически)")
    print(SEP)

    verify_ser(output_file, out_w, out_h, len(rw2_files))


# ══════════════════════════════════════════════════════════════
# ТОЧКА ВХОДА
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) == 3:
        in_folder = sys.argv[1]
        out_file  = sys.argv[2]
    elif len(sys.argv) == 1:
        in_folder = INPUT_FOLDER
        out_file  = OUTPUT_FILE
    else:
        print("Использование:")
        print("  python rw2_to_ser.py")
        print("  python rw2_to_ser.py <папка> <выход.ser>")
        sys.exit(1)

    run_conversion(
        input_folder=in_folder,
        output_file=out_file,
        enable_crop=ENABLE_CROP,
        crop_padding_px=CROP_PADDING_PX,
        crop_analysis_frames=CROP_ANALYSIS_FRAMES,
        force_black_level=FORCE_BLACK_LEVEL,
        force_color_id=FORCE_COLOR_ID,
        write_trailer=WRITE_TRAILER,
        observer=SER_OBSERVER,
        instrument=SER_INSTRUMENT,
        telescope=SER_TELESCOPE,
    )