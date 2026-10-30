"""Single-image or recursive-directory inference for the released SegFormer-B5."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from contextlib import contextmanager, nullcontext
from itertools import islice
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


HERE = Path(__file__).resolve().parent
N_CLASSES = 14
TILE_SIZE = 512
STRIDE = 384
MAX_IMAGE_PIXELS = 250_000_000
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_device(value: str = "auto") -> torch.device:
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("--device must be cpu, cuda, cuda:N, or auto")
    if device.type == "cuda" and (
        not torch.cuda.is_available()
        or (device.index is not None and device.index >= torch.cuda.device_count())
    ):
        raise ValueError("Requested CUDA device is unavailable; use --device cpu")
    return device


def load_labels(path: Path = HERE / "labels.json") -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    classes = metadata.get("classes", [])
    palette = np.asarray(metadata.get("palette", []))
    if (
        len(classes) != N_CLASSES
        or any(not isinstance(name, str) or not name for name in classes)
        or len(set(classes)) != N_CLASSES
        or palette.shape != (N_CLASSES, 3)
        or not np.issubdtype(palette.dtype, np.integer)
        or np.any((palette < 0) | (palette > 255))
        or metadata.get("raw_ids") != list(range(1, N_CLASSES + 1))
        or metadata.get("ignore_index") != 255
    ):
        raise ValueError("labels.json must contain 14 classes, RGB colors, IDs 1..14 and ignore_index=255")
    return metadata


def load_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    from mmcv import Config
    from mmseg.models import build_segmentor

    config = Config.fromfile(str(HERE / "model_config.py"))
    model = build_segmentor(config.model)
    state = torch.load(str(checkpoint), map_location="cpu")
    if not isinstance(state, dict) or not state or any(
        not isinstance(key, str) or not isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise ValueError("Checkpoint must be the released plain tensor state_dict")
    model.load_state_dict(state, strict=True)
    del state
    return model.to(device).eval()


@torch.inference_mode()
def predict_tiles(
    model: torch.nn.Module, crops: Sequence[Image.Image], device: torch.device
) -> np.ndarray:
    arrays = []
    for crop in crops:
        array = np.array(crop, dtype=np.float32)
        array /= 255.0
        array -= MEAN
        array /= STD
        arrays.append(array.transpose(2, 0, 1))
    tensor = torch.from_numpy(np.ascontiguousarray(np.stack(arrays))).to(device)
    context = torch.cuda.amp.autocast() if device.type == "cuda" else nullcontext()
    with context:
        logits = model.encode_decode(tensor, [{} for _ in crops])
    logits = F.interpolate(
        logits.float(), size=(TILE_SIZE, TILE_SIZE), mode="bilinear", align_corners=False
    )
    result = logits.cpu().numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError("Model returned non-finite logits")
    return result


def sliding_starts(length: int, tile: int, stride: int) -> List[int]:
    if min(length, tile, stride) < 1 or stride > tile:
        raise ValueError("Require positive dimensions and 1 <= stride <= tile")
    last = max(0, length - tile)
    return sorted(set(range(0, last + 1, stride)) | {last})


def sliding_predict(
    image: Image.Image,
    batch_predict: Callable[[Sequence[Image.Image]], np.ndarray],
    batch_size: int = 1,
    tile: int = TILE_SIZE,
    stride: int = STRIDE,
) -> Tuple[np.ndarray, dict]:
    """Fuse overlapping logits in a CPU strip, using the training script's order."""
    width, height = image.size
    horizontal = width > height
    major, minor = (width, height) if horizontal else (height, width)
    starts = sliding_starts(major, tile, stride)
    crosses = sliding_starts(minor, tile, stride)
    if batch_size < 1:
        raise ValueError("--batch-size must be positive")
    accumulator = np.zeros((N_CLASSES, min(tile, major), minor), dtype=np.float32)
    labels = np.empty((height, width), dtype=np.uint8)
    positions = iter((row, start, cross) for row, start in enumerate(starts) for cross in crosses)
    while True:
        coordinates = list(islice(positions, batch_size))
        if not coordinates:
            break
        crops = []
        try:
            for _, start, cross in coordinates:
                x, y = (start, cross) if horizontal else (cross, start)
                crop = image.crop((x, y, min(width, x + tile), min(height, y + tile))).convert("RGB")
                crops.append(crop)
                if crop.size != (tile, tile):
                    raw = np.pad(
                        np.asarray(crop),
                        ((0, tile - crop.height), (0, tile - crop.width), (0, 0)),
                        mode="edge",
                    )
                    crop.close()
                    crops[-1] = Image.fromarray(raw)
            batch_logits = batch_predict(crops)
            if len(batch_logits) != len(coordinates):
                raise ValueError("Tile output count differs from batch input count")
            for (row, start, cross), logits in zip(coordinates, batch_logits):
                x, y = (start, cross) if horizontal else (cross, start)
                right, bottom = min(width, x + tile), min(height, y + tile)
                if logits.shape != (N_CLASSES, tile, tile) or not np.isfinite(logits).all():
                    raise ValueError("Tile predictor must return finite 14-channel logits at tile size")
                valid = logits[:, :bottom - y, :right - x]
                if horizontal:
                    valid = valid.transpose(0, 2, 1)
                accumulator[:, :valid.shape[1], cross:cross + valid.shape[2]] += valid
                if cross != crosses[-1]:
                    continue
                end = starts[row + 1] if row + 1 < len(starts) else major
                count = end - start
                # Every class has the same overlap count: sum and mean have the same argmax.
                merged = accumulator[:, :count].argmax(0).astype(np.uint8) + 1
                if horizontal:
                    labels[:, start:end] = merged.T
                else:
                    labels[start:end] = merged
                if row + 1 < len(starts):
                    retained = accumulator.shape[1] - count
                    accumulator[:, :retained] = accumulator[:, count:].copy()
                    accumulator[:, retained:] = 0
        finally:
            for crop in crops:
                crop.close()
    return labels, dict(tiles=len(starts) * len(crosses), fusion_buffer_bytes=accumulator.nbytes)


def resize_predict(
    image: Image.Image, batch_predict: Callable[[Sequence[Image.Image]], np.ndarray]
) -> np.ndarray:
    with image.convert("RGB") as rgb:
        with rgb.resize((TILE_SIZE, TILE_SIZE), Image.Resampling.BILINEAR) as resized:
            logits = batch_predict([resized])[0]
    if logits.shape != (N_CLASSES, TILE_SIZE, TILE_SIZE) or not np.isfinite(logits).all():
        raise ValueError("Resize predictor must return finite 14-channel 512x512 logits")
    best = np.full((image.height, image.width), -np.inf, dtype=np.float32)
    labels = np.ones((image.height, image.width), dtype=np.uint8)
    # Upsample logits before argmax, one class at a time to avoid a full 14-channel image.
    for index, plane in enumerate(logits):
        scores = F.interpolate(
            torch.from_numpy(plane)[None, None],
            size=(image.height, image.width),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        selected = scores > best
        best[selected] = scores[selected]
        labels[selected] = index + 1
    return labels


@contextmanager
def open_prediction_image(path: Path) -> Iterator[Image.Image]:
    previous = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ValueError("Image exceeds the 250,000,000 pixel limit: " + str(path))
                image.load()
                yield image
    finally:
        Image.MAX_IMAGE_PIXELS = previous


def collect_inputs(input_path: Path, output: Path) -> Tuple[Path, List[Path]]:
    source = Path(input_path).resolve()
    output = Path(output).resolve()
    if source.is_file():
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError("Unsupported image extension: " + source.suffix)
        return source.parent, [source]
    if not source.is_dir():
        raise FileNotFoundError("Input image or directory does not exist: " + str(source))
    if output == source or output in source.parents or source in output.parents:
        raise ValueError("Input and output directories must be separate; neither may contain the other")
    paths = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise ValueError("No supported images found in " + str(source))
    return source, paths


def prediction_paths(
    output: Path, source_root: Path, source: Path, image_size: Optional[Tuple[int, int]] = None
) -> Dict[str, Path]:
    relative = source.relative_to(source_root)
    overlay_suffix = ".png" if image_size is not None and max(image_size) >= 65500 else ".jpg"
    return {
        kind: output / kind / relative.parent / (relative.name + suffix)
        for kind, suffix in (
            ("labels", "_labels.png"),
            ("color", "_color.png"),
            ("overlay", "_overlay" + overlay_suffix),
        )
    }


def save_prediction(
    image: Image.Image, labels: np.ndarray, palette: np.ndarray, destinations: Dict[str, Path]
) -> None:
    if labels.shape != (image.height, image.width) or labels.dtype != np.uint8:
        raise ValueError("Prediction must be a uint8 label image at the original resolution")
    if labels.min() < 1 or labels.max() > N_CLASSES:
        raise ValueError("Predicted label IDs must be 1..14")
    existing = next((path for path in destinations.values() if path.exists()), None)
    if existing is not None:
        raise FileExistsError("Output already exists; choose another --output directory: " + str(existing))
    for path in destinations.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        for path in destinations.values():
            with path.open("xb"):
                pass
            created.append(path)
        with Image.fromarray(labels) as raw:
            raw.save(destinations["labels"], compress_level=3)
        band = max(1, min(256, 1_048_576 // image.width))
        with Image.new("RGB", image.size) as canvas:
            for top in range(0, image.height, band):
                bottom = min(top + band, image.height)
                with Image.fromarray(palette[labels[top:bottom] - 1]) as color:
                    canvas.paste(color, (0, top))
            canvas.save(destinations["color"], compress_level=3)
            for top in range(0, image.height, band):
                bottom = min(top + band, image.height)
                box = (0, top, image.width, bottom)
                with image.crop(box).convert("RGB") as source, canvas.crop(box) as color:
                    with Image.blend(source, color, 0.45) as overlay:
                        canvas.paste(overlay, (0, top))
            if destinations["overlay"].suffix == ".png":
                canvas.save(destinations["overlay"], compress_level=3)
            else:
                canvas.save(destinations["overlay"], quality=95)
    except BaseException:
        for path in created:
            path.unlink()
        raise


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Image file or recursively scanned directory")
    parser.add_argument("--output", required=True, type=Path, help="Separate directory for labels/color/overlay")
    parser.add_argument("--checkpoint", type=Path, default=HERE / "shanshuiseg_b5.pth")
    parser.add_argument("--device", default="auto", help="auto (default), cpu, cuda, or cuda:N")
    parser.add_argument("--mode", choices=("sliding", "resize"), default="sliding")
    parser.add_argument("--batch-size", type=int, default=1, help="Sliding-window batch size (default: 1)")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    try:
        source_root, paths = collect_inputs(args.input, args.output)
        device = resolve_device(args.device)
        metadata = load_labels()
        palette = np.asarray(metadata["palette"], dtype=np.uint8)
        model = load_model(args.checkpoint, device)
        batch_predict = lambda crops: predict_tiles(model, crops, device)
        print("Device: %s | Mode: %s | Images: %d" % (device, args.mode, len(paths)), flush=True)
        for index, path in enumerate(paths, 1):
            print("[%d/%d] %s" % (index, len(paths), path), flush=True)
            with open_prediction_image(path) as image:
                destinations = prediction_paths(args.output, source_root, path, image.size)
                if any(destination.resolve() == path.resolve() for destination in destinations.values()):
                    raise ValueError("Output would overwrite the input image")
                if any(destination.exists() for destination in destinations.values()):
                    raise FileExistsError("Prediction output already exists; choose another --output directory")
                if args.mode == "sliding":
                    labels, info = sliding_predict(image, batch_predict, args.batch_size)
                    print("  Tiles: %d" % info["tiles"], flush=True)
                else:
                    labels = resize_predict(image, batch_predict)
                save_prediction(image, labels, palette, destinations)
        print("Saved predictions to " + str(args.output.resolve()), flush=True)
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print("Error: %s" % error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
