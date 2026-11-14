"""Join adjacent Moon tiles into larger, overlapping terrain image patches."""
import argparse
import json
import logging
from pathlib import Path
import re

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(r"moon_z8_r(\d+)_c(\d+)\.jpg", re.IGNORECASE)


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return number


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Must be zero or greater")
    return number


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "resources/zoom_8")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "resources/combined_zoom_8")
    parser.add_argument("--start-row", type=nonnegative, default=6)
    parser.add_argument("--start-col", type=nonnegative, default=0)
    parser.add_argument("--rows", type=positive, default=4, help="Tile rows per image")
    parser.add_argument("--cols", type=positive, default=4, help="Tile columns per image")
    parser.add_argument("--stride", type=positive, default=2,
                        help="Move this many tile positions between windows (default: 2)")
    parser.add_argument("--limit", type=positive, help="Maximum images to generate")
    return parser.parse_args()


def combine(args):
    source = args.input_dir.resolve()
    output = args.output_dir.resolve()
    if source == output:
        raise ValueError("Use an output folder separate from the input tiles")
    if not source.is_dir():
        raise FileNotFoundError(source)
    tiles = {}
    for path in source.iterdir():
        match = PATTERN.fullmatch(path.name)
        if match and path.is_file():
            key = tuple(map(int, match.groups()))
            if key in tiles:
                raise ValueError(f"Duplicate tile coordinates: {key}")
            tiles[key] = path
    if not tiles:
        raise ValueError("No moon_z8_r<row>_c<column>.jpg tiles found")
    with Image.open(next(iter(tiles.values()))) as tile:
        width, height = tile.size
    pixel_count = width * height * args.cols * args.rows
    # Keep working memory reasonable on a 16 GB laptop.
    if pixel_count > 64_000_000:
        raise ValueError("Window exceeds 64 million pixels; reduce --rows or --cols")
    output.mkdir(parents=True, exist_ok=True)
    max_row = max(row for row, col in tiles)
    max_col = max(col for row, col in tiles)
    saved = skipped = 0
    for row in range(args.start_row, max_row - args.rows + 2, args.stride):
        for col in range(args.start_col, max_col - args.cols + 2, args.stride):
            positions = [(r, c) for r in range(row, row + args.rows)
                         for c in range(col, col + args.cols)]
            # Never invent black terrain where a tile is absent.
            if any(position not in tiles for position in positions):
                skipped += 1
                continue
            canvas = Image.new("RGB", (width * args.cols, height * args.rows))
            inputs = []
            try:
                for r, c in positions:
                    path = tiles[r, c]
                    before = path.stat()
                    with Image.open(path) as tile:
                        if tile.size != (width, height):
                            raise ValueError(f"Unexpected tile size: {path.name}: {tile.size}")
                        canvas.paste(tile.convert("RGB"), ((c - col) * width, (r - row) * height))
                    after = path.stat()
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        raise OSError(f"Tile changed while reading: {path.name}")
                    inputs.append({"filename": path.name, "row": r, "col": c,
                                   "size_bytes": after.st_size, "mtime_ns": after.st_mtime_ns})
                name = f"moon_z8_r{row}-{row + args.rows - 1}_c{col}-{col + args.cols - 1}"
                temporary = output / f"{name}.png.part"
                canvas.save(temporary, format="PNG")
                temporary.replace(output / f"{name}.png")
                metadata = {"source_dir": str(source), "image": f"{name}.png",
                            "size_pixels": list(canvas.size), "tile_size_pixels": [width, height],
                            "layout": "columns increase left-to-right; rows increase top-to-bottom",
                            "tiles": inputs}
                temporary = output / f"{name}.json.part"
                temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
                temporary.replace(output / f"{name}.json")
                saved += 1
                logging.info("Saved %s (%s x %s pixels)", name, *canvas.size)
            except OSError as error:
                skipped += 1
                logging.warning("Skipped window r%s c%s: %s", row, col, error)
            finally:
                canvas.close()
            if args.limit is not None and saved >= args.limit:
                return saved, skipped
    return saved, skipped


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    try:
        args = parse_args()
        saved, skipped = combine(args)
        logging.info("Saved %s images; skipped %s incomplete/unreadable windows. Output: %s",
                     saved, skipped, args.output_dir.resolve())
        return 0 if saved else 1
    except (OSError, ValueError) as error:
        logging.error("%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
