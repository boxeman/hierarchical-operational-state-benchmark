from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import shutil
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.colors import Color, HexColor, white
from reportlab.pdfgen import canvas


TAG_RE = re.compile(r"\{.*\}(.+)")
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
PATH_TOKEN_RE = re.compile(r"[MLCZ]|-?\d+(?:\.\d+)?")


def local_name(tag: str) -> str:
    match = TAG_RE.match(tag)
    return match.group(1) if match else tag


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_length(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    match = NUM_RE.search(value)
    return float(match.group(0)) if match else default


def parse_color(value: str | None, fallback: str | None = None) -> str | None:
    if value is None:
        return fallback
    value = value.strip()
    if value in {"none", "transparent"}:
        return None
    if value.startswith("#") and len(value) in {4, 7}:
        if len(value) == 4:
            return "#" + "".join(ch * 2 for ch in value[1:])
        return value
    return fallback


def parse_opacity(style: dict[str, str], inherited: float = 1.0) -> float:
    value = style.get("opacity")
    if value is None:
        return inherited
    try:
        return max(0.0, min(1.0, float(value))) * inherited
    except ValueError:
        return inherited


def parse_style_blocks(root: ET.Element) -> dict[str, dict[str, str]]:
    classes: dict[str, dict[str, str]] = {}
    for elem in root.iter():
        if local_name(elem.tag) != "style" or not elem.text:
            continue
        for cls, body in re.findall(r"\.([A-Za-z0-9_-]+)\s*\{([^}]+)\}", elem.text, flags=re.S):
            props: dict[str, str] = {}
            for item in body.split(";"):
                if ":" in item:
                    key, value = item.split(":", 1)
                    props[key.strip()] = value.strip()
            classes[cls] = props
    return classes


def merged_style(elem: ET.Element, classes: dict[str, dict[str, str]]) -> dict[str, str]:
    style: dict[str, str] = {}
    for cls in elem.attrib.get("class", "").split():
        style.update(classes.get(cls, {}))
    for key, value in elem.attrib.items():
        if key == "style":
            for item in value.split(";"):
                if ":" in item:
                    k, v = item.split(":", 1)
                    style[k.strip()] = v.strip()
        else:
            style[key] = value
    return style


def svg_size(root: ET.Element) -> tuple[int, int]:
    width = parse_length(root.attrib.get("width"))
    height = parse_length(root.attrib.get("height"))
    if width and height:
        return int(round(width)), int(round(height))
    view_box = root.attrib.get("viewBox") or root.attrib.get("viewbox")
    if view_box:
        parts = [float(p) for p in re.split(r"[\s,]+", view_box.strip()) if p]
        if len(parts) == 4:
            return int(round(parts[2])), int(round(parts[3]))
    return 1600, 900


def transform_from_attr(attr: str | None, current: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    sx, sy, tx, ty = current
    if not attr:
        return current
    for name, body in re.findall(r"(translate|scale)\(([^)]+)\)", attr):
        nums = [float(n) for n in NUM_RE.findall(body)]
        if name == "translate" and nums:
            tx += sx * nums[0]
            ty += sy * (nums[1] if len(nums) > 1 else 0.0)
        elif name == "scale" and nums:
            sx *= nums[0]
            sy *= nums[1] if len(nums) > 1 else nums[0]
    return sx, sy, tx, ty


def pt(x: float, y: float, transform: tuple[float, float, float, float]) -> tuple[float, float]:
    sx, sy, tx, ty = transform
    return tx + sx * x, ty + sy * y


def pdf_color(value: str | None, opacity: float = 1.0):
    if value is None:
        return None
    color = HexColor(value)
    return Color(color.red, color.green, color.blue, alpha=opacity)


def pil_color(value: str | None, opacity: float = 1.0):
    if value is None:
        return None
    value = value.lstrip("#")
    r, g, b = int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)
    return (r, g, b, int(round(255 * opacity)))


def font_name(weight: str | None) -> str:
    return "Helvetica-Bold" if weight and weight not in {"400", "normal"} else "Helvetica"


def pil_font(size: int, bold: bool = False):
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), max(6, size))
    return ImageFont.load_default()


class Renderer:
    def __init__(self, svg_path: Path, pdf_path: Path, tiff_path: Path, dpi: int):
        self.svg_path = svg_path
        self.root = ET.parse(svg_path).getroot()
        self.classes = parse_style_blocks(self.root)
        self.width, self.height = svg_size(self.root)
        self.pdf = canvas.Canvas(str(pdf_path), pagesize=(self.width, self.height))
        self.pdf.setFillColor(white)
        self.pdf.rect(0, 0, self.width, self.height, stroke=0, fill=1)
        self.dpi = dpi
        self.scale = dpi / 96
        self.image = Image.new("RGBA", (int(self.width * self.scale), int(self.height * self.scale)), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.tiff_path = tiff_path

    def render(self) -> None:
        for child in self.root:
            self.render_elem(child, (1.0, 1.0, 0.0, 0.0), 1.0)
        self.pdf.showPage()
        self.pdf.save()
        rgb = self.image.convert("RGB")
        rgb.save(self.tiff_path, compression="tiff_lzw", dpi=(self.dpi, self.dpi))

    def ypdf(self, y: float) -> float:
        return self.height - y

    def sp(self, x: float, y: float) -> tuple[float, float]:
        return x * self.scale, y * self.scale

    def render_elem(self, elem: ET.Element, transform: tuple[float, float, float, float], opacity: float) -> None:
        name = local_name(elem.tag)
        if name in {"defs", "style", "title", "desc"}:
            return
        transform = transform_from_attr(elem.attrib.get("transform"), transform)
        style = merged_style(elem, self.classes)
        opacity = parse_opacity(style, opacity)

        if name == "g":
            for child in elem:
                self.render_elem(child, transform, opacity)
            return

        if name == "rect":
            self.rect(style, transform, opacity)
        elif name == "line":
            self.line(style, transform, opacity)
        elif name == "text":
            self.text(elem, style, transform, opacity)
        elif name == "circle":
            self.circle(style, transform, opacity)
        elif name == "polyline":
            self.polyline(style, transform, opacity)
        elif name == "path":
            self.path(style, transform, opacity)

    def rect(self, style: dict[str, str], transform, opacity: float) -> None:
        x, y = pt(parse_length(style.get("x")), parse_length(style.get("y")), transform)
        width = parse_length(style.get("width"), self.width)
        height = parse_length(style.get("height"), self.height)
        if style.get("width") == "100%":
            width = self.width
        if style.get("height") == "100%":
            height = self.height
        sx, sy, _, _ = transform
        width *= sx
        height *= sy
        rx = parse_length(style.get("rx")) * sx
        fill = parse_color(style.get("fill"))
        stroke = parse_color(style.get("stroke"))
        stroke_width = parse_length(style.get("stroke-width"), 1.0) * ((sx + sy) / 2)

        self.pdf.setLineWidth(stroke_width)
        if stroke:
            self.pdf.setStrokeColor(pdf_color(stroke, opacity))
        if fill:
            self.pdf.setFillColor(pdf_color(fill, opacity))
        if rx > 0:
            self.pdf.roundRect(x, self.ypdf(y + height), width, height, rx, stroke=1 if stroke else 0, fill=1 if fill else 0)
        else:
            self.pdf.rect(x, self.ypdf(y + height), width, height, stroke=1 if stroke else 0, fill=1 if fill else 0)

        xy = [*self.sp(x, y), *self.sp(x + width, y + height)]
        if rx > 0:
            self.draw.rounded_rectangle(xy, radius=rx * self.scale, fill=pil_color(fill, opacity), outline=pil_color(stroke, opacity), width=max(1, int(stroke_width * self.scale)) if stroke else 1)
        else:
            self.draw.rectangle(xy, fill=pil_color(fill, opacity), outline=pil_color(stroke, opacity), width=max(1, int(stroke_width * self.scale)) if stroke else 1)

    def line(self, style: dict[str, str], transform, opacity: float) -> None:
        x1, y1 = pt(parse_length(style.get("x1")), parse_length(style.get("y1")), transform)
        x2, y2 = pt(parse_length(style.get("x2")), parse_length(style.get("y2")), transform)
        stroke = parse_color(style.get("stroke"), "#000000")
        sx, sy, _, _ = transform
        stroke_width = parse_length(style.get("stroke-width"), 1.0) * ((sx + sy) / 2)
        self.pdf.setStrokeColor(pdf_color(stroke, opacity))
        self.pdf.setLineWidth(stroke_width)
        self.pdf.line(x1, self.ypdf(y1), x2, self.ypdf(y2))
        self.draw.line([self.sp(x1, y1), self.sp(x2, y2)], fill=pil_color(stroke, opacity), width=max(1, int(stroke_width * self.scale)))

    def text(self, elem: ET.Element, style: dict[str, str], transform, opacity: float) -> None:
        text = "".join(elem.itertext()).strip()
        if not text:
            return
        x, y = pt(parse_length(style.get("x")), parse_length(style.get("y")), transform)
        sx, sy, _, _ = transform
        size = parse_length(style.get("font-size"), 12.0) * ((sx + sy) / 2)
        weight = style.get("font-weight")
        fill = parse_color(style.get("fill"), "#000000")
        anchor = style.get("text-anchor", "start")

        self.pdf.setFillColor(pdf_color(fill, opacity))
        self.pdf.setFont(font_name(weight), size)
        text_width = self.pdf.stringWidth(text, font_name(weight), size)
        draw_x = x - text_width / 2 if anchor == "middle" else x - text_width if anchor == "end" else x
        self.pdf.drawString(draw_x, self.ypdf(y), text)

        font = pil_font(int(round(size * self.scale)), bold=bool(weight and weight not in {"400", "normal"}))
        bbox = self.draw.textbbox((0, 0), text, font=font)
        pix_width = bbox[2] - bbox[0]
        px, py = self.sp(x, y - size)
        if anchor == "middle":
            px -= pix_width / 2
        elif anchor == "end":
            px -= pix_width
        self.draw.text((px, py), text, font=font, fill=pil_color(fill, opacity))

    def circle(self, style: dict[str, str], transform, opacity: float) -> None:
        cx, cy = pt(parse_length(style.get("cx")), parse_length(style.get("cy")), transform)
        sx, sy, _, _ = transform
        r = parse_length(style.get("r"), 1.0) * ((sx + sy) / 2)
        fill = parse_color(style.get("fill"))
        stroke = parse_color(style.get("stroke"))
        stroke_width = parse_length(style.get("stroke-width"), 1.0) * ((sx + sy) / 2)
        if fill:
            self.pdf.setFillColor(pdf_color(fill, opacity))
        if stroke:
            self.pdf.setStrokeColor(pdf_color(stroke, opacity))
        self.pdf.setLineWidth(stroke_width)
        self.pdf.circle(cx, self.ypdf(cy), r, stroke=1 if stroke else 0, fill=1 if fill else 0)
        xy = [*self.sp(cx - r, cy - r), *self.sp(cx + r, cy + r)]
        self.draw.ellipse(xy, fill=pil_color(fill, opacity), outline=pil_color(stroke, opacity), width=max(1, int(stroke_width * self.scale)) if stroke else 1)

    def polyline(self, style: dict[str, str], transform, opacity: float) -> None:
        raw = style.get("points", "")
        nums = [float(n) for n in NUM_RE.findall(raw)]
        points = [pt(nums[i], nums[i + 1], transform) for i in range(0, len(nums) - 1, 2)]
        if len(points) < 2:
            return
        stroke = parse_color(style.get("stroke"), "#000000")
        sx, sy, _, _ = transform
        stroke_width = parse_length(style.get("stroke-width"), 1.0) * ((sx + sy) / 2)
        self.pdf.setStrokeColor(pdf_color(stroke, opacity))
        self.pdf.setLineWidth(stroke_width)
        p = self.pdf.beginPath()
        p.moveTo(points[0][0], self.ypdf(points[0][1]))
        for x, y in points[1:]:
            p.lineTo(x, self.ypdf(y))
        self.pdf.drawPath(p, stroke=1, fill=0)
        self.draw.line([self.sp(x, y) for x, y in points], fill=pil_color(stroke, opacity), width=max(1, int(stroke_width * self.scale)))

    def path(self, style: dict[str, str], transform, opacity: float) -> None:
        d = style.get("d", "")
        tokens = PATH_TOKEN_RE.findall(d)
        if not tokens:
            return
        stroke = parse_color(style.get("stroke"), parse_color(style.get("fill"), "#000000"))
        fill = parse_color(style.get("fill"))
        sx, sy, _, _ = transform
        stroke_width = parse_length(style.get("stroke-width"), 1.0) * ((sx + sy) / 2)
        p = self.pdf.beginPath()
        points_for_pil: list[tuple[float, float]] = []
        last = (0.0, 0.0)
        i = 0
        while i < len(tokens):
            cmd = tokens[i]
            i += 1
            if cmd == "M":
                x, y = pt(float(tokens[i]), float(tokens[i + 1]), transform)
                i += 2
                p.moveTo(x, self.ypdf(y))
                last = (x, y)
                points_for_pil.append((x, y))
            elif cmd == "L":
                x, y = pt(float(tokens[i]), float(tokens[i + 1]), transform)
                i += 2
                p.lineTo(x, self.ypdf(y))
                last = (x, y)
                points_for_pil.append((x, y))
            elif cmd == "C":
                x1, y1 = pt(float(tokens[i]), float(tokens[i + 1]), transform)
                x2, y2 = pt(float(tokens[i + 2]), float(tokens[i + 3]), transform)
                x3, y3 = pt(float(tokens[i + 4]), float(tokens[i + 5]), transform)
                i += 6
                p.curveTo(x1, self.ypdf(y1), x2, self.ypdf(y2), x3, self.ypdf(y3))
                for t in [j / 20 for j in range(1, 21)]:
                    bx = (1 - t) ** 3 * last[0] + 3 * (1 - t) ** 2 * t * x1 + 3 * (1 - t) * t**2 * x2 + t**3 * x3
                    by = (1 - t) ** 3 * last[1] + 3 * (1 - t) ** 2 * t * y1 + 3 * (1 - t) * t**2 * y2 + t**3 * y3
                    points_for_pil.append((bx, by))
                last = (x3, y3)
            elif cmd == "Z":
                p.close()
                if points_for_pil:
                    points_for_pil.append(points_for_pil[0])

        if stroke:
            self.pdf.setStrokeColor(pdf_color(stroke, opacity))
        if fill:
            self.pdf.setFillColor(pdf_color(fill, opacity))
        self.pdf.setLineWidth(stroke_width)
        self.pdf.drawPath(p, stroke=1 if stroke and stroke_width > 0 else 0, fill=1 if fill else 0)

        if points_for_pil:
            if fill:
                self.draw.polygon([self.sp(x, y) for x, y in points_for_pil], fill=pil_color(fill, opacity))
            if stroke:
                self.draw.line([self.sp(x, y) for x, y in points_for_pil], fill=pil_color(stroke, opacity), width=max(1, int(stroke_width * self.scale)))


def convert(svg_path: Path, pdf_dir: Path, tiff_dir: Path, dpi: int) -> dict[str, object]:
    pdf_path = pdf_dir / f"{svg_path.stem}.pdf"
    tiff_path = tiff_dir / f"{svg_path.stem}.tiff"
    renderer = Renderer(svg_path, pdf_path, tiff_path, dpi)
    renderer.render()
    return {
        "source_svg": str(svg_path),
        "width_px": renderer.width,
        "height_px": renderer.height,
        "pdf": str(pdf_path),
        "pdf_bytes": pdf_path.stat().st_size,
        "pdf_sha256": sha256(pdf_path),
        "tiff": str(tiff_path),
        "tiff_bytes": tiff_path.stat().st_size,
        "tiff_sha256": sha256(tiff_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SVG submission figures to PDF and TIFF.")
    parser.add_argument("--figure-root", default="outputs/patterns_submission/lean_submission_bundle/figures")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    figure_root = Path(args.figure_root)
    svg_paths = sorted((figure_root / "main_figures").glob("*.svg"))
    svg_paths += sorted((figure_root / "graphical_abstract").glob("*.svg"))
    if not svg_paths:
        raise FileNotFoundError(f"No SVG files found below {figure_root}")

    pdf_dir = figure_root / "submission_pdf"
    tiff_dir = figure_root / "submission_tiff"
    if args.clean:
        shutil.rmtree(pdf_dir, ignore_errors=True)
        shutil.rmtree(tiff_dir, ignore_errors=True)
        shutil.rmtree(figure_root / "test_lo", ignore_errors=True)
        test_pdf = figure_root / "test_figure1.pdf"
        if test_pdf.exists():
            test_pdf.unlink()
    pdf_dir.mkdir(parents=True, exist_ok=True)
    tiff_dir.mkdir(parents=True, exist_ok=True)

    rows = [convert(svg, pdf_dir, tiff_dir, args.dpi) for svg in svg_paths]

    manifest_path = figure_root / "submission_format_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    notes_path = figure_root / "submission_format_notes.md"
    notes_path.write_text(
        "# Submission Figure Format Notes\n\n"
        "The original SVG files remain the editable/source figure files. For Patterns/Cell Press-style "
        "submission packaging, this folder also provides PDF derivatives for vector-preserving upload "
        f"and {args.dpi}-DPI LZW-compressed TIFF fallbacks for portals that request raster figure files.\n\n"
        "- PDF derivatives: `submission_pdf/`\n"
        "- TIFF derivatives: `submission_tiff/`\n"
        "- Conversion manifest: `submission_format_manifest.csv`\n\n"
        "The derivatives were generated from the lean bundle SVG files with the local "
        "`tools/convert_submission_figures.py` renderer. PDF is preferred for line/vector "
        "figures when the submission portal accepts it; TIFFs are included as high-resolution fallbacks.\n",
        encoding="utf-8",
    )

    print(f"Converted {len(rows)} SVG files.")
    print(f"PDF output: {pdf_dir}")
    print(f"TIFF output: {tiff_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
