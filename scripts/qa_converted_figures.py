from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageSequence


def font(size: int, bold: bool = False):
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def render_pdf(pdftoppm: str, pdf: Path, out_dir: Path, dpi: int) -> Path:
    prefix = out_dir / pdf.stem
    png = out_dir / f"{pdf.stem}.png"
    if png.exists():
        png.unlink()
    subprocess.run(
        [pdftoppm, "-png", "-singlefile", "-r", str(dpi), str(pdf), str(prefix)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if not png.exists():
        raise FileNotFoundError(f"Rendered PNG not found for {pdf}")
    return png


def first_tiff_frame(path: Path) -> Image.Image:
    image = Image.open(path)
    frame = next(ImageSequence.Iterator(image)).convert("RGB")
    image.close()
    return frame


def thumb(image: Image.Image, width: int, height: int) -> Image.Image:
    copy = image.copy()
    copy.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "white")
    x = (width - copy.width) // 2
    y = (height - copy.height) // 2
    canvas.paste(copy, (x, y))
    return canvas


def make_pair_contact_sheet(rows: list[dict[str, object]], out_path: Path) -> None:
    title_font = font(28, True)
    label_font = font(18, True)
    small_font = font(14)
    cell_w, cell_h = 560, 430
    label_h = 72
    gap = 28
    margin = 40
    width = margin * 2 + cell_w * 2 + gap
    height = margin * 2 + 56 + len(rows) * (cell_h + label_h + gap)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 24), "Converted Figure Visual QA: PDF render vs TIFF fallback", fill="#222", font=title_font)
    y = margin + 56

    for row in rows:
        name = str(row["name"])
        pdf_img = Image.open(row["pdf_png"]).convert("RGB")
        tiff_img = first_tiff_frame(Path(str(row["tiff"])))
        pdf_thumb = thumb(pdf_img, cell_w, cell_h)
        tiff_thumb = thumb(tiff_img, cell_w, cell_h)
        pdf_img.close()
        sheet.paste(pdf_thumb, (margin, y + label_h))
        sheet.paste(tiff_thumb, (margin + cell_w + gap, y + label_h))
        draw.rectangle([margin, y + label_h, margin + cell_w, y + label_h + cell_h], outline="#bbb", width=2)
        draw.rectangle([margin + cell_w + gap, y + label_h, margin + cell_w * 2 + gap, y + label_h + cell_h], outline="#bbb", width=2)
        draw.text((margin, y), name, fill="#222", font=label_font)
        draw.text((margin, y + 28), f"PDF render: {row['pdf_size']}", fill="#555", font=small_font)
        draw.text((margin + cell_w + gap, y + 28), f"TIFF: {row['tiff_size']} @ {row['tiff_dpi']}", fill="#555", font=small_font)
        y += cell_h + label_h + gap

    sheet.save(out_path)


def make_format_contact_sheet(rows: list[dict[str, object]], key: str, out_path: Path, title: str) -> None:
    title_font = font(28, True)
    label_font = font(16, True)
    cell_w, cell_h = 430, 310
    gap = 24
    margin = 40
    cols = 2
    width = margin * 2 + cols * cell_w + (cols - 1) * gap
    body_rows = (len(rows) + cols - 1) // cols
    height = margin * 2 + 55 + body_rows * (cell_h + 50 + gap)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 24), title, fill="#222", font=title_font)
    y0 = margin + 55

    for i, row in enumerate(rows):
        col = i % cols
        rr = i // cols
        x = margin + col * (cell_w + gap)
        y = y0 + rr * (cell_h + 50 + gap)
        if key == "pdf_png":
            image = Image.open(row[key]).convert("RGB")
        else:
            image = first_tiff_frame(Path(str(row[key])))
        tile = thumb(image, cell_w, cell_h)
        image.close()
        sheet.paste(tile, (x, y + 34))
        draw.rectangle([x, y + 34, x + cell_w, y + 34 + cell_h], outline="#bbb", width=2)
        draw.text((x, y), str(row["name"]), fill="#222", font=label_font)

    sheet.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create visual QA contact sheets for converted figure formats.")
    parser.add_argument("--figure-root", default="outputs/patterns_submission/lean_submission_bundle/figures")
    parser.add_argument("--out-dir", default="outputs/patterns_submission/lean_submission_bundle/figures/format_visual_qa")
    parser.add_argument("--pdftoppm", default="pdftoppm")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    figure_root = Path(args.figure_root)
    out_dir = Path(args.out_dir)
    rendered_dir = out_dir / "pdf_rendered_png"
    rendered_dir.mkdir(parents=True, exist_ok=True)

    pdf_dir = figure_root / "submission_pdf"
    tiff_dir = figure_root / "submission_tiff"
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    rows: list[dict[str, object]] = []

    for pdf in pdfs:
        tiff = tiff_dir / f"{pdf.stem}.tiff"
        if not tiff.exists():
            raise FileNotFoundError(f"Missing TIFF fallback for {pdf.stem}")
        pdf_png = render_pdf(args.pdftoppm, pdf, rendered_dir, args.dpi)
        with Image.open(pdf_png) as pdf_img:
            pdf_size = pdf_img.size
        with first_tiff_frame(tiff) as tiff_img:
            tiff_size = tiff_img.size
            tiff_dpi = tiff_img.info.get("dpi")
        rows.append(
            {
                "name": pdf.stem,
                "pdf": str(pdf),
                "pdf_png": str(pdf_png),
                "pdf_size": f"{pdf_size[0]}x{pdf_size[1]}",
                "tiff": str(tiff),
                "tiff_size": f"{tiff_size[0]}x{tiff_size[1]}",
                "tiff_dpi": tiff_dpi,
            }
        )

    metrics_path = out_dir / "figure_format_qa_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    make_pair_contact_sheet(rows, out_dir / "pdf_vs_tiff_contact_sheet.png")
    make_format_contact_sheet(rows, "pdf_png", out_dir / "pdf_render_contact_sheet.png", "PDF Render Contact Sheet")
    make_format_contact_sheet(rows, "tiff", out_dir / "tiff_contact_sheet.png", "TIFF Fallback Contact Sheet")

    print(f"QA rows: {len(rows)}")
    print(f"Metrics: {metrics_path}")
    print(f"Pair contact sheet: {out_dir / 'pdf_vs_tiff_contact_sheet.png'}")
    print(f"PDF contact sheet: {out_dir / 'pdf_render_contact_sheet.png'}")
    print(f"TIFF contact sheet: {out_dir / 'tiff_contact_sheet.png'}")


if __name__ == "__main__":
    main()
