from __future__ import annotations

import io
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _source_line(source: dict[str, Any]) -> str:
    label = source.get("label") or "S?"
    title = source.get("title") or source.get("file_name") or source.get("url") or source.get("source_id") or "Источник"
    coordinates = []
    if source.get("page"):
        coordinates.append(f"стр. {source['page']}")
    if source.get("slide"):
        coordinates.append(f"слайд {source['slide']}")
    suffix = f" ({', '.join(coordinates)})" if coordinates else ""
    url = source.get("url")
    return f"- [{label}] [{title}]({url}){suffix}" if url else f"- [{label}] {title}{suffix}"


def build_markdown_report(run: dict[str, Any]) -> str:
    request = run.get("request_json") or {}
    result = run.get("result_json") or {}
    if run.get("status") != "complete" or not result:
        raise ValueError("Report is available only for a completed Agentic RAG job")
    generated = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z")
    lines = [
        "# Отчёт Nornikel Agentic RAG",
        "",
        f"**Сформирован:** {generated}",
        f"**Режим ответа:** {result.get('mode') or 'unknown'}",
        f"**Уверенность:** {float(result.get('confidence') or 0):.2f}",
        "",
        "## Вопрос",
        "",
        str(request.get("query") or ""),
        "",
        "## Ответ",
        "",
        str(result.get("answer_markdown") or ""),
        "",
        "## Источники",
        "",
    ]
    sources = result.get("sources") or []
    lines.extend(_source_line(item) for item in sources)
    if not sources:
        lines.append("- Проверяемые источники не найдены.")
    gaps = result.get("gaps") or []
    if gaps:
        lines.extend(["", "## Пробелы", "", *[f"- {item}" for item in gaps]])
    contradictions = result.get("contradictions") or []
    if contradictions:
        lines.extend([
            "", "## Противоречия", "",
            *[f"- {item.get('summary') or item.get('reason') or str(item)}" for item in contradictions],
        ])
    warnings = result.get("warnings") or []
    if warnings:
        lines.extend(["", "## Ограничения", "", *[f"- {item}" for item in warnings]])
    return "\n".join(lines).strip() + "\n"


def _plain_markdown(value: str) -> str:
    value = re.sub(r"!\[([^]]*)\]\([^)]+\)", r"\1", value)
    value = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", value)
    value = re.sub(r"[`*_>#]", "", value)
    return value.strip()


def build_pdf_report(run: dict[str, Any]) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

    request = run.get("request_json") or {}
    result = run.get("result_json") or {}
    if run.get("status") != "complete" or not result:
        raise ValueError("Report is available only for a completed Agentic RAG job")

    regular_candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    )
    bold_candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
    )
    regular = next((path for path in regular_candidates if path.is_file()), None)
    bold = next((path for path in bold_candidates if path.is_file()), regular)
    if regular is None:
        raise RuntimeError("A Unicode TrueType font is required for PDF export")
    pdfmetrics.registerFont(TTFont("ReportUnicode", str(regular)))
    pdfmetrics.registerFont(TTFont("ReportUnicodeBold", str(bold)))

    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "ReportBody", parent=styles["BodyText"], fontName="ReportUnicode",
        fontSize=9.5, leading=14, textColor=colors.HexColor("#172033"), alignment=TA_LEFT,
        spaceAfter=4,
    )
    heading = ParagraphStyle(
        "ReportHeading", parent=body, fontName="ReportUnicodeBold", fontSize=15,
        leading=19, spaceBefore=10, spaceAfter=7,
    )
    title = ParagraphStyle(
        "ReportTitle", parent=heading, fontSize=20, leading=24,
        textColor=colors.HexColor("#111827"), spaceAfter=14,
    )
    meta = ParagraphStyle(
        "ReportMeta", parent=body, fontSize=8.5, textColor=colors.HexColor("#667085"),
    )
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=17 * mm, bottomMargin=17 * mm,
        title="Nornikel Agentic RAG report",
    )
    story: list[Any] = [
        Paragraph("Отчёт Nornikel Agentic RAG", title),
        Paragraph(
            f"Режим: {result.get('mode') or 'unknown'} · Уверенность: {float(result.get('confidence') or 0):.2f} · "
            f"Сформирован: {datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M')}",
            meta,
        ),
        Spacer(1, 5 * mm),
        Paragraph("Вопрос", heading),
        Paragraph(_plain_markdown(str(request.get("query") or "")) or "—", body),
        Paragraph("Ответ", heading),
    ]
    answer_lines = str(result.get("answer_markdown") or "").splitlines()
    for raw in answer_lines:
        line = raw.strip()
        if not line:
            story.append(Spacer(1, 2 * mm))
            continue
        if line.startswith("#"):
            story.append(Paragraph(_plain_markdown(line), heading))
        elif line.startswith(("- ", "* ")):
            story.append(Paragraph(f"• {_plain_markdown(line[2:])}", body))
        else:
            story.append(Paragraph(_plain_markdown(line).replace("|", " · "), body))
    story.append(Paragraph("Источники", heading))
    for source in result.get("sources") or []:
        story.append(Paragraph(_plain_markdown(_source_line(source).removeprefix("- ")), body))
    if result.get("gaps"):
        story.append(Paragraph("Пробелы", heading))
        for item in result["gaps"]:
            story.append(Paragraph(f"• {_plain_markdown(str(item))}", body))
    if result.get("contradictions"):
        story.append(Paragraph("Противоречия", heading))
        for item in result["contradictions"]:
            story.append(Paragraph(f"• {_plain_markdown(str(item))}", body))
    document.build(story)
    return buffer.getvalue()
