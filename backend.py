"""CSV/PDF input, reviewed question records, and CSV-to-PDF conversion.

This file contains the website's data handling. It does not train models;
model_pipeline.py connects the four separate model files.
"""
from __future__ import annotations

import csv
import hashlib
import io
import re
import threading
from pathlib import Path
from xml.sax.saxutils import escape

import fitz
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import LongTable, Paragraph, SimpleDocTemplate, Spacer, TableStyle

MAX_BYTES = 10 * 1024 * 1024
MAX_ROWS = 3000
COLUMNS = ["year", "session", "exam_order", "question_text", "marks", "topic", "question_group_id"]
ALIASES = {
    "question_text": ["question_text", "question", "questions", "text"],
    "year": ["year", "exam_year"],
    "session": ["session", "exam_session", "exam_type"],
    "exam_order": ["exam_order", "order", "paper_order"],
    "marks": ["marks", "mark", "score"],
    "topic": ["topic", "chapter", "unit"],
    "question_group_id": ["question_group_id", "group_id", "question_group"],
}
_FONT_LOCK = threading.Lock()


def clean(value) -> str:
    """Make empty spreadsheet cells safe without turning NaN into question text."""
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def check_file(data: bytes):
    if not data or len(data) > MAX_BYTES:
        raise ValueError("Choose a non-empty file smaller than 10 MB.")


def read_csv(data: bytes) -> tuple[pd.DataFrame, list[str]]:
    """Accept common Excel CSV encodings and separators; reject malformed rows."""
    check_file(data)
    warnings = []
    try:
        text = data.decode("utf-16") if data.startswith((b"\xff\xfe", b"\xfe\xff")) else data.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = data.decode("cp1252")
            warnings.append("Read as a Windows CSV. Please check special characters.")
        except UnicodeDecodeError as exc:
            raise ValueError("Save the CSV using UTF-8 encoding and upload it again.") from exc
    try:
        try:
            separator = csv.Sniffer().sniff(text[:10000], delimiters=",;\t|").delimiter
        except csv.Error:
            separator = ","
        reader = csv.reader(io.StringIO(text, newline=""), delimiter=separator, strict=True)
        headers = [cell.strip() for cell in next(reader, [])]
        if not headers or any(not h for h in headers) or len(set(headers)) != len(headers):
            raise ValueError("The CSV needs unique, non-empty column headings.")
        if len(headers) > 20:
            raise ValueError("Use a CSV with no more than 20 columns.")
        records = []
        for line, cells in enumerate(reader, 2):
            if not cells or not any(cell.strip() for cell in cells):
                continue
            if len(cells) != len(headers):
                raise ValueError(f"CSV line {line} has the wrong number of cells. Put questions containing commas inside double quotes.")
            if any(len(cell) > 12000 for cell in cells):
                raise ValueError("Each CSV cell must have at most 12,000 characters.")
            records.append(cells)
            if len(records) > MAX_ROWS:
                raise ValueError(f"Upload no more than {MAX_ROWS} question rows at a time.")
        if not records:
            raise ValueError("The CSV has headings but no data rows.")
        return pd.DataFrame(records, columns=headers), warnings
    except csv.Error as exc:
        raise ValueError("The CSV has invalid quoting. Check it in Excel and save it again.") from exc


def guess_mapping(frame: pd.DataFrame) -> dict:
    headings = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in frame.columns}
    result = {}
    for field, names in ALIASES.items():
        result[field] = next((headings[re.sub(r"[^a-z0-9]", "", name)] for name in names
                              if re.sub(r"[^a-z0-9]", "", name) in headings), None)
    return result


def map_csv(frame, mapping, year=2025, session="Annual", exam_order=None):
    """Map any chosen CSV headings into the same records used for PDF questions."""
    if mapping.get("question_text") not in frame.columns:
        raise ValueError("Choose the CSV column containing questions.")
    output = pd.DataFrame(index=frame.index)
    defaults = {"year": year, "session": session, "marks": None, "topic": "Unassigned", "question_group_id": ""}
    for column in COLUMNS:
        if mapping.get(column) in frame.columns:
            # Restore our own spreadsheet-safe CSV exports without changing text.
            output[column] = frame[mapping[column]].map(
                lambda value: value[1:] if isinstance(value, str) and value.startswith("'")
                and value[1:].lstrip().startswith(("=", "+", "-", "@")) else value)
        else:
            output[column] = defaults.get(column, "")
    if mapping.get("exam_order") not in frame.columns:
        if mapping.get("year") in frame.columns:
            try:
                years = pd.to_numeric(output["year"], errors="raise")
                output["exam_order"] = exam_order if years.nunique() == 1 and exam_order is not None else years * 100 + 1
            except (TypeError, ValueError) as exc:
                raise ValueError("Exam years must be numbers such as 2024.") from exc
        else:
            output["exam_order"] = exam_order if exam_order is not None else year * 100 + 1
    return validate_questions(output)


def question_identity(text: str) -> str:
    # Exact wording is grouped automatically. Similar wording needs human review.
    # Preserve numbers and mathematical operators: x < 5 and x > 5 differ.
    normalized = clean(text).casefold().rstrip(" .?!;:")
    return "Q-" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:10]


def _integer(value, label, minimum, maximum):
    try:
        number = float(clean(value))
        if number != int(number) or not minimum <= number <= maximum:
            raise ValueError
        return int(number)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a whole number from {minimum} to {maximum}.") from exc


def validate_questions(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate atomically before replacing the session's reviewed records."""
    if frame.empty or len(frame) > MAX_ROWS:
        raise ValueError(f"Keep between 1 and {MAX_ROWS} questions in the project.")
    if "question_text" not in frame.columns:
        raise ValueError("The question_text column is missing.")
    records = []
    for index, raw in enumerate(frame.to_dict("records"), 1):
        text = clean(raw.get("question_text"))
        if not 5 <= len(text) <= 12000:
            raise ValueError(f"Question {index} needs between 5 and 12,000 characters.")
        year = _integer(raw.get("year"), f"Year on row {index}", 1950, 2100)
        session = clean(raw.get("session")) or "Annual"
        order = _integer(raw.get("exam_order", year * 100 + 1), f"Exam order on row {index}", 1, 9999999)
        mark = clean(raw.get("marks"))
        try:
            marks = float(mark) if mark else None
            if marks is not None and not 0 <= marks <= 1000:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Marks on row {index} must be a number from 0 to 1000 or empty.") from exc
        topic = clean(raw.get("topic")) or "Unassigned"
        group = clean(raw.get("question_group_id")) or question_identity(text)
        if len(session) > 80 or len(topic) > 100 or len(group) > 100:
            raise ValueError("Keep sessions, topic names and group IDs shorter than 100 characters.")
        records.append(dict(year=year, session=session, exam_order=order, question_text=text,
                            marks=marks, topic=topic, question_group_id=group))
    output = pd.DataFrame(records, columns=COLUMNS).drop_duplicates().reset_index(drop=True)
    papers = output[["year", "session", "exam_order"]].drop_duplicates()
    if len(papers) > 40:
        raise ValueError("Use no more than 40 comparable exam papers.")
    if papers.duplicated(["year", "session"]).any():
        raise ValueError("Use the same exam_order for every question in one year/session.")
    if papers.duplicated("exam_order").any():
        raise ValueError("Different papers need different exam_order numbers. For two sessions in one year, use values such as 202401 and 202402.")
    years = papers.sort_values("exam_order")["year"].tolist()
    if years != sorted(years):
        raise ValueError("Exam order must run from the oldest year to the newest year.")
    for group, subset in output.groupby("question_group_id", sort=False):
        topics = set(subset["topic"]) - {"Unassigned"}
        if len(topics) > 1:
            raise ValueError(f"Question group {group} has different topics. Give it one consistent topic name.")
        if topics:
            output.loc[output["question_group_id"] == group, "topic"] = next(iter(topics))
    return output


_QUESTION_START = re.compile(r"^\s*(?:Q(?:uestion)?\.?\s*\d{1,3}\s*[.:)\-]?\s*|"
                             r"\d{1,2}\s*\([a-z]\)\s*|\d{1,2}[.)]\s+|\([a-z]\)\s+|[a-z][.)]\s+)", re.I)
_MARKS = re.compile(r"\s*(?:\[(\d{1,3}(?:\.\d+)?)\s*(?:marks?)?\]|"
                    r"\((\d{1,3}(?:\.\d+)?)\s*marks?\)|marks?\s*[:=]\s*(\d{1,3}(?:\.\d+)?))\s*$", re.I)
_IGNORE = re.compile(r"^(?:page\s*\d+(?:\s*of\s*\d+)?|time\s*:.*|(?:total|maximum)\s*marks?\s*:.*|"
                     r"instructions?\s*:.*|answer\s+(?:any|all)\b.*|section\s+[a-z0-9]+\s*:?.*|examdesk\s*\|.*)$", re.I)


def read_pdf(data, year, session="Annual", exam_order=None):
    """Extract selectable-text PDFs. Never pretend a scanned image has no questions."""
    check_file(data)
    try:
        with fitz.open(stream=data, filetype="pdf") as document:
            if document.needs_pass:
                raise ValueError("This PDF is password protected. Upload an unlocked copy.")
            if len(document) > 100:
                raise ValueError("Use a PDF with no more than 100 pages.")
            pages = [page.get_text("text", sort=True) for page in document]
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("The PDF could not be opened. Check that it is a valid PDF.") from exc
    if sum(len(page.strip()) for page in pages) < 30:
        raise ValueError("This looks like a scanned/image PDF. Convert it to a searchable PDF with OCR first, or upload its questions as CSV.")
    active, blocks = [], []
    text = "\n".join(pages)
    for line in text.splitlines():
        line = line.strip()
        if not line or _IGNORE.match(line):
            continue
        start = _QUESTION_START.match(line)
        if start:
            if active:
                blocks.append(" ".join(active))
            active = [line[start.end():].strip()]
        elif active:
            active.append(line)
    if active:
        blocks.append(" ".join(active))
    warnings = ["Check the extracted question boundaries, marks and any diagrams against the PDF."]
    if not blocks:
        blocks = [clean(part) for part in re.split(r"\n\s*\n", text) if len(clean(part)) >= 12]
        warnings.append("No question numbers were found. Paragraphs were used; you can split or add rows in the review table.")
    if any(len(page.strip()) < 10 for page in pages):
        warnings.append("Some pages have very little text. Check for scanned pages or diagram-only questions.")
    records = []
    for block in blocks:
        match = _MARKS.search(block)
        marks = float(next(value for value in match.groups() if value)) if match else None
        body = clean(block[:match.start()] if match else block)
        if len(body) >= 5:
            records.append(dict(year=year, session=session, exam_order=exam_order or year * 100 + 1,
                                question_text=body, marks=marks, topic="Unassigned", question_group_id=""))
    if not records:
        raise ValueError("No question text was found. Upload a CSV transcription instead.")
    return validate_questions(pd.DataFrame(records)), warnings


def csv_bytes(frame):
    """Export CSV safely for opening in spreadsheet applications."""
    output = frame.copy()
    for column in output.columns:
        output[column] = output[column].map(
            lambda value: "'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value)
    return output.to_csv(index=False).encode("utf-8-sig")


def make_pdf(frame, title="Question paper", mode="questions", question_column="question_text", marks_column="marks", columns=None):
    """Convert a question CSV or a general CSV table into a downloadable PDF."""
    if frame.empty or len(frame) > MAX_ROWS:
        raise ValueError(f"Convert between 1 and {MAX_ROWS} rows.")
    if mode not in {"questions", "table"}:
        raise ValueError("Choose question layout or table layout.")
    columns = list(columns if columns is not None else frame.columns)
    if mode == "questions" and question_column not in frame.columns:
        raise ValueError("Select the column containing question text.")
    if mode == "table" and (not columns or len(columns) > 12 or any(c not in frame.columns for c in columns)):
        raise ValueError("Select between 1 and 12 columns for table layout.")
    with _FONT_LOCK:
        if "Examdesk" not in pdfmetrics.getRegisteredFontNames():
            root = Path(__file__).resolve().parent
            pdfmetrics.registerFont(TTFont("Examdesk", str(root / "DejaVuSans.ttf")))
            pdfmetrics.registerFont(TTFont("ExamdeskBold", str(root / "DejaVuSans-Bold.ttf")))
    def safe(value):
        return escape(clean(value))
    ink = colors.HexColor("#253344")
    body = ParagraphStyle("Body", fontName="Examdesk", fontSize=10, leading=16, textColor=ink, spaceAfter=12)
    small = ParagraphStyle("Small", parent=body, fontSize=8, leading=11, textColor=colors.HexColor("#69717A"))
    heading = ParagraphStyle("Heading", parent=body, fontName="ExamdeskBold", fontSize=21, leading=27)
    page = landscape(A4) if mode == "table" and len(columns) > 5 else A4
    buffer = io.BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=page, leftMargin=38, rightMargin=38, topMargin=38, bottomMargin=40,
                                 title=clean(title)[:120], author="Examdesk")
    story = [Paragraph(safe(title) or "Question paper", heading),
             Paragraph(f"{len(frame)} {'questions' if mode == 'questions' else 'rows'} | Converted from CSV", small), Spacer(1, 14)]
    if mode == "questions":
        for index, row in enumerate(frame.to_dict("records"), 1):
            text = clean(row.get(question_column))
            if not text:
                raise ValueError(f"Question {index} is empty. Correct the CSV first.")
            marks = clean(row.get(marks_column))
            suffix = f" [{safe(marks)} marks]" if marks else ""
            story.append(Paragraph(f'<font name="ExamdeskBold">Q{index}.</font> {safe(text)}{suffix}', body))
    else:
        cell = ParagraphStyle("Cell", parent=body, fontSize=8, leading=11, spaceAfter=0, wordWrap="CJK")
        header = ParagraphStyle("Header", parent=cell, fontName="ExamdeskBold")
        rows = [[Paragraph(safe(column), header) for column in columns]]
        rows.extend([[Paragraph(safe(row.get(column)), cell) for column in columns] for row in frame.to_dict("records")])
        table = LongTable(rows, colWidths=[document.width / len(columns)] * len(columns), repeatRows=1, splitInRow=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF2F1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAFBFA")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#E2E5E8")),
            ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
        story.append(table)
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Examdesk", 8)
        canvas.setFillColor(colors.HexColor("#69717A"))
        canvas.drawString(38, 17, f"Examdesk | Page {doc.page}")
        canvas.restoreState()
    try:
        document.build(story, onFirstPage=footer, onLaterPages=footer)
    except Exception as exc:
        raise ValueError("The PDF could not be laid out. Try fewer table columns or shorter cells.") from exc
    return buffer.getvalue()
