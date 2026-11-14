"""Постановления Пленума Верховного Суда РФ (ППВС).

Источник — открытый датасет ``Alexey5676/russian-supreme-court-plenum-acts``
(CC0): 1 503 постановления Пленума ВС РФ 1961–2026 гг., собранные с
официального раздела vsrf.ru. Тексты получены из PDF, поэтому перед чанкингом
чистятся: убираются колонтитулы, номера страниц, переносы слов, подписи.

В корпус попадают только постановления по уголовным делам (упоминание
Уголовного кодекса / УК РФ во вводной части), принятые после 1 января 1997 г.
(после вступления в силу действующего УК РФ), кроме актов «О внесении
изменений» — они не содержат самостоятельных разъяснений.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import Settings

MONTHS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6,
    "июля": 7, "августа": 8, "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}
DATE_RE = re.compile(r"от\s+(\d{1,2})\s+([а-яё]+)\s+(\d{4})\s+года", re.IGNORECASE)
PARA_RE = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2})?)\.\s+(.*)$")
PARA_PAREN_RE = re.compile(r"^\s*(\d{1,2})\)\s+(.*)$")
PAGE_RE = re.compile(r"^\s*\d{1,3}\s*$")
SERVICE_RE = re.compile(
    r"^\s*(ПОСТАНОВЛЕНИЕ ПЛЕНУМА|ПЛЕНУМА? ВЕРХОВНОГО СУДА|ПЛЕНУМ ВЕРХОВНОГО СУДА|"
    r"Постановление Пленума Верховного|ПОСТАНОВЛЕНИЕ|Приложение|Приложения)\b",
    re.IGNORECASE,
)
SIG_RE = re.compile(r"^\s*(Председатель Верховного Суда|Секретарь Пленума|Судья Верховного Суда)", re.IGNORECASE)
AMEND_RE = re.compile(r"^\s*\((?:с изменениями|в ред\.|с изм\.)", re.IGNORECASE)
TOPIC_RE = re.compile(r"^\s*(О|Об)\s+", re.IGNORECASE)


@dataclass
class Paragraph:
    label: str
    text: str
    kind: str = "paragraph"  # paragraph | preamble | whole


@dataclass
class PlenumDoc:
    doc_id: str
    number: str
    date: str
    title: str
    annotation: str
    url: str
    paragraphs: list[Paragraph] = field(default_factory=list)
    added_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def parse_russian_date(title: str) -> str | None:
    match = DATE_RE.search(title or "")
    if not match:
        return None
    day, month, year = match.groups()
    month_number = MONTHS.get(month.lower())
    if not month_number:
        return None
    return f"{year}-{month_number:02d}-{int(day):02d}"


def clean_pdf_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ").replace("\u200b", "")
    text = re.sub(r"([а-яёА-ЯЁ])-\n\s*([а-яё])", r"\1\2", text)
    lines: list[str] = []
    for raw in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", raw).strip()
        if not line:
            continue
        if PAGE_RE.match(line):
            continue
        if SERVICE_RE.match(line) and len(line) < 120:
            continue
        if AMEND_RE.match(line):
            continue
        if SIG_RE.match(line):
            break
        lines.append(line)
    return "\n".join(lines)


def split_paragraphs(text: str) -> tuple[str, list[Paragraph]]:
    """Возвращает (заголовок-тема, пункты)."""
    lines = text.split("\n")
    title = ""
    start = 0
    for index, line in enumerate(lines[:8]):
        if TOPIC_RE.match(line) and len(line) > 8:
            title = line
            start = index + 1
            # Заголовок мог переноситься на следующие строки.
            for _ in range(3):
                if start >= len(lines):
                    break
                nxt = lines[start]
                if (
                    PARA_RE.match(nxt)
                    or PARA_PAREN_RE.match(nxt)
                    or re.match(r"^\s*(от|№|\(|Пленум|В целях|В связи|Рассмотрев|Заслушав)", nxt, re.IGNORECASE)
                    or "г. №" in nxt
                    or len(nxt) < 4
                ):
                    break
                title = f"{title} {nxt}"
                start += 1
            break
    if not title:
        for index, line in enumerate(lines[:4]):
            if len(line) > 15 and not line.lower().startswith("постановление"):
                title = line
                start = index + 1
                break

    paragraphs: list[Paragraph] = []
    current: Paragraph | None = None
    preamble: list[str] = []

    for line in lines[start:]:
        match = PARA_RE.match(line) or PARA_PAREN_RE.match(line)
        if match:
            current = Paragraph(label=match.group(1), text=line, kind="paragraph")
            paragraphs.append(current)
        elif current is None:
            preamble.append(line)
        else:
            current.text = f"{current.text} {line}"

    preamble_text = " ".join(preamble).strip()
    if not paragraphs:
        whole = " ".join(line for line in lines[start:]).strip()
        if whole:
            paragraphs.append(Paragraph(label="", text=whole, kind="whole"))
    elif len(preamble_text) > 120:
        paragraphs.insert(0, Paragraph(label="", text=preamble_text, kind="preamble"))
    return title, paragraphs


def is_criminal(title: str, text: str) -> bool:
    intro = text[:4000].lower()
    return (
        "уголовн" in (title or "").lower()
        or "уголовного кодекса" in intro
        or "ук рф" in intro
    )


def is_amendment(title: str) -> bool:
    lowered = (title or "").lower().lstrip()
    return lowered.startswith(
        ("о внесении", "о дополнении", "об изменении", "о признании утратившим", "о порядке введения")
    )


def normalize_title(title: str) -> str:
    title = re.sub(r"\s+", " ", title).strip(" ,;.-")
    title = re.split(r"\s+(?:от\s+\d{1,2}\s+[а-яё]+\s+\d{4}|№\s*\d)", title, maxsplit=1)[0]
    return title.strip(" ,;.-")


DATASET_URL = (
    "https://huggingface.co/datasets/Alexey5676/russian-supreme-court-plenum-acts"
    "/resolve/main/plenum_resolutions.jsonl"
)


def download_dataset(path: Path, verbose: bool = True) -> Path:
    import requests

    path.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"[ppvs] скачиваю датасет ({DATASET_URL})...")
    with requests.get(DATASET_URL, stream=True, timeout=300) as response:
        response.raise_for_status()
        with path.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
    if verbose:
        print(f"[ppvs] сохранено: {path} ({path.stat().st_size / 1e6:.1f} МБ)")
    return path


def build_ppvs(
    settings: Settings,
    dataset_path: Path | None = None,
    refresh: bool = False,
    verbose: bool = True,
) -> dict:
    dataset_path = dataset_path or (settings.data_dir / "raw" / "plenum_resolutions.jsonl")
    output_path = settings.data_dir / "raw" / "ppvs.jsonl"
    if not dataset_path.exists():
        download_dataset(dataset_path, verbose=verbose)

    rows = [json.loads(line) for line in dataset_path.open(encoding="utf-8")]
    docs: list[PlenumDoc] = []
    for row in rows:
        if row.get("doc_type") != "resolution":
            continue
        date = row.get("date_accepted") or parse_russian_date(row.get("title") or "")
        if not date or date < "1997-01-01":
            continue
        raw_text = (row.get("text") or "").strip()
        if len(raw_text) < 800:
            continue
        cleaned = clean_pdf_text(raw_text)
        topic, paragraphs = split_paragraphs(cleaned)
        candidate_title = normalize_title(topic or row.get("title") or "")
        if is_amendment(candidate_title):
            continue
        if candidate_title.lower().startswith(("признано утратившим силу", "утратило силу")):
            continue
        if not is_criminal(candidate_title, cleaned):
            continue
        docs.append(
            PlenumDoc(
                doc_id=str(row.get("id")),
                number=str(row.get("number")),
                date=date,
                title=candidate_title,
                annotation=(row.get("annotation") or "").strip(),
                url=row.get("source_url") or "",
                paragraphs=paragraphs,
                added_at=time.strftime("%Y-%m-%d"),
            )
        )

    docs.sort(key=lambda d: (d.date, int(re.sub(r"\D", "", d.number) or 0)))
    with output_path.open("w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")

    stats = {
        "documents": len(docs),
        "paragraphs": sum(len(d.paragraphs) for d in docs),
        "with_date": sum(1 for d in docs if d.date),
        "path": str(output_path),
    }
    if verbose:
        print(f"[ppvs] постановлений: {len(docs)}, пунктов: {stats['paragraphs']}")
    return stats


def load_ppvs(path: Path | str) -> list[PlenumDoc]:
    path = Path(path)
    docs: list[PlenumDoc] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            data = json.loads(line)
            data["paragraphs"] = [Paragraph(**p) for p in data.get("paragraphs", [])]
            docs.append(PlenumDoc(**data))
    return docs
