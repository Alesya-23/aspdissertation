"""Загрузка и разбор текста УК РФ.

Источник — постатейные страницы ukodeksrf.ru: на каждой странице в блоке
``div.field--name-body`` лежит только текст нормы (до заголовка «Комментарий»).
Комментарии, судебная практика и история поправок отбрасываются.

Результат — ``data/raw/uk_rf.jsonl``: одна строка на статью, внутри —
иерархия (часть кодекса -> раздел -> глава) и элементы статьи
(части, примечания, вводный текст).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup

from .config import Settings

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

# «1. Кража...», «2.1. ...»
PART_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2})?)\.\s+(.*)$")
# «Примечание.» / «Примечания. 1. ...» — только с заглавной буквы и на границе слова
NOTE_INLINE_RE = re.compile(r"(?:^|\s)Примечани[ея]\.?(?=\s|$)")
# «а) ...», «б.1) ...»
ITEM_FIND_RE = re.compile(r"(?:^|\s)([а-яё](?:\.\d{1,2})?)\)")
STOP_RE = re.compile(
    r"^(Комментарий|Другой комментарий|Постатейный комментарий|Судебная практика|"
    r"Комментарии к статье)",
    re.IGNORECASE,
)
ARTICLE_LABEL_RE = re.compile(r"^Статья\s+(\d+(?:\.\d+)?)\.\s*(.*)$")


@dataclass
class Element:
    """Атомарный фрагмент статьи: часть, примечание или вводный текст."""

    label: str
    text: str
    kind: str  # part | note | preamble | whole
    items: list[str] = field(default_factory=list)


@dataclass
class Article:
    number: str
    title: str
    url: str
    part: str
    section: str
    chapter: str
    elements: list[Element] = field(default_factory=list)
    repealed: bool = False
    repealed_note: str = ""
    fetched_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Article":
        elements = [Element(**e) for e in data.get("elements", [])]
        return cls(
            number=data["number"],
            title=data["title"],
            url=data["url"],
            part=data.get("part", ""),
            section=data.get("section", ""),
            chapter=data.get("chapter", ""),
            elements=elements,
            repealed=data.get("repealed", False),
            repealed_note=data.get("repealed_note", ""),
            fetched_at=data.get("fetched_at", ""),
        )


def normalize_space(text: str) -> str:
    text = text.replace("\xa0", " ").replace("\u200b", "")
    return re.sub(r"\s+", " ", text).strip()


def clean_node_text(node) -> str:
    for sup in node.find_all("sup"):
        sup.decompose()
    text = node.get_text(" ")
    text = normalize_space(text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([«(])\s+", r"\1", text)
    text = re.sub(r"\s+([»)])", r"\1", text)
    return text


def fetch_html(url: str, session: requests.Session, retries: int = 3) -> str:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, timeout=40)
            if response.status_code == 404:
                raise FileNotFoundError(url)
            response.raise_for_status()
            response.encoding = "utf-8"
            return response.text
        except FileNotFoundError:
            raise
        except Exception as exc:  # сетевые сбои и 5xx
            last_error = exc
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"не удалось скачать {url}: {last_error}")


def parse_toc(html: str) -> list[ArticleRef]:
    """Разбирает оглавление в плоский список статей с иерархией."""
    soup = BeautifulSoup(html, "html.parser")
    container = soup.select_one("div.field--name-body") or soup
    refs: list[ArticleRef] = []

    def walk(ul, depth: int, ctx: dict) -> None:
        for li in ul.find_all("li", recursive=False):
            anchor = li.find("a", recursive=False)
            span = li.find("span", recursive=False)
            node = anchor or span
            if node is None:
                continue
            label = normalize_space(node.get_text(" "))
            href = (anchor.get("href") or "") if anchor else ""
            ctx = dict(ctx)
            if depth == 1:
                ctx["part"] = _part_name(label, href)
            elif depth == 2:
                ctx["section"] = label
            elif depth == 3:
                ctx["chapter"] = label
            elif depth == 4 and href:
                match = ARTICLE_LABEL_RE.match(label)
                if not match:
                    continue
                number, title = match.group(1), match.group(2).strip()
                refs.append(
                    ArticleRef(
                        number=number,
                        title=title.rstrip(" ."),
                        url=href,
                        part=ctx.get("part", ""),
                        section=ctx.get("section", ""),
                        chapter=ctx.get("chapter", ""),
                    )
                )
            nested = li.find("ul", recursive=False)
            if nested is not None:
                walk(nested, depth + 1, ctx)

    for top in container.find_all("ul", recursive=False):
        walk(top, 1, {})
    return refs


def _part_name(label: str, href: str) -> str:
    if "особенная" in label.lower() or "/ch-2" in href:
        return "Особенная часть"
    if "общая" in label.lower() or "/ch-1" in href:
        return "Общая часть"
    return normalize_space(re.sub(r"^Часть\s+\d+\.?\s*", "", label))


@dataclass
class ArticleRef:
    number: str
    title: str
    url: str
    part: str
    section: str
    chapter: str


def parse_article(html: str, ref: ArticleRef, base_url: str) -> Article:
    soup = BeautifulSoup(html, "html.parser")
    body = soup.select_one("div.field--name-body")
    if body is None:
        raise ValueError(f"на странице {ref.url} нет текста статьи")

    lines: list[str] = []
    for node in body.find_all(["p", "h2", "h3"]):
        text = clean_node_text(node)
        if not text:
            continue
        if node.name in ("h2", "h3") and STOP_RE.match(text):
            break
        lines.append(text)

    # Убираем служебную строку «Новая редакция Ст. N УК РФ».
    if lines and lines[0].lower().startswith("новая редакция"):
        lines = lines[1:]

    url = ref.url if ref.url.startswith("http") else base_url.rstrip("/") + "/" + ref.url.lstrip("/")
    article = Article(
        number=ref.number,
        title=ref.title,
        url=url,
        part=ref.part,
        section=ref.section,
        chapter=ref.chapter,
        fetched_at=time.strftime("%Y-%m-%d"),
    )

    if not lines:
        return article

    first = lines[0].lower()
    if "утратила силу" in first or "утратил силу" in first:
        article.repealed = True
        article.repealed_note = lines[0]
        return article

    article.elements = split_elements(lines)
    if not article.elements:
        article.repealed = True
        article.repealed_note = lines[0]
    return article


def split_elements(lines: Iterable[str]) -> list[Element]:
    """Делит текст статьи на части и примечания."""
    elements: list[Element] = []
    current: Element | None = None
    in_notes = False

    def start(label: str, text: str, kind: str) -> None:
        nonlocal current
        current = Element(label=label, text=text, kind=kind)
        elements.append(current)

    def start_note(text: str) -> None:
        match = re.match(r"^(\d{1,2})\.\s*(.*)$", text)
        if match:
            start(match.group(1), text, "note")
        else:
            start(str(len([e for e in elements if e.kind == "note"]) + 1), text, "note")

    for line in lines:
        if not in_notes:
            # «Примечания» может стоять в середине абзаца после «Утратило силу...».
            inline = NOTE_INLINE_RE.search(line)
            if inline:
                before = line[: inline.start()].strip()
                if before:
                    if current is None:
                        start("", before, "preamble")
                    else:
                        current.text = f"{current.text} {before}".strip()
                in_notes = True
                rest = line[inline.end() :].strip()
                if rest:
                    start_note(rest)
                continue
            part_match = PART_RE.match(line)
            if part_match:
                start(part_match.group(1), line, "part")
                continue
        else:
            note_match = re.match(r"^(\d{1,2})\.\s+", line)
            if note_match:
                start(note_match.group(1), line, "note")
                continue

        if current is None:
            start("", line, "preamble")
        else:
            current.text = f"{current.text} {line}".strip()

    for element in elements:
        if element.kind == "part":
            element.items = ITEM_FIND_RE.findall(element.text)
    return elements


def build_corpus(
    settings: Settings,
    refresh: bool = False,
    limit: int | None = None,
    verbose: bool = True,
) -> dict:
    """Скачивает оглавление и все статьи, сохраняет JSONL. Возвращает статистику."""
    settings.html_dir.mkdir(parents=True, exist_ok=True)
    settings.corpus_path.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    toc_path = settings.html_dir / "_toc.html"
    if refresh or not toc_path.exists():
        toc_html = fetch_html(settings.source_url + "/oglavleniye", session)
        toc_path.write_text(toc_html, encoding="utf-8")
        time.sleep(settings.request_delay)
    else:
        toc_html = toc_path.read_text(encoding="utf-8")

    refs = parse_toc(toc_html)
    # Оглавление может ссылаться на одну статью дважды (например, из разных глав).
    seen: set[str] = set()
    unique: list[ArticleRef] = []
    for ref in refs:
        key = ref.url.rstrip("/")
        if key not in seen:
            seen.add(key)
            unique.append(ref)
    refs = unique
    # Один и тот же номер не должен встречаться дважды (сайт иногда дублирует статью).
    deduped: list[ArticleRef] = []
    seen_numbers: set[str] = set()
    for ref in refs:
        if ref.number not in seen_numbers:
            seen_numbers.add(ref.number)
            deduped.append(ref)
    refs = deduped
    if limit:
        refs = refs[:limit]

    if verbose:
        print(f"[corpus] статей в оглавлении: {len(refs)}")

    articles: list[Article] = []
    failed: list[str] = []
    for idx, ref in enumerate(refs, 1):
        slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", ref.url.strip("/")) or f"st-{ref.number}"
        cache = settings.html_dir / f"{slug}.html"
        try:
            if refresh or not cache.exists():
                html = fetch_html(settings.source_url + ref.url, session)
                cache.write_text(html, encoding="utf-8")
                time.sleep(settings.request_delay)
            else:
                html = cache.read_text(encoding="utf-8")
            articles.append(parse_article(html, ref, settings.source_url))
        except Exception as exc:
            failed.append(f"{ref.number}: {exc}")
        if verbose and idx % 25 == 0:
            print(f"[corpus] {idx}/{len(refs)}")

    with settings.corpus_path.open("w", encoding="utf-8") as fh:
        for article in articles:
            fh.write(json.dumps(article.to_dict(), ensure_ascii=False) + "\n")

    repealed = sum(1 for a in articles if a.repealed)
    stats = {
        "articles": len(articles),
        "repealed": repealed,
        "elements": sum(len(a.elements) for a in articles),
        "failed": failed,
        "path": str(settings.corpus_path),
    }
    if verbose:
        print(
            f"[corpus] сохранено {len(articles)} статей "
            f"(утратили силу: {repealed}), элементов: {stats['elements']}"
        )
        if failed:
            print(f"[corpus] ошибок: {len(failed)}; первые: {failed[:3]}")
    return stats


def load_corpus(path: Path | str) -> list[Article]:
    path = Path(path)
    articles: list[Article] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                articles.append(Article.from_dict(json.loads(line)))
    return articles


def corpus_text(article: Article) -> str:
    """Полный текст статьи без комментариев — для small-to-big раскрытия."""
    return "\n".join(element.text for element in article.elements)
