import unittest

from rag.chunking import FixedChunker, StructureChunker, build_chunks, chunk_stats
from rag.corpus import Article, Element


def words(text: str) -> int:
    return len(text.split())


def make_article(number: str = "158", title: str = "Кража") -> Article:
    return Article(
        number=number,
        title=title,
        url=f"https://example.test/st-{number}",
        part="Особенная часть",
        section="Раздел VIII. Преступления в сфере экономики",
        chapter="Глава 21. Преступления против собственности",
        elements=[
            Element("1", "1. Кража, то есть тайное хищение чужого имущества, - наказывается лишением свободы.", "part"),
            Element("2", "2. Кража, совершенная: а) группой лиц; б) с проникновением в жилище.", "part", ["а", "б"]),
            Element("1", "1. Под хищением понимаются противоправные изъятие и обращение чужого имущества.", "note"),
            Element("2", "2. Значительный ущерб гражданину определяется с учетом имущественного положения.", "note"),
        ],
    )


def make_three_part_article() -> Article:
    return Article(
        number="1",
        title="Тест",
        url="https://example.test/st-1",
        part="Общая часть",
        section="Раздел I",
        chapter="Глава 1",
        elements=[
            Element("1", "1. Первая короткая часть нормы.", "part"),
            Element("2", "2. Вторая короткая часть нормы.", "part"),
            Element("3", "3. Третья короткая часть нормы.", "part"),
        ],
    )


class StructureChunkerTest(unittest.TestCase):
    def test_header_contains_path_and_part(self) -> None:
        chunks = StructureChunker(words, max_tokens=200).chunk_article(make_article())
        self.assertEqual(len(chunks), 2)  # части + примечания
        part_chunk = next(c for c in chunks if c.kind == "part")
        self.assertIn("УК РФ", part_chunk.text)
        self.assertIn("Глава 21", part_chunk.text)
        self.assertIn("Статья 158. Кража", part_chunk.text)
        self.assertIn("ч. 1, 2", part_chunk.text)
        self.assertIn("ст. 158", part_chunk.citation())

    def test_notes_are_separate(self) -> None:
        chunks = StructureChunker(words, max_tokens=200).chunk_article(make_article())
        notes = [c for c in chunks if c.kind == "note"]
        self.assertEqual(len(notes), 1)
        self.assertIn("Примечания", notes[0].text)
        self.assertIn("Под хищением", notes[0].text)

    def test_respects_token_budget(self) -> None:
        chunks = StructureChunker(words, max_tokens=40).chunk_article(make_article())
        for chunk in chunks:
            self.assertLessEqual(chunk.tokens, 45, chunk.text)

    def test_overlap_repeats_previous_part(self) -> None:
        chunks = StructureChunker(words, max_tokens=26, overlap_parts=1).chunk_article(
            make_three_part_article()
        )
        parts = [c for c in chunks if c.kind == "part"]
        self.assertGreaterEqual(len(parts), 2)
        self.assertIn("2. Вторая", parts[1].body)

    def test_no_overlap_when_disabled(self) -> None:
        chunks = StructureChunker(words, max_tokens=26, overlap_parts=0).chunk_article(
            make_three_part_article()
        )
        parts = [c for c in chunks if c.kind == "part"]
        self.assertGreaterEqual(len(parts), 2)
        self.assertNotIn("2. Вторая", parts[1].body)

    def test_repealed_article_is_skipped(self) -> None:
        article = make_article()
        article.repealed = True
        self.assertEqual(StructureChunker(words).chunk_article(article), [])

    def test_ids_are_unique(self) -> None:
        chunks = build_chunks(
            [make_article(), make_article("159", "Мошенничество")], StructureChunker(words)
        )
        ids = [c.id for c in chunks]
        self.assertEqual(len(ids), len(set(ids)))

    def test_stats(self) -> None:
        chunks = StructureChunker(words).chunk_article(make_article())
        stats = chunk_stats(chunks)
        self.assertEqual(stats["count"], len(chunks))
        self.assertEqual(stats["articles"], 1)


class FixedChunkerTest(unittest.TestCase):
    def test_window_and_overlap(self) -> None:
        chunker = FixedChunker(
            encode=lambda text: text.split(),
            decode=lambda ids: " ".join(ids),
            count_tokens=words,
            chunk_tokens=12,
            overlap=0.25,
        )
        chunks = chunker.chunk_article(make_article())
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(c.tokens <= 30 for c in chunks))
        self.assertTrue(all(c.strategy == "fixed" for c in chunks))


if __name__ == "__main__":
    unittest.main()
