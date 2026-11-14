import unittest

from rag.chunking import PlenumChunker
from rag.ppvs import Paragraph, PlenumDoc, clean_pdf_text, normalize_title, parse_russian_date, split_paragraphs


class PPVSParseTest(unittest.TestCase):
    def test_date(self) -> None:
        self.assertEqual(
            parse_russian_date("Постановление Пленума ВС РФ от 27 января 1999 года № 1"),
            "1999-01-27",
        )
        self.assertIsNone(parse_russian_date("без даты"))

    def test_clean_removes_pages_and_headers(self) -> None:
        raw = "12\nПЛЕНУМ ВЕРХОВНОГО СУДА\nО судебной практике\nпо делам об убийстве\nПредседатель Верховного Суда\nВ.М. Лебедев"
        cleaned = clean_pdf_text(raw)
        self.assertNotIn("12", cleaned.split("\n"))
        self.assertNotIn("ПЛЕНУМ", cleaned)
        self.assertNotIn("Лебедев", cleaned)
        self.assertIn("О судебной практике", cleaned)

    def test_split_paragraphs(self) -> None:
        text = "О судебной практике по делам об убийстве\nВ целях обеспечения правильного применения закона Пленум постановляет:\n1. Первый пункт.\nПродолжение первого пункта.\n2. Второй пункт."
        title, paragraphs = split_paragraphs(text)
        self.assertEqual(title, "О судебной практике по делам об убийстве")
        labels = [(p.kind, p.label) for p in paragraphs]
        self.assertIn(("paragraph", "1"), labels)
        self.assertIn(("paragraph", "2"), labels)
        first = next(p for p in paragraphs if p.label == "1")
        self.assertIn("Продолжение первого пункта", first.text)

    def test_normalize_title(self) -> None:
        self.assertEqual(
            normalize_title("О судебной практике по делам о краже от 6 февраля 2007 г. № 7"),
            "О судебной практике по делам о краже",
        )


class PPVSChunkerTest(unittest.TestCase):
    def test_header_and_citation(self) -> None:
        doc = PlenumDoc(
            doc_id="1",
            number="29",
            date="2002-12-27",
            title="О судебной практике по делам о краже, грабеже и разбое",
            annotation="",
            url="https://example.test",
            paragraphs=[
                Paragraph("1", "1. Первый пункт."),
                Paragraph("2", "2. Второй пункт."),
            ],
        )
        counter = lambda text: len(text.split())
        chunks = PlenumChunker(counter, max_tokens=200).chunk_document(doc)
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertIn("ППВС", chunk.citation())
        self.assertIn("27.12.2002", chunk.citation())
        self.assertIn("Пункты 1, 2", chunk.text)
        self.assertEqual(chunk.act, "ППВС")
        self.assertEqual(chunk.doc_number, "29")

    def test_no_cross_document(self) -> None:
        def make(doc_id: str) -> PlenumDoc:
            return PlenumDoc(
                doc_id=doc_id,
                number=doc_id,
                date="2020-01-01",
                title="Тест",
                annotation="",
                url="",
                paragraphs=[Paragraph(str(i), f"{i}. Пункт {i}.") for i in range(1, 6)],
            )

        chunker = PlenumChunker(lambda t: len(t.split()), max_tokens=20)
        for doc in (make("1"), make("2")):
            for chunk in chunker.chunk_document(doc):
                self.assertEqual(chunk.doc_id, doc.doc_id)


if __name__ == "__main__":
    unittest.main()
