import unittest

from rag.corpus import parse_article, parse_toc

TOC_HTML = """
<html><body><main>
<div class="field--name-body">
  <ul class="menu">
    <li class="menu-item--expanded"><a href="/ch-1">Часть 1. Общая часть</a>
      <ul class="menu">
        <li class="menu-item--expanded"><a href="/ch-1/rzd-1">Раздел I. Уголовный закон</a>
          <ul class="menu">
            <li class="menu-item--expanded"><a href="/ch-1/rzd-1/gl-1">Глава 1. Задачи и принципы</a>
              <ul class="menu">
                <li><a href="/ch-1/rzd-1/gl-1/st-1-uk-rf">Статья 1. Уголовное законодательство</a></li>
              </ul>
            </li>
          </ul>
        </li>
      </ul>
    </li>
    <li class="menu-item--expanded"><a href="/ch-2">Часть 2. Особенная часть</a>
      <ul class="menu">
        <li class="menu-item--expanded"><a href="/ch-2/rzd-8">Раздел VIII. Экономика</a>
          <ul class="menu">
            <li class="menu-item--expanded"><a href="/ch-2/rzd-8/gl-21">Глава 21. Собственность</a>
              <ul class="menu">
                <li><a href="/ch-2/rzd-8/gl-21/st-158-uk-rf">Статья 158. Кража</a></li>
                <li><a href="/ch-2/rzd-8/gl-21/st-159-3-uk-rf">Статья 159.3. Мошенничество с электронными средствами</a></li>
              </ul>
            </li>
          </ul>
        </li>
      </ul>
    </li>
  </ul>
</div>
</main></body></html>
"""

ARTICLE_HTML = """
<html><body><main>
<div class="field--name-body">
  <p><strong>Новая редакция Ст. 158 УК РФ</strong></p>
  <p>1. Кража, то есть тайное хищение чужого имущества, -</p>
  <p>наказывается лишением свободы на срок до двух лет.</p>
  <p>2. Кража, совершенная:</p>
  <p>а) группой лиц по предварительному сговору;</p>
  <p>б) с незаконным проникновением в жилище, -</p>
  <p>наказывается лишением свободы на срок до шести лет.</p>
  <p>Примечания. 1. Под хищением понимаются противоправные изъятие и обращение.</p>
  <p>2. Значительный ущерб определяется с учетом имущественного положения.</p>
  <h2>Комментарий к Статье 158 УК РФ</h2>
  <p>Кража — это тайное хищение. Этот текст не должен попасть в корпус.</p>
</div>
</main></body></html>
"""


class CorpusParseTest(unittest.TestCase):
    def test_toc_hierarchy(self) -> None:
        refs = parse_toc(TOC_HTML)
        self.assertEqual([r.number for r in refs], ["1", "158", "159.3"])
        self.assertEqual(refs[1].part, "Особенная часть")
        self.assertEqual(refs[1].section, "Раздел VIII. Экономика")
        self.assertEqual(refs[1].chapter, "Глава 21. Собственность")
        self.assertEqual(refs[1].title, "Кража")

    def test_article_parts_and_notes(self) -> None:
        refs = parse_toc(TOC_HTML)
        article = parse_article(ARTICLE_HTML, refs[1], "https://example.test")
        self.assertFalse(article.repealed)
        kinds = [(e.kind, e.label) for e in article.elements]
        self.assertEqual(kinds, [("part", "1"), ("part", "2"), ("note", "1"), ("note", "2")])
        self.assertNotIn("Комментарий", " ".join(e.text for e in article.elements))
        self.assertIn("наказывается", article.elements[0].text)
        self.assertIn("жилище", article.elements[1].text)

    def test_repealed_article(self) -> None:
        refs = parse_toc(TOC_HTML)
        html = ARTICLE_HTML.replace(
            "1. Кража, то есть тайное хищение чужого имущества, -",
            "Статья 158. Утратила силу - Федеральный закон от 01.01.2000 N 1-ФЗ.",
        )
        article = parse_article(html, refs[1], "https://example.test")
        self.assertTrue(article.repealed)
        self.assertEqual(article.elements, [])


if __name__ == "__main__":
    unittest.main()
