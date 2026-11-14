"""Генерация ответа локальной GGUF-моделью через llama.cpp.

Модель по умолчанию — та же, что в ``init.py``: YandexGPT-5-Lite-8B-instruct
(Q4_K_M). Контекст 4096 токенов, поэтому в промпт попадает 4–6 фрагментов,
а не весь кодекс. Ответ обязан ссылаться на [S1], [S2] и номера статей.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

SYSTEM_PROMPT = """Ты — ассистент по уголовному праву Российской Федерации. Отвечай ТОЛЬКО по фрагментам Уголовного кодекса РФ из сообщения пользователя.

Обязательные правила:
1. Начни ответ с нормы: «Согласно ст. 158 ч. 3 УК РФ …».
2. После каждого утверждения ставь ссылку на фрагмент: [S1], [S2].
3. Указывай статью и часть УК РФ, к которым относится вывод.
4. Если во фрагментах нет ответа, напиши ровно: «В предоставленных фрагментах УК РФ ответа нет.»
5. Не ссылайся на статьи, которых нет во фрагментах, и не выдумывай формулировки.
6. Пиши кратко, по-русски, без вступлений.
"""

USER_TEMPLATE = """Фрагменты УК РФ:
{context}

Вопрос: {question}

Ответ (с обязательными ссылками [S1], [S2] и номером статьи):"""


class Generator:
    def __init__(
        self,
        model_path: Path | str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        temperature: float = 0.1,
        max_new_tokens: int = 512,
        verbose: bool = False,
    ) -> None:
        from llama_cpp import Llama

        self.model_path = str(model_path)
        self.n_ctx = n_ctx
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_batch=256,
            use_mlock=False,
            verbose=verbose,
        )

    def build_prompt(self, question: str, context: str) -> str:
        return USER_TEMPLATE.format(context=context, question=question)

    def count_tokens(self, text: str) -> int:
        return len(self.llm.tokenize(text.encode("utf-8"), add_bos=False))

    def fit_context(self, question: str, context: str, max_new_tokens: int | None = None) -> str:
        """Урезает контекст, чтобы промпт гарантированно влез в n_ctx."""
        reserve = (max_new_tokens or self.max_new_tokens) + 64
        budget = self.n_ctx - reserve
        blocks = context.split("\n\n")
        while blocks and self.count_tokens(self.build_prompt(question, "\n\n".join(blocks))) > budget:
            blocks.pop()
        if not blocks and context:
            # Даже один фрагмент не влез — отдаём хотя бы начало первого.
            blocks = [context[: max(200, budget * 2)]]
        return "\n\n".join(blocks)

    def answer(
        self,
        question: str,
        context: str,
        stream: bool = True,
    ) -> Iterator[str] | str:
        context = self.fit_context(question, context)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self.build_prompt(question, context)},
        ]
        kwargs = dict(
            temperature=self.temperature,
            top_p=0.9,
            repeat_penalty=1.1,
            max_tokens=self.max_new_tokens,
        )
        try:
            response = self.llm.create_chat_completion(messages, stream=stream, **kwargs)
        except Exception:
            response = self._complete_fallback(question, context, stream, **kwargs)

        if stream:
            return self._iter_stream(response)
        return response["choices"][0]["message"]["content"].strip()

    def _iter_stream(self, response: Iterable[dict]) -> Iterator[str]:
        for part in response:
            delta = part["choices"][0].get("delta", {})
            piece = delta.get("content")
            if piece:
                yield piece

    def _complete_fallback(self, question: str, context: str, stream: bool, **kwargs):
        prompt = (
            f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n"
            f"{self.build_prompt(question, context)}\n<|assistant|>\n"
        )
        result = self.llm.create_completion(prompt, stream=stream, stop=["\n<|user|>", "\n<|system|>"], **kwargs)
        if stream:
            return (
                {"choices": [{"delta": {"content": part["choices"][0]["text"]}}]}
                for part in result
            )
        return {"choices": [{"message": {"content": result["choices"][0]["text"]}}]}
