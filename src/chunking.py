from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []

        for start in range(0, len(text), step):
            chunk = text[start:start + self.chunk_size]
            chunks.append(chunk)

            if start + self.chunk_size >= len(text):
                break

        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        # Chia văn bản thành các câu.
        # Hỗ trợ ".", "!", "?" theo sau bởi khoảng trắng/newline.
        sentences = re.split(
            r"(?<=[.!?])\s+",
            text.strip()
        )

        # Loại bỏ phần tử rỗng
        sentences = [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

        chunks: list[str] = []

        # Gom tối đa max_sentences_per_chunk câu vào một chunk
        for i in range(
            0,
            len(sentences),
            self.max_sentences_per_chunk
        ):
            group = sentences[
                i:i + self.max_sentences_per_chunk
            ]

            chunk = " ".join(group).strip()

            if chunk:
                chunks.append(chunk)

        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = [
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]

    def __init__(
        self,
        separators: list[str] | None = None,
        chunk_size: int = 500
    ) -> None:
        self.separators = (
            self.DEFAULT_SEPARATORS
            if separators is None
            else list(separators)
        )

        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        chunks = self._split(
            text.strip(),
            self.separators
        )

        return [
            chunk.strip()
            for chunk in chunks
            if chunk.strip()
        ]

    def _split(
        self,
        current_text: str,
        remaining_separators: list[str]
    ) -> list[str]:

        current_text = current_text.strip()

        if not current_text:
            return []

        # Nếu văn bản đã đủ nhỏ thì giữ nguyên.
        if len(current_text) <= self.chunk_size:
            return [current_text]

        # Không còn separator nào để thử.
        # Fallback: cắt cứng theo chunk_size.
        if not remaining_separators:
            return [
                current_text[i:i + self.chunk_size]
                for i in range(
                    0,
                    len(current_text),
                    self.chunk_size
                )
            ]

        separator = remaining_separators[0]
        next_separators = remaining_separators[1:]

        # Separator cuối cùng là "".
        # Khi đó cắt trực tiếp theo ký tự.
        if separator == "":
            return [
                current_text[i:i + self.chunk_size]
                for i in range(
                    0,
                    len(current_text),
                    self.chunk_size
                )
            ]

        # Thử chia bằng separator hiện tại.
        parts = current_text.split(separator)

        # Nếu separator không tồn tại,
        # thử separator tiếp theo.
        if len(parts) == 1:
            return self._split(
                current_text,
                next_separators
            )

        chunks: list[str] = []
        current_chunk = ""

        for part in parts:
            part = part.strip()

            if not part:
                continue

            # Ghép phần hiện tại vào chunk.
            if current_chunk:
                candidate = (
                    current_chunk
                    + separator
                    + part
                )
            else:
                candidate = part

            # Nếu vẫn nằm trong giới hạn chunk_size
            if len(candidate) <= self.chunk_size:
                current_chunk = candidate

            else:
                # Chunk hiện tại đã đầy.
                if current_chunk:
                    chunks.extend(
                        self._split(
                            current_chunk,
                            next_separators
                        )
                    )

                current_chunk = part

        # Xử lý phần còn lại.
        if current_chunk:
            chunks.extend(
                self._split(
                    current_chunk,
                    next_separators
                )
            )

        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(
        x * y
        for x, y in zip(a, b)
    )


def compute_similarity(
    vec_a: list[float],
    vec_b: list[float]
) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """

    if not vec_a or not vec_b:
        return 0.0

    dot_product = _dot(
        vec_a,
        vec_b
    )

    magnitude_a = math.sqrt(
        sum(
            x * x
            for x in vec_a
        )
    )

    magnitude_b = math.sqrt(
        sum(
            x * x
            for x in vec_b
        )
    )

    # Tránh chia cho 0
    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0

    return dot_product / (
        magnitude_a * magnitude_b
    )


class ChunkingStrategyComparator:
    """
    Run all built-in chunking strategies
    and compare their results.
    """

    def compare(
        self,
        text: str,
        chunk_size: int = 200
    ) -> dict:

        # Strategy 1: Fixed-size
        fixed_chunks = FixedSizeChunker(
            chunk_size=chunk_size,
            overlap=0
        ).chunk(text)

        # Strategy 2: Sentence-based
        sentence_chunks = SentenceChunker(
            max_sentences_per_chunk=3
        ).chunk(text)

        # Strategy 3: Recursive
        recursive_chunks = RecursiveChunker(
            chunk_size=chunk_size
        ).chunk(text)

        strategies = {
            "fixed_size": fixed_chunks,
            "by_sentences": sentence_chunks,
            "recursive": recursive_chunks,
        }

        result: dict = {}

        for name, chunks in strategies.items():

            lengths = [
                len(chunk)
                for chunk in chunks
            ]

            result[name] = {
                "chunks": chunks,
                "count": len(chunks),
                "avg_length": (
                    sum(lengths) / len(lengths)
                    if lengths
                    else 0.0
                ),
            }

        return result

class HeadingChunker:
    """
    Chia tài liệu Markdown theo heading (#, ##, ###, ...).

    Mỗi chunk bắt đầu từ một heading và chứa toàn bộ nội dung
    cho đến trước heading tiếp theo.
    """

    HEADING_PATTERN = re.compile(r"(?m)^(#{1,6})\s+(.+?)\s*$")

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        text = text.strip()
        matches = list(self.HEADING_PATTERN.finditer(text))

        if not matches:
            return [text]

        chunks: list[str] = []

        for index, match in enumerate(matches):
            start = match.start()
            end = (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else len(text)
            )

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

        return chunks