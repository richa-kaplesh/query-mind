import re
from core.models import ExtractedPage

class TextChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._heading_pattern = re.compile(r"^\[\[HEADING:(.*?)\]\](.*)$")

    def chunk_pages(self, pages: list[ExtractedPage]) -> list[dict]:
        all_chunks = []
        for page in pages:
            chunks = self.chunk_text(page.text, page.metadata.model_dump())
            all_chunks.extend(chunks)
        return all_chunks

    def chunk_text(self, text: str, metadata: dict | None = None) -> list[dict]:
        if metadata is None:
            metadata = {}

        lines = text.split("\n")
        sections: list[tuple[str | None, list[str]]] = []
        current_heading = None
        current_lines: list[str] = []

        for raw_line in lines:
            if not raw_line.strip():
                continue
            match = self._heading_pattern.match(raw_line)
            if match:
                heading, content = match.group(1), match.group(2).strip()
            else:
                heading, content = current_heading, raw_line.strip()

            if heading != current_heading and current_lines:
                sections.append((current_heading, current_lines))
                current_lines = []

            current_heading = heading
            if content:
                current_lines.append(content)

        if current_lines:
            sections.append((current_heading, current_lines))

        chunks = []
        for heading, section_lines in sections:
            chunks.extend(self._chunk_section(heading, section_lines, metadata))
        return chunks

    def _chunk_section(self, heading, lines: list[str], metadata: dict) -> list[dict]:
            chunks = []          # final list of chunk-dicts we'll return
            buffer = []          # lines waiting to become the next chunk
            buffer_len = 0        # total characters currently in buffer

            for line in lines:
                line_len = len(line) + 1   # +1 for the "\n" that will join it

                # if adding this line would overflow the current chunk, seal it off first
                if buffer and (buffer_len + line_len > self.chunk_size):

                    # --- seal the current buffer into a chunk ---
                    chunk_text = "\n".join(buffer)
                    chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            **metadata,
                            "chunk_index": len(chunks),
                            "heading": heading,
                        }
                    })

                    # --- figure out overlap: keep the last few lines for the next chunk ---
                    new_buffer = []
                    new_buffer_len = 0
                    for line_from_end in reversed(buffer):
                        extra_len = len(line_from_end) + 1
                        if new_buffer_len + extra_len > self.chunk_overlap:
                            break
                        new_buffer.append(line_from_end)   # append instead of insert(0, ...)
                        new_buffer_len += extra_len

                    new_buffer.reverse()   # we added them backwards, so flip to correct order

                    buffer = new_buffer
                    buffer_len = new_buffer_len

                # add the current line to (whatever's left of) the buffer
                buffer.append(line)
                buffer_len += line_len

            # after the loop, seal whatever's left in the buffer as the final chunk
            if buffer:
                chunk_text = "\n".join(buffer)
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_index": len(chunks),
                        "heading": heading,
                    }
                })

            return chunks