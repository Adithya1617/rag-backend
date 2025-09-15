import re
from typing import List, Dict
def naive_token_count(text: str) -> int:
    return max(1, len(text.split()))
def chunk_text(text: str, max_tokens: int = 500, overlap: int = 100) -> List[Dict]:
    paragraphs = re.split(r"\n\s*\n|(?<=\.)\s+(?=[A-Z])", text)
    chunks = []
    current = ""
    start_char = 0
    char_cursor = 0
    chunk_id = 0
    for p in paragraphs:
        p = p.strip()
        if not p:
            char_cursor += 1
            continue
        candidate = (current + "\n\n" + p).strip() if current else p
        if naive_token_count(candidate) <= max_tokens:
            if not current:
                start_char = char_cursor
            current = candidate
            char_cursor += len(p) + 2
        else:
            if current:
                chunks.append({"chunk_id": chunk_id, "text": current, "start_char": start_char, "end_char": start_char + len(current)})
                chunk_id += 1
            sentences = re.split(r'(?<=[.!?]) +', p)
            cur2 = ""
            sstart = char_cursor
            for s in sentences:
                cand2 = (cur2 + " " + s).strip() if cur2 else s
                if naive_token_count(cand2) <= max_tokens:
                    cur2 = cand2
                else:
                    chunks.append({"chunk_id": chunk_id, "text": cur2, "start_char": sstart, "end_char": sstart + len(cur2)})
                    chunk_id += 1
                    cur2 = s
                    sstart += len(s)
            if cur2:
                chunks.append({"chunk_id": chunk_id, "text": cur2, "start_char": sstart, "end_char": sstart + len(cur2)})
                chunk_id += 1
            current = ""
            char_cursor += len(p) + 2
    if current:
        chunks.append({"chunk_id": chunk_id, "text": current, "start_char": start_char, "end_char": start_char + len(current)})
    final = []
    for i, c in enumerate(chunks):
        text = c['text']
        if i > 0:
            prev_tokens = final[-1]['text'].split()
            overlap_tokens = prev_tokens[-overlap:] if len(prev_tokens) > overlap else prev_tokens
            text = ' '.join(overlap_tokens) + ' ' + text
        final.append({**c, 'text': text})
    return final
