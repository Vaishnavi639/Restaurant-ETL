import re

def normalize_extracted_text(raw: str) -> str:

    if not raw:
        return ""

    text = raw
    text = re.sub(r'-{2,}\s*Page\s*\d+\s*-{2,}', '\n', text, flags=re.IGNORECASE)
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if len(s) <= 1:
            continue
        if re.match(r'^[\W_]+$', s):
            continue
        s = re.sub(r'\.{2,}', ' ', s)
        lines.append(s)

    text = "\n".join(lines)
    text = re.sub(r'\n{2,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)

    # trim
    text = text.strip()

    return text

