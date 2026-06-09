from __future__ import annotations

import re
import unicodedata

# TOKEN_PATTERN = re.compile(r"[a-ząćęłńóśźż]+(?:[./][a-ząćęłńóśźż]+)*")
TOKEN_PATTERN = re.compile(r"[a-ząćęłńóśźż0-9]+(?:[-./][a-ząćęłńóśźż0-9]+)*")

def tokenize(text: str) -> list[str]:
	normalized = unicodedata.normalize("NFKC", text.lower())
	return TOKEN_PATTERN.findall(normalized)


def add_ngrams(tokens: list[str], sizes: tuple[int, ...] = (1, 2)) -> list[str]:
	combined_tokens = list(tokens)

	for size in sizes:
		if size < 2:
			continue

		for index in range(len(tokens) - size + 1):
			ngram = "_".join(tokens[index : index + size])
			combined_tokens.append(ngram)

	return combined_tokens