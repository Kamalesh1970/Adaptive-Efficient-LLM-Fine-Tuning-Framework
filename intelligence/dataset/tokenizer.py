"""
Tokenizer Abstraction Layer.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field

from intelligence.dataset.schema import NormalizedSample, SchemaType
from core.exceptions import TokenizationError
from core.logger import get_logger

logger = get_logger("tokenizer")


class TokenizedSample(BaseModel):
    """Output container for a tokenized text sample."""
    input_ids: List[int] = Field(description="Sequence of token IDs")
    attention_mask: List[int] = Field(description="Attention mask (1 for token, 0 for padding)")
    token_count: int = Field(description="Number of non-padding tokens")


class MockTokenizer:
    """Lightweight, deterministic word/char mock tokenizer for offline testing."""

    def __init__(
        self,
        max_length: int = 512,
        pad_token_id: int = 0,
        unk_token_id: int = 1,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
    ):
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.vocab: Dict[str, int] = {}
        self.next_id = 4

    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        truncation: bool = True,
        padding: bool = False,
        add_special_tokens: bool = True,
    ) -> Dict[str, List[int]]:
        effective_max = max_length or self.max_length
        words = text.strip().split()

        token_ids = []
        if add_special_tokens:
            token_ids.append(self.bos_token_id)

        for w in words:
            clean_w = w.lower()
            if clean_w not in self.vocab:
                self.vocab[clean_w] = self.next_id
                self.next_id += 1
            token_ids.append(self.vocab[clean_w])

        if add_special_tokens:
            token_ids.append(self.eos_token_id)

        # Truncation
        if truncation and len(token_ids) > effective_max:
            token_ids = token_ids[:effective_max]
            if add_special_tokens:
                token_ids[-1] = self.eos_token_id

        token_count = len(token_ids)
        attention_mask = [1] * token_count

        # Padding
        if padding and len(token_ids) < effective_max:
            pad_len = effective_max - len(token_ids)
            token_ids.extend([self.pad_token_id] * pad_len)
            attention_mask.extend([0] * pad_len)

        return {
            "input_ids": token_ids,
            "attention_mask": attention_mask,
            "token_count": token_count,
        }


class TokenizerWrapper:
    """Wrapper supporting both Hugging Face PreTrainedTokenizer and MockTokenizer."""

    def __init__(
        self,
        tokenizer_obj: Any,
        max_length: int = 512,
        truncation: bool = True,
        padding: bool = False,
    ):
        self.tokenizer = tokenizer_obj
        self.max_length = max_length
        self.truncation = truncation
        self.padding = padding

    def encode_sample(self, sample: NormalizedSample) -> TokenizedSample:
        """
        Tokenizes a NormalizedSample into TokenizedSample.

        Args:
            sample: NormalizedSample instance.

        Returns:
            TokenizedSample containing input_ids, attention_mask, and token_count.
        """
        text = sample.get_text_content()
        if not text:
            raise TokenizationError("Cannot tokenize empty text sample")

        try:
            if hasattr(self.tokenizer, "encode") and not hasattr(self.tokenizer, "__call__"):
                res = self.tokenizer.encode(
                    text,
                    max_length=self.max_length,
                    truncation=self.truncation,
                    padding=self.padding,
                )
                return TokenizedSample(
                    input_ids=res["input_ids"],
                    attention_mask=res["attention_mask"],
                    token_count=res["token_count"],
                )
            elif callable(self.tokenizer):
                # Standard Hugging Face tokenizer call interface
                res = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    truncation=self.truncation,
                    padding="max_length" if self.padding else False,
                    return_tensors=None,
                )
                input_ids = res["input_ids"]
                attention_mask = res.get("attention_mask", [1] * len(input_ids))
                token_count = sum(1 for m in attention_mask if m == 1)
                return TokenizedSample(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_count=token_count,
                )
            else:
                raise TokenizationError(f"Provided tokenizer of type {type(self.tokenizer).__name__} is not callable")
        except Exception as e:
            if isinstance(e, TokenizationError):
                raise
            raise TokenizationError(f"Failed to tokenize sample text: {e}") from e

    def encode_dataset(self, samples: List[NormalizedSample]) -> List[TokenizedSample]:
        """Tokenizes a list of NormalizedSample objects."""
        return [self.encode_sample(s) for s in samples]
