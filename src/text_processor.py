import re
from typing import Dict, List, Optional
import torch
from transformers import BertTokenizer


class TextProcessor:
    def __init__(
        self,
        tokenizer_path: str = "src/models/saved_tokenizer",
        model_name_fallback: str = "bert-base-uncased",
        max_len: int = 64,
    ):
        self.max_len = max_len
        self.tokenizer = self._load_tokenizer(tokenizer_path, model_name_fallback)

    def _load_tokenizer(self, tokenizer_path: str, model_name_fallback: str) -> BertTokenizer:
        try:
            return BertTokenizer.from_pretrained(tokenizer_path)
        except Exception:
            return BertTokenizer.from_pretrained(model_name_fallback)

    @staticmethod
    def clean_text(text: str) -> str:
        text = (text or "").strip()
        text = re.sub(r"https?://\\S+|www\\.\\S+", " ", text)
        text = re.sub(r"\\s+", " ", text)
        return text

    def encode(self, text: str, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        text = self.clean_text(text)
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        if device is not None:
            encoded = {k: v.to(device) for k, v in encoded.items()}
        return encoded

    def get_input_ids(self, text: str, device: Optional[torch.device] = None) -> torch.Tensor:
        return self.encode(text, device=device)["input_ids"]
