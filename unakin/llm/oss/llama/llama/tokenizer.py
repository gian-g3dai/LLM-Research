# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
from logging import getLogger
from typing import List, Optional

from sentencepiece import SentencePieceProcessor


logger = getLogger()


class Tokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        # logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()

        self.prefix_id: Optional[int] = self.sp_model.PieceToId("▁<PRE>") or None
        self.middle_id: Optional[int] = self.sp_model.PieceToId("▁<MID>") or None
        self.suffix_id: Optional[int] = self.sp_model.PieceToId("▁<SUF>") or None
        self.eot_id: Optional[int] = self.sp_model.PieceToId("▁<EOT>") or None

        if self.n_words > 32016:
            self.rev_id: Optional[int] = self.sp_model.PieceToId("[REV]")
        else:
            self.rev_id: Optional[int] = None

        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id} - PAD ID: {self.pad_id}"
        )
        logger.info(
            f"Prefix ID: {self.prefix_id} - Middle ID: {self.middle_id} - Suffix ID: {self.suffix_id} - EOT ID: {self.eot_id}"
        )
        logger.info(f"Reverse ID: {self.rev_id}")

        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)

    def vocab_size(self) -> int:
        return self.sp_model.vocab_size()