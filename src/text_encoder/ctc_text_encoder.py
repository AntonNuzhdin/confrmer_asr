import os
import re
from collections import defaultdict
from string import ascii_lowercase

import torch
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, build_ctcdecoder

# TODO add CTC decode
# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(
        self, alphabet=None, language_model_path=None, alphabet_path=None, **kwargs
    ):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """
        lm_lowercase_path = os.path.join(
            os.path.dirname(language_model_path), "lm_lowercase.arpa"
        )
        if not os.path.exists(lm_lowercase_path):
            with open(language_model_path, "r") as f_upper:
                with open(lm_lowercase_path, "w") as f_lower:
                    for line in f_upper:
                        f_lower.write(line.lower())

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        if alphabet_path:
            with open(alphabet_path) as file:
                alph = [sym.lower() for sym in file.read().strip().split("\n")]

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        print(language_model_path, alphabet_path)
        if language_model_path:
            self.decoder_bs = build_ctcdecoder(
                labels=self.vocab,
                kenlm_model_path=lm_lowercase_path,
                unigrams=alph,
            )
        else:
            self.decoder_bs = BeamSearchDecoderCTC(Alphabet(self.vocab, False), None)

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    # from seminar
    def ctc_decode(self, inds) -> str:
        decoded = []
        last_char = self.EMPTY_TOK
        for ind in inds:
            cur_char = self.ind2char[ind]
            if cur_char == self.EMPTY_TOK:
                continue
            if cur_char != last_char:
                decoded.append(cur_char)
            last_char = cur_char
        return "".join(decoded)

    # from seminar
    def ctc_beam_search_decode_slow(self, probs, beam_size=5):
        probs = torch.exp(probs)
        dp = {
            ("", self.EMPTY_TOK): 1.0,
        }
        for prob in probs:
            dp = self._expand_and_merge_path(dp, prob)
            dp = self._truncate_paths(dp, beam_size)
        dp = [
            (prefix, proba)
            for (prefix, _), proba in sorted(dp.items(), key=lambda x: -x[1])
        ]
        return dp[0][0]

    def _expand_and_merge_path(self, dp, next_token_probs):
        new_dp = defaultdict(float)
        for ind, next_token_prob in enumerate(next_token_probs):
            cur_char = self.ind2char[ind]
            for (prefix, last_char), v in dp.items():
                if last_char == cur_char:
                    new_prefix = prefix
                else:
                    if cur_char != self.EMPTY_TOK:
                        new_prefix = prefix + cur_char
                    else:
                        new_prefix = prefix
                new_dp[(new_prefix, cur_char)] += v * next_token_prob
        return new_dp

    def ctc_beam_search_decode(self, probs, beam_size=5):
        probs = probs.detach().cpu().numpy()
        return self.decoder_bs.decode(probs, beam_size)

    def _truncate_paths(self, dp, beam_size):
        return dict(sorted(list(dp.items()), key=lambda x: -x[1])[:beam_size])

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
