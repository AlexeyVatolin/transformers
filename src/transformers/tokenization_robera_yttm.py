from typing import List, Optional, Tuple

from .tokenization_utils_base import TextInput
from .tokenization_utils import PreTrainedTokenizer
import youtokentome as yttm


class RobertaTokenizerYttm(PreTrainedTokenizer):
    def __init__(self, model_path,
               bos_token="<BOS>",
               eos_token="<EOS>",
               unk_token="<UNK>",
               pad_token="<PAD>",
               ):
        self.bpe = yttm.BPE(model=model_path)
        self.bos_token = bos_token
        self.eos_token = eos_token
        self._pad_token = pad_token
        self.mask_token = '<mask>'

    def __len__(self):
        return self.bpe.vocab_size() + 1  # +1 из-за отсутствия <mask> токена в словаре

    def build_inputs_with_special_tokens(self, token_ids_0: List, token_ids_1: Optional[List] = None):
        output = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        if token_ids_1 is None:
            return output

        return output + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    def tokenize(self, text: TextInput, **kwargs):
        return self.bpe.encode([text], output_type=yttm.OutputType.SUBWORD)[0]

    def convert_tokens_to_ids(self, tokens):
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self.bpe.vocab_size() if tokens == self.mask_token else self.bpe.subword_to_id(tokens)

        return [self.bpe.vocab_size() if x == self.mask_token else self.bpe.subword_to_id(x) for x in tokens]

    def get_special_tokens_mask(self, token_ids_0: List, token_ids_1: Optional[List] = None,
                                already_has_special_tokens: bool = False) -> List[int]:
        return [1 if x in {self.bos_token_id, self.pad_token_id, self.eos_token_id} else 0 for x in token_ids_0]

    def num_special_tokens_to_add(self, pair=False):
        token_ids_0 = []
        token_ids_1 = []
        return len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1 if pair else None))

    def save_pretrained(self, save_directory) -> Tuple[str]:
        pass
        # return super().save_pretrained(save_directory)

    @property
    def max_len(self):
        return 512

    @property
    def bos_token_id(self):
        return self.bpe.subword_to_id(self.bos_token)

    @property
    def pad_token_id(self):
        return self.bpe.subword_to_id(self.pad_token)

    @property
    def eos_token_id(self):
        return self.bpe.subword_to_id(self.eos_token)

    @property
    def mask_token_id(self):
        return self.bpe.subword_to_id(self.eos_token)


