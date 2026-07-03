"""Offline WordLevel tokenizer factory shared across omnitok tests."""

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast


def make_word_level_tokenizer(
    vocab_tokens=("<unk>",),
    *,
    normalizer=None,
    whitespace=False,
    bos_eos=False,
    added_tokens=None,
    chat_template=None,
):
    """Build a tiny offline WordLevel tokenizer.

    vocab_tokens become the base model vocab (ids 0..n-1, <unk> required);
    added_tokens layer on via add_tokens (str or AddedToken).
    """
    tokens = list(vocab_tokens)
    if bos_eos:
        tokens += [t for t in ("<s>", "</s>") if t not in tokens]
    backend = Tokenizer(
        WordLevel({t: i for i, t in enumerate(tokens)}, unk_token="<unk>")
    )
    if normalizer is not None:
        backend.normalizer = normalizer
    if whitespace:
        backend.pre_tokenizer = Whitespace()
    roles = {"unk_token": "<unk>"}
    if bos_eos:
        roles.update(bos_token="<s>", eos_token="</s>")
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, **roles)
    if added_tokens:
        tok.add_tokens(list(added_tokens))
    if chat_template is not None:
        tok.chat_template = chat_template
    return tok
