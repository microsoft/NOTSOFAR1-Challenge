"""
NOTSOFAR adopts the same text normalizer as the CHiME-8 DASR track.
This code is aligned with the CHiME-8 repo:
https://github.com/chimechallenge/chime-utils/tree/main/chime_utils/text_norm
"""

from .basic import BasicTextNormalizer as BasicTextNormalizer
from .english import EnglishTextNormalizer as EnglishTextNormalizer


def get_txt_norm(txt_norm):
    assert txt_norm in ["chime8", None]
    if txt_norm is None:
        return None
    elif txt_norm == "chime8":
        return EnglishTextNormalizer()
    else:
        raise NotImplementedError
