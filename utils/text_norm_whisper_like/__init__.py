"""
NOTSOFAR adopts the same text normalizer as the CHiME-8 DASR track.
This code is copied from the CHiME-8 repo:
https://github.com/chimechallenge/chime-utils/tree/main/chime_utils/text_norm
"""

from .basic import BasicTextNormalizer as BasicTextNormalizer
from .english import EnglishTextNormalizer as EnglishTextNormalizer
from whisper.normalizers import EnglishTextNormalizer as OriginalEnglishTextNormalizer


def get_txt_norm(txt_norm):
    if txt_norm is None:
        return None
    elif txt_norm == "chime8":
        return EnglishTextNormalizer()
    elif txt_norm == "whisper":
        return OriginalEnglishTextNormalizer()
    else:
        raise NotImplementedError()
