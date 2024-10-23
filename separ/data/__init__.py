from .sdp import SDP
from .tokenizer import *
from .conll import CoNLL, EnhancedCoNLL
from .ptb import PTB 
from .transform import sdp_to_conll, conll_to_sdp

TKZ_TYPES = {obj.EXTENSION: obj for obj in (Tokenizer, PretrainedTokenizer, OneHotTokenizer, CharacterTokenizer)}
