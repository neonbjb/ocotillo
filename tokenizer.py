import torch
import re
import inflect
from tokenizers import Tokenizer
from unidecode import unidecode


""" from https://github.com/keithito/tacotron """

'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
'''

_inflect = inflect.engine()
_comma_number_re = re.compile(r'([0-9][0-9\,]+[0-9])')
_decimal_number_re = re.compile(r'([0-9]+\.[0-9]+)')
_pounds_re = re.compile(r'£([0-9\,]*[0-9]+)')
_dollars_re = re.compile(r'\$([0-9\.\,]*[0-9]+)')
_ordinal_re = re.compile(r'[0-9]+(st|nd|rd|th)')
_number_re = re.compile(r'[0-9]+')


def _remove_commas(m):
  return m.group(1).replace(',', '')


def _expand_decimal_point(m):
  return m.group(1).replace('.', ' point ')


def _expand_dollars(m):
  match = m.group(1)
  parts = match.split('.')
  if len(parts) > 2:
    return match + ' dollars'  # Unexpected format
  dollars = int(parts[0]) if parts[0] else 0
  cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
  if dollars and cents:
    dollar_unit = 'dollar' if dollars == 1 else 'dollars'
    cent_unit = 'cent' if cents == 1 else 'cents'
    return '%s %s, %s %s' % (dollars, dollar_unit, cents, cent_unit)
  elif dollars:
    dollar_unit = 'dollar' if dollars == 1 else 'dollars'
    return '%s %s' % (dollars, dollar_unit)
  elif cents:
    cent_unit = 'cent' if cents == 1 else 'cents'
    return '%s %s' % (cents, cent_unit)
  else:
    return 'zero dollars'


def _expand_ordinal(m):
  return _inflect.number_to_words(m.group(0))


def _expand_number(m):
  num = int(m.group(0))
  if num > 1000 and num < 3000:
    if num == 2000:
      return 'two thousand'
    elif num > 2000 and num < 2010:
      return 'two thousand ' + _inflect.number_to_words(num % 100)
    elif num % 100 == 0:
      return _inflect.number_to_words(num // 100) + ' hundred'
    else:
      return _inflect.number_to_words(num, andword='', zero='oh', group=2).replace(', ', ' ')
  else:
    return _inflect.number_to_words(num, andword='')


def normalize_numbers(text):
  text = re.sub(_comma_number_re, _remove_commas, text)
  text = re.sub(_pounds_re, r'\1 pounds', text)
  text = re.sub(_dollars_re, _expand_dollars, text)
  text = re.sub(_decimal_number_re, _expand_decimal_point, text)
  text = re.sub(_ordinal_re, _expand_ordinal, text)
  text = re.sub(_number_re, _expand_number, text)
  return text

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
  ('mrs', 'misess'),
  ('mr', 'mister'),
  ('dr', 'doctor'),
  ('st', 'saint'),
  ('co', 'company'),
  ('jr', 'junior'),
  ('maj', 'major'),
  ('gen', 'general'),
  ('drs', 'doctors'),
  ('rev', 'reverend'),
  ('lt', 'lieutenant'),
  ('hon', 'honorable'),
  ('sgt', 'sergeant'),
  ('capt', 'captain'),
  ('esq', 'esquire'),
  ('ltd', 'limited'),
  ('col', 'colonel'),
  ('ft', 'fort'),
]]


def expand_abbreviations(text):
  for regex, replacement in _abbreviations:
    text = re.sub(regex, replacement, text)
  return text


def expand_numbers(text):
  return normalize_numbers(text)


def lowercase(text):
  return text.lower()


def collapse_whitespace(text):
  return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
  return unidecode(text)


def basic_cleaners(text):
  '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text


def transliteration_cleaners(text):
  '''Pipeline for non-English text that transliterates to ASCII.'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text


def english_cleaners(text):
  '''Pipeline for English text, including number and abbreviation expansion.'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = expand_numbers(text)
  text = expand_abbreviations(text)
  text = collapse_whitespace(text)
  text = text.replace('"', '')
  return text


'''
My own stuff starts here.
'''

class VoiceBpeTokenizer:
    """
    Byte-Pair-Encoding tokenizer trained using the HuggingFace framework specifically for modeling speech to text.
    """

    def __init__(self, vocab_file='data/bpe_lowercase_asr_256.json'):
        if vocab_file is not None:
            self.tokenizer = Tokenizer.from_file(vocab_file)

    def remove_extraneous_punctuation(self, word):
        """
        Removes unnecessary punctuation or replaces it with a consolidates set of punctuation where it might be
        linguistically useful. Most of this function is currently unused since the cleaners we use take care of a lot
        of this punctuation.
        """
        replacement_punctuation = {
            '{': '(', '}': ')',
            '[': '(', ']': ')',
            '`': '\'', '—': '-',
            '—': '-', '`': '\'',
            'ʼ': '\''
        }
        replace = re.compile("|".join([re.escape(k) for k in sorted(replacement_punctuation, key=len, reverse=True)]), flags=re.DOTALL)
        word = replace.sub(lambda x: replacement_punctuation[x.group(0)], word)

        # TODO: some of these are spoken ('@', '%', '+', etc). Integrate them into the cleaners.
        extraneous = re.compile(r'^[@#%_=\$\^&\*\+\\]$')
        word = extraneous.sub('', word)
        return word

    def preprocess_text(self, txt):
        """
        Preprocesses text before tokenization. This should be used on truth data when measuring model performance.
        """
        txt = english_cleaners(txt)
        txt = self.remove_extraneous_punctuation(txt)
        return txt

    def encode(self, txt):
        """
        Encodes txt into a list of integer tokens.
        """
        txt = self.preprocess_text(txt)
        # This is required because we overrode the default way that Hugging Face's BPE tokenizer works (it appends a
        # "start of word" indicator to tokens that start words. In our case, we don't want two different versions of
        # basic tokens like "*th" (as in "there") and "th" (as in "with"). This provably degrades model performance.
        # Instead, [SPACE] is treated as a meta-token.
        txt = txt.replace(' ', '[SPACE]')
        return self.tokenizer.encode(txt).ids

    def decode(self, seq):
        """
        Decodes a list or tensor of integer tokens back into text.
        """
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()
        txt = self.tokenizer.decode(seq, skip_special_tokens=False).replace(' ', '')
        txt = txt.replace('[SPACE]', ' ')
        txt = txt.replace('[STOP]', '')
        txt = txt.replace('[UNK]', '')

        return txt


if __name__ == '__main__':
    input = "Let's test how this tokenizer does with a string of input I just arbitrarily thought up."
    tok = VoiceBpeTokenizer()
    ids = tok.encode(input)
    print(input)
    print(tok.preprocess_text(input))
    print(ids)
    print(tok.decode(ids))
