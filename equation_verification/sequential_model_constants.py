from collections import OrderedDict
from equation_verification.constants import UNARY_FNS, BINARY_FNS, NUMBER_ENCODER, \
                                            NUMBER_DECODER, SYMBOL_ENCODER, CONSTANTS

SYMBOL_CLASSES = OrderedDict([
    ('Symbol', ['var_%d' % d for d in range(10)]),
    ('Integer', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -2, -3]),
    ('Rational', ['2/5', '-1/2', '0']),
    ('Float', [0.7]),
    ('Unary', UNARY_FNS),
    ('Binary', BINARY_FNS),
    ('parentheses',['(', ')', ','])])


def build_vocab():
    vocab = OrderedDict()
    ctr = 0

    for symbol_type, values in SYMBOL_CLASSES.items():
        for value in values:
            vocab["%s_%s" % (symbol_type, str(value))] = ctr
            ctr += 1
    for symbol_name in CONSTANTS:
        vocab[symbol_name] = ctr
        ctr += 1
    return vocab


VOCAB = build_vocab()