from collections import OrderedDict

UNARY_FNS = ['sin', 'cos', 'csc', 'sec', 'tan', 'cot',
             'asin', 'acos', 'acsc', 'asec', 'atan', 'acot',
             'sinh', 'cosh', 'csch', 'sech', 'tanh', 'coth',
             'asinh', 'acosh', 'acsch', 'asech', 'atanh', 'acoth',
             'exp']

BINARY_FNS = ['Add', 'Mul', 'Pow', 'log']

NUMBER_ENCODER = "Number_enc"
NUMBER_DECODER = "Number_dec"
SYMBOL_ENCODER = "Symbol"

SYMBOL_CLASSES = OrderedDict([
    ('Symbol', ['var_%d' % d for d in range(10)]),
    ('Integer', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -2, -3]),
    ('Rational', ['2/5', '-1/2', '0']),
    ('Float', [0.7])])

CONSTANTS = ['NegativeOne', 'NaN', 'Infinity', 'Exp1', 'Pi', 'One', 'Half']


def get_vocab_key(function_name, value):
    if function_name in ['Integer', 'Rational', 'Float', 'Symbol']:
        key = function_name + '_%s' % str(value)
        print('key:', key)
    else:
        key = function_name

    return key


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

PRETTYNAMES = dict([
    ("Equality", "="),
    (SYMBOL_ENCODER, "EMBEDDING"),
    (NUMBER_ENCODER, "ENCODER"),
    (NUMBER_DECODER, "DECODER"),
    ("Add", "+"),
    ("Pow", "^"),
    ("Mul", "\u00D7"),
    ('NegativeOne', "-1"),
    ('NaN', "nan"),
    ('Infinity', "inf"),
    ('Exp1', 'e'),
    ('Pi', '\u03C0'),
    ('One','1'),
    ('Half', '1/2'),
    (SYMBOL_ENCODER+'_var_0', 'x'),
    (SYMBOL_ENCODER+'_var_1', 'y'),
    (SYMBOL_ENCODER+'_var_2', 'z'),
    (SYMBOL_ENCODER+'_var_3', 'w'),
])