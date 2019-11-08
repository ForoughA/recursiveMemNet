import json

import torch
from torch.utils.data import Dataset, RandomSampler, SequentialSampler, DataLoader
from equation_verification.constants import VOCAB, SYMBOL_CLASSES, CONSTANTS, BINARY_FNS, UNARY_FNS, \
    NUMBER_ENCODER, SYMBOL_ENCODER, NUMBER_DECODER, PRETTYNAMES
from torch.autograd import Variable


def encoded_batch(examples):
    # encode the values in-place
    def encode_value_at_node(node):
        if not node.is_leaf:
            node.value = None
            node.encoded_value = None
        elif node.is_a_floating_point:
            node.value = float(node.function_name)
            node.encoded_value = Variable(torch.FloatTensor([node.value]))
        else:
            node.value = VOCAB[node.function_name]
            node.encoded_value = Variable(torch.LongTensor([node.value])[0])
    encoded_examples = []
    for tree, label, depth in examples:
        tree_copy = eval(repr(tree))
        tree_copy.apply(encode_value_at_node)
        encoded_examples.append((tree_copy, Variable(torch.FloatTensor([float(label)])), depth))
    return encoded_examples

def sequential_sampler(trios, batch_size):
    trios = encoded_batch(trios)
    # packaging the data using PyTorch compatible structures
    dataset = ExampleDataset(trios)
    sampler = SequentialSampler(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        collate_fn=lambda x: x
        # TODO: learn whether it helps to encode here
    )
    return loader

def load_equation_tree_examples(train_path, validation_path, test_path, batch_size=1, numeric=False, eval_depth=None, unify_one_zero=True, filter=None):
    with open(train_path, "rt") as fin:
        train_json = json.loads(fin.read())
        train_trio = build_equation_tree_examples_list(train_json, numeric=numeric, unify_one_zero=unify_one_zero, filter=filter)
        train_trio = encoded_batch(train_trio)
        print("Train size: %d" % len(train_trio))
        # packaging the data using PyTorch compatible structures
        train_dataset = ExampleDataset(train_trio)
        train_sampler = RandomSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=0,
            collate_fn=lambda x:x #TODO: learn whether it helps to encode here
        )
        train_eval_sampler = SequentialSampler(train_dataset)
        train_eval_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_eval_sampler,
            num_workers=0,
            collate_fn=lambda x: x  # TODO: learn whether it helps to encode here
        )
    with open(validation_path, "rt") as fin:
        validation_json = json.loads(fin.read())
        validation_trio = build_equation_tree_examples_list(validation_json, depth=eval_depth, unify_one_zero=unify_one_zero, filter=filter)
        validation_trio = encoded_batch(validation_trio)
        print("Validation size: %d" % len(validation_trio))
        # packaging the data using PyTorch compatible structures
        validation_dataset = ExampleDataset(validation_trio)
        validation_sampler = SequentialSampler(validation_dataset)
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            sampler=validation_sampler,
            num_workers=0,
            collate_fn=lambda x: x
            # TODO: learn whether it helps to encode here
        )
    with open(test_path, "rt") as fin:
        test_json = json.loads(fin.read())
        test_trio = build_equation_tree_examples_list(test_json, depth=eval_depth, unify_one_zero=unify_one_zero, filter=filter)
        test_trio = encoded_batch(test_trio)
        print("test size: %d" % len(test_trio))
        # packaging the data using PyTorch compatible structures
        test_dataset = ExampleDataset(test_trio)
        test_sampler = SequentialSampler(test_dataset)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            sampler=test_sampler,
            num_workers=0,
            collate_fn=lambda x: x
            # TODO: learn whether it helps to encode here
        )
    return train_loader, train_eval_loader, validation_loader, test_loader


def build_equation_tree_examples_list(json_dataset, numeric=False, depth=None, unify_one_zero=True, filter=None):
    """

    Args:
        json_dataset: a json structure with the following schema
        [
            [
                {} --> an example at depth x
                ...
            ] --> a list containing examples at depth x
            ...
        ] --> contains lists of examples at various depths
    Returns:
        a list of trio of BinaryEqnTree and its label and depth
    """
    result = []
    for i, group in enumerate(json_dataset):
        for example in group:
            eqn_tree, label, d = load_single_equation_tree_example(example, unify_one_zero=unify_one_zero)
            assert label in {0, 1}
            if depth is not None and d not in depth:
                continue
            if eqn_tree.is_numeric() and numeric == False:
                continue # do not add numeric examples unless `numeric` flag set
            if eqn_tree.is_numeric() and label == 0:
                continue
            if filter is not None and not filter(eqn_tree):
                continue # filter is a criteria for choosing a tree (tree->bool)
            result.append((eqn_tree, label, d))
    return result


def load_single_equation_tree_example(example, unify_one_zero=True):
    """

    Args:
        example: a dictionary of schema
        {
            "equation": {
                "vars": value for each constant node 'NegativeOne', 'Pi', 'One',
                        'Half', 'Integer', 'Rational', 'Float'
                "numNodes": number of nodes in this tree, discounting #
                "variables": dictionary of ?,
                "depth": depth of each node in this tree
                "nodeNum": unique ids of each node
                "func": the actual list of nodes in this (binary) equation tree,
                    unary functions are still encoded as having two children,
                    the right one being NULL (#)
            },
            "label": "1" if the lhs of the equation equals rhs else "0"
        },

    Returns:
        a BinaryEqnTree corresponding to `example`, paired with its label
    """
    functions = example['equation']['func'].split(",")
    values = example['equation']['vars'].split(",")
    if unify_one_zero:
        functions = ["Integer" if function == "One" else function
                     for function in functions]  # replace One with Integer
        functions = ["Integer" if function == "Rational" and value == "0" else function
                     for function, value in zip(functions, values)] # replace Rational_0 with Integer_0
    eqn_tree = BinaryEqnTree.build_from_preorder(functions, values)
    label = int(example['label'])
    depth = max(int(d) for d in example['equation']["depth"].split(",") if d != "#")
    if eqn_tree.is_numeric():
        if eqn_tree.lchild.maybe_extract_number_constant_node() and eqn_tree.rchild.maybe_extract_number_constant_node():
            # raise AssertionError("both sides of the eqn are number constants: %s" % eqn_tree)
            # data for training NUMBER_ENCODER and NUMBER_DECODER, label is always the right child
            assert eqn_tree.rchild.function_name == NUMBER_ENCODER, 'right child should always be a number constant or label'
            eqn_tree.rchild = eqn_tree.rchild.maybe_extract_number_constant_node()
            eqn_tree.lchild = BinaryEqnTree(NUMBER_DECODER, eqn_tree.lchild, None)
        elif eqn_tree.lchild.maybe_extract_number_constant_node() is not None:
            eqn_tree.lchild = eqn_tree.lchild.maybe_extract_number_constant_node()
            eqn_tree.rchild = BinaryEqnTree(NUMBER_DECODER, eqn_tree.rchild, None)
        elif eqn_tree.rchild.maybe_extract_number_constant_node() is not None:
            eqn_tree.rchild = eqn_tree.rchild.maybe_extract_number_constant_node()
            eqn_tree.lchild = BinaryEqnTree(NUMBER_DECODER, eqn_tree.lchild, None)
        else:
            raise AssertionError("bad number equation: %s" % eqn_tree)
    eqn_tree.raw = example
    eqn_tree.label = label
    eqn_tree.depth = depth
    return eqn_tree, label, depth


class BinaryEqnTree:

    NULL = "#"

    def __init__(self, function_name, lchild, rchild,
                 is_a_floating_point=False, raw=None, label=None, depth=None):
        """

        Args:
            function_name: the name of the node
            lchild: the left child (a BinaryEqnTree or None)
            rchild: the right child (a BinaryEqnTree or None)
        """
        #TODO: make value a more general construct, i.e. a dictionary, or an object so that more than one value can be stored at a node
        if lchild is None and rchild is not None:
            raise ValueError("A tree can have the following children:" + "\n"
            "    lchild=None, rchild=None or" + "\n"
            "    lchild!=None, rchild=None or" + "\n"
            "    lchild!=None, rchild!=None or" + "\n"
            "Got the following instead:" + "\n"
            "    lchild=%s, rchild=%s" % (repr(lchild), repr(rchild)))
        self.function_name = function_name
        self.lchild = lchild
        self.rchild = rchild
        self.is_a_floating_point = is_a_floating_point
        self.value = None
        self.encoded_value = None
        self.is_binary = lchild is not None and rchild is not None
        self.is_unary = lchild is not None and rchild is None
        self.is_leaf = lchild is None and rchild is None
        self.raw = raw
        self.label = label
        self.depth = depth

    def apply(self, fn):
        if self.lchild is not None:
            self.lchild.apply(fn)
        if self.rchild is not None:
            self.rchild.apply(fn)
        fn(self)

    def all(self, pred):
        result = pred(self)
        if self.lchild is not None:
            result = result and self.lchild.all(pred)
        if self.rchild is not None:
            result = result and self.rchild.all(pred)
        return result

    def maybe_extract_number_constant_node(self):
        if self.function_name == NUMBER_ENCODER:
            return BinaryEqnTree(self.lchild.function_name, None, None, is_a_floating_point=True)
        if self.function_name == "Mul":
            if self.lchild.function_name == SYMBOL_ENCODER and \
               self.lchild.lchild.function_name == "NegativeOne" and \
               self.rchild.function_name == NUMBER_ENCODER:
                return BinaryEqnTree("-"+self.rchild.lchild.function_name, None, None, is_a_floating_point=True)
            if self.rchild.function_name == SYMBOL_ENCODER and \
                self.rchild.lchild.function_name == "NegativeOne" and \
                self.lchild.function_name == NUMBER_ENCODER:
                return BinaryEqnTree("-"+self.lchild.lchild.function_name, None, None, is_a_floating_point=True)
        return None

    def is_numeric(self):
        if self.function_name != "Equality":
            raise ValueError("is_numeric should only be called on the root of an equation tree")
        return self._is_numeric()

    def _is_numeric(self):
        if self.is_leaf:
            return self.is_a_floating_point
        if self.is_unary:
            return self.lchild._is_numeric()
        if self.is_binary:
            return self.lchild._is_numeric() or self.rchild._is_numeric()
        raise AssertionError(str(self))

    def __str__(self):
        if self.is_binary:
            return "{}({}, {})".format(self.function_name,
                                       str(self.lchild),
                                       str(self.rchild))
        elif self.is_unary:
            return "{}({})".format(self.function_name,
                                   str(self.lchild))
        elif self.is_leaf:
            return "{}={}".format(self.function_name, self.value)
        else:
            raise RuntimeError("Invalid tree:\n%s" % repr(self))

    def pretty_str(self, prettify=True):
        if self.function_name == "Add":
            lstr = self.lchild.pretty_str()
            rstr = self.rchild.pretty_str()
            return "({} + {})".format(lstr, rstr)
        elif self.function_name == "Mul":
            lstr = self.lchild.pretty_str()
            rstr = self.rchild.pretty_str()
            if self.lchild.function_name == "Add":
                lstr = "(%s)" % lstr
            if self.rchild.function_name == "Add":
                rstr = "(%s)" % rstr
            return "({} * {})".format(lstr, rstr)
        elif self.function_name == "Pow":
            lstr = self.lchild.pretty_str()
            rstr = self.rchild.pretty_str()
            return "({}^{})".format(lstr, rstr)
        elif self.function_name == "log":
            lstr = self.lchild.pretty_str()
            rstr = self.rchild.pretty_str()
            return "log({}, {})".format( lstr, rstr)
        elif self.function_name == "Equality":
            lstr = self.lchild.pretty_str()
            rstr = self.rchild.pretty_str()
            return "{} = {}".format(lstr, rstr)
        elif self.function_name == NUMBER_ENCODER or self.function_name == SYMBOL_ENCODER:
            return self.lchild.pretty_str()
        elif self.is_unary:
            return "({}({}))".format(self.function_name,
                                   self.lchild.pretty_str())
        elif self.is_leaf:
            name = self.function_name
            if not prettify:
                return name
            if name in PRETTYNAMES:
                name = PRETTYNAMES[name]
            elif name.startswith("Integer") or name.startswith("Float") or \
                name.startswith("Symbol") or name.startswith("Rational"):
                name = "_".join(name.split("_")[1:])
            return "{}".format(name)
        else:
            raise RuntimeError("Can't prettify tree :\n%s" % repr(self))


    def __repr__(self):
        return "BinaryEqnTree({},{},{},{},{},{},{})".format(repr(self.function_name),
                                         repr(self.lchild),
                                         repr(self.rchild),
                                         repr(self.is_a_floating_point),
                                         repr(self.raw),
                                         repr(self.label),
                                         repr(self.depth))

    @staticmethod
    def build_from_preorder(functions, values):
        """
        Recovers a BinaryEqnTree from its preorder list.
        WARNING: This method relies on modifying `functions` in-place.

        Args:
            functions: pre-order traversal of a BinaryEqnTree
            values: pre-order traversal of a BinaryEqnTree's values
                (leaf nodes and only leaf nodes have values, currently each leaf
                node has exactly one value, and that is fed as input to the
                function at the leaf, e.g. Leaf node is Symbol and value is "x".
                Here Symbol can be effectively understood as an embedding layer,
                and "x" is the input to that layer. Of course in actual
                execution "x" would be turned into an index into the embedding
                matrix first, so the input would be something like
                torch.LongTensor([12]).)
        Returns:
            the recovered BinaryEqnTree
        """
        function_name = functions.pop(0)
        value = values.pop(0)
        if function_name == BinaryEqnTree.NULL:
            return None

        value = None if value == "" else value

        lchild = BinaryEqnTree.build_from_preorder(functions, values)
        rchild = BinaryEqnTree.build_from_preorder(functions, values)

        if function_name == "Number":
            leaf = BinaryEqnTree(value, None, None, is_a_floating_point=True)
            return BinaryEqnTree(NUMBER_ENCODER, leaf, None)

        elif function_name in SYMBOL_CLASSES:
            leaf = BinaryEqnTree("%s_%s" % (function_name, value), None, None)
            return BinaryEqnTree(SYMBOL_ENCODER, leaf, None)

        elif function_name in CONSTANTS:
            leaf = BinaryEqnTree(function_name, None, None)
            return BinaryEqnTree(SYMBOL_ENCODER, leaf, None)

        elif function_name in BINARY_FNS or function_name in UNARY_FNS or \
                function_name == "Equality":
            return BinaryEqnTree(function_name, lchild, rchild)

        else:
            raise RuntimeError("Uncategorized function name: %s" % function_name)

class ExampleDataset(Dataset):
    """
    Generic Dataset.
    """

    def __init__(self, examples):
        self.examples = [ex for ex in examples]

    def __getitem__(self, item):
        return self.examples[item]

    def __len__(self):
        return len(self.examples)
