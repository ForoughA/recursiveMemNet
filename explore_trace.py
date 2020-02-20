import cmd
import pickle
import random
import sys
import os
import traceback
import numpy as np
from collections import deque, OrderedDict
from itertools import product

from graphviz import Digraph
from equation_verification.constants import NUMBER_ENCODER, NUMBER_DECODER, SYMBOL_ENCODER
from argparse import ArgumentParser
random.seed(0)


parser = ArgumentParser()
parser.add_argument("--dump-dir", type=str, default="graphviz_dump")
parser.add_argument("--trace-files", type=str, nargs="+", default=None)
args = parser.parse_args()

if os.path.exists(args.dump_dir):
    txt = None
    while txt not in {'yes', 'no'}:
        txt=input("The folder %s already exists. Are you sure you want to dump output to this folder? [yes/no]" % args.dump_dir)
    if txt == 'yes':
        pass
    else:
        exit(0)

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
    ('Half', '1/2')
])
def visualize(tree, view="False"):
    view = eval(view)
    s = Digraph('structs', filename=os.path.join(args.dump_dir, '%s.gv' % tree.id),
                node_attr={'shape': 'record'},
                )
    s.graph_attr['rankdir'] = 'BT'
    def build(tree, id):
        myid = id[0]
        id[0] += 1
        name = tree.function_name
        # if name in PRETTYNAMES:
        #     name = PRETTYNAMES[tree.function_name]
        # elif name.startswith("Integer") or name.startswith("Float") or name.startswith("Symbol") or name.startswith("Rational"):
        #     name = "_".join(name.split("_")[1:])


        # draw node
        payload = ""
        if hasattr(tree, 'output'):
            if not isinstance(tree.output, list):
                rep_payload = "|{rep | %s}" % str(round(tree.output, 2))
            elif isinstance(tree.output[0], list):
                rep_payload = "|{rep | {%s}}" % "|".join(
                    [("{%s}" % "|".join([str(round(i, 2))
                                         for i in row]))
                     for row in tree.output])
            else:
                rep_payload = "|{rep | %s}" % "|".join(
                    [str(round(i, 2)) for i in tree.output])
            payload += rep_payload
        if hasattr(tree, 'bias'):
            if not isinstance(tree.bias, list):
                rep_payload = "|{bias | %s}" % str(round(tree.bias, 2))
            elif isinstance(tree.bias[0], list):
                rep_payload = "|{bias | {%s}}" % "|".join(
                    [("{%s}" % "|".join([str(round(i, 2))
                                         for i in row]))
                     for row in tree.bias])
            else:
                rep_payload = "|{bias | %s}" % "|".join(
                    [str(round(i, 2)) for i in tree.bias])
            payload += rep_payload
        if myid == 0:
            payload += "|{label| %s} |{correct | %s}| {p_equals| %s}" % (str(tree.label), str(tree.correct), str(round(tree.probability, 2)))
        if hasattr(tree, 'action'):
            if isinstance(tree.action[0], list):
                mem_payload = "|{act | {%s}}" % "|".join(
                    [("{%s}" % "|".join([str(round(i, 2))
                                         for i in row]))
                     for row in tree.action])
            else:
                mem_payload = "|{act | %s}" % "|".join(
                    [str(round(i, 2)) for i in tree.action])
            payload += mem_payload
        if hasattr(tree, 'memory'):
            if isinstance(tree.memory[0], list):
                mem_payload = "|{mem | {%s}}" % "|".join(
                    [("{%s}" % "|".join([str(round(i, 2))
                                         for i in row]))
                     for row in tree.memory])
            else:
                mem_payload = "|{mem | %s}" % "|".join(
                    [str(round(i, 2)) for i in tree.memory])
            payload += mem_payload

        s.node(str(myid), '{{<name>  %s} %s}' % (name, payload))
        if tree.is_leaf:
            pass
        elif tree.is_unary:
            idl = build(tree.lchild, id)
            s.edges([('%d' % idl, '%d' % myid)])
        elif tree.is_binary:
            idl = build(tree.lchild, id)
            idr = build(tree.rchild, id)
            s.edges([( '%d' % idl, '%d' % myid), ('%d' % idr, '%d' % myid)])
        return myid
    build(tree, [0])
    s.render(os.path.join(args.dump_dir, "%s %s" % (tree.pretty_str().replace("/"," div "), tree.id)), view=view)
    l = np.array(tree.lchild.output)
    r = np.array(tree.rchild.output)
    lnorm = np.linalg.norm(l)
    rnorm = np.linalg.norm(r)
    dot = np.dot(l, r)
    cosine_similarity = dot / (lnorm * rnorm)
    #print(tree.lchild.pretty_str(), round(lnorm,2), tree.rchild.pretty_str(), round(rnorm,2), "dot", round(dot,2), round(cosine_similarity,2))


class Shell(cmd.Cmd):

    def __init__(self, trace):
        super().__init__()
        self.curr_obj = trace
        self.stack = deque()

    def do_ls(self, arg):
            if len(self.curr_obj) > 10:
                show = self.curr_obj[:10]
            else:
                show = self.curr_obj
            for i, item in enumerate(show):
                print(i, item)

    def do_cd(self, arg):
            if arg == "..":
                if len(self.stack) == 0:
                    print("already at top level")
                else:
                    self.curr_obj = self.stack.pop()
            elif isinstance(self.curr_obj, dict) and arg in self.curr_obj:
                self.stack.append(self.curr_obj)
                self.curr_obj = self.curr_obj[arg]
            else:
                print("arg is not a folder")

    def do_sel(self, arg):
        kwargs = self._parse(arg)
        self._sel(**kwargs)

    def _sel(self, mode="random", n="1", depth=None, choice=None):
        n = int(n)
        depth = int(depth) if depth else None
        self.selection = []
        domain = self.curr_obj
        if mode == "random":
            if depth is not None:
                domain = [tree for tree in domain if tree[0].depth == depth]
            self.selection = random.sample(domain, n)
        if mode == "select":
            self.selection = [domain[int(idx)] for idx in choice.split(",")]
        if mode == "all":
            self.selection = [x for x in domain]

    def do_plot(self, arg):
        kwargs = self._parse(arg)
        for trees in self.selection:
            print("."*163)
            for tree in trees:
                print("d{4:d} {1:s} {2:>10.2f} {3:s} {0:50s}".format(tree.id, tree.pretty_str(prettify=True), tree.probability, "Right" if abs(int(tree.raw["label"]) - tree.probability) < 0.5 else "Wrong", tree.depth))
                # print(tree.raw["equation"]["func"], tree.raw["equation"]["vars"], tree.raw["label"])
                visualize(tree, **kwargs)

    def onecmd(self, line):
        try:
            return super().onecmd(line)
        except:
            traceback.print_exc()

    def _parse(self, arg):
        args = arg.split()
        kwargs = {}
        for kwarg in args:
            key = kwarg.split("=")[0]
            val = kwarg.split("=")[1]
            kwargs[key] = val
        return kwargs

def main():
    traces = []
    for trace_file in args.trace_files:
        with open(trace_file, "rb") as f:
            trace = pickle.load(f)
            name = trace_file.replace(".", "-").replace("/","-")
            traces.append((name, trace))
    sets = traces[0][1].keys()
    assert all([trace.keys() == sets for _, trace in traces])
    ultimate_trace = dict()
    for set in sets:
        ultimate_trace[set] = []
    for set in traces[0][1]:
        for i, tree in enumerate(trace[set]):
            ultimate_trace[set].append([])
    for name, trace in traces:
        for set in trace:
            for i, tree in enumerate(trace[set]):
                assert str(traces[0][1][set][i]) == str(tree)
                tree.id = "%s_%d_%s" % (set, i, name)
                ultimate_trace[set][i].append(tree)

    ultimate_bins = dict()
    for set in ultimate_trace:
        binnames = product(*[[True, False] for _ in range(len(traces))])
        bins = OrderedDict((name, []) for name in binnames)
        for trees in ultimate_trace[set]:
            id = tuple(tree.correct for tree in trees)
            bins[id].append(trees)
        ultimate_bins[set] = bins
    for set in ultimate_bins:
        print(set)
        for id in list(ultimate_bins[set].keys()):
            ultimate_bins[set][str(id)] = ultimate_bins[set][id]
            idstr = "|"
            for i, correct in enumerate(id):
                if correct:
                    idstr += args.trace_files[i] + "(right)|"
                else:
                    idstr += args.trace_files[i] + "(wrong)|"
            print(idstr, len(ultimate_bins[set][id]), len(ultimate_bins[set][id]) / len(ultimate_trace[set]))
    for set in ultimate_bins:
        for id in list(ultimate_bins[set].keys()):
            if isinstance(id, tuple):
                del ultimate_bins[set][id]
    Shell(ultimate_bins).cmdloop()

if __name__ == "__main__":
    main()
