import ast
import json

import _ast
from sympy import sympify

from equation_verification import constants

NAME = {
    "x": "var_0",
    "y": "var_1",
    "z": "var_2",
    "w": "var_3",
}


def build(t):
    if isinstance(t, _ast.Name):
        if t.id == "Pi" or t.id == "pi":
            return T.PI()
        return T.v(NAME[t.id])
    if isinstance(t, _ast.Num):
        if isinstance(t.n, float):
            return T.f(t.n)
        elif isinstance(t.n, int):
            return T.i(t.n)
    elif isinstance(t, _ast.Call):
        return T(t.func.id, None, *[build(c) for c in t.args])
    elif isinstance(t, _ast.BinOp):
        op = t.op.__class__.__name__
        if op == "Add":
            pass
        elif op == "Mult":
            op = "Mul"
        elif op == "Pow":
            pass
        elif op == "Sub":
            return T("Add", None, build(t.left),
                     T("Mul", None, T.NEGATIVE_ONE(), build(t.right)))
        elif op == "Div":
            return T("Mul", None, build(t.left),
                     T("Pow", None, build(t.right), T.NEGATIVE_ONE()))
        else:
            raise AssertionError("Unknown BinOp %s" % op)
        return T(op, None, build(t.left), build(t.right))
    elif isinstance(t, _ast.UnaryOp):
        op = t.op.__class__.__name__
        if op == "USub":
            if not isinstance(t.operand, _ast.Num):
                raise AssertionError(
                    "- only applies to number, got %s" % ast.dump(t.operand))
            if not t.operand.n == 1:
                return T("Mul", None, T.NEGATIVE_ONE(), build(t.operand))
                # raise AssertionError("- only applies to number 1, got %s" %
                #  str(t.operand.n))
            return T.NEGATIVE_ONE()
        else:
            raise AssertionError("Unknown UnaryOp %s" % op)
    else:
        raise AssertionError("Unknown Node %s" % t.__class__.__name__)

def parse_expression(s):
    tree = ast.parse(s).body[0].value
    if isinstance(tree, ast.Compare):
        raise AssertionError("%s" % tree.__class__.__name__)
    return build(tree)

def parse_equation(s):
    tree = ast.parse(s).body[0].value
    assert isinstance(tree, _ast.Compare)
    return T("Equality", None, build(tree.left), build(tree.comparators[0]))

class T:

    def __init__(self, name, varname, l=None, r=None):
        self.name = name
        self.varname = varname
        self.l = l
        self.r = r
        self.children = [x for x in [l,r] if x is not None]

    @classmethod
    def v(cls, s):
        return T("Symbol", s, None, None)

    @classmethod
    def i(cls, i):
        return T("Integer", str(i), None, None)

    @classmethod
    def f(cls, f):
        return T("Number", str(round(f,2)), None, None)

    @classmethod
    def NEGATIVE_ONE(cls):
        return T("NegativeOne", "-1", None, None)

    @classmethod
    def PI(cls):
        return T("Pi", "pi", None, None)

    def __str__(self):
        if len(self.children) == 0:
            return self.name + "(" + self.varname + ")"
        else:
            return self.name + "(" + ", ".join([c.__str__() for c in self.children]) + ")"

    def sympy_str(self):
        if len(self.children) == 0:
           return self.varname
        else:
            return (self.name if self.name != "Equality" else "Eq")+ "(" + ", ".join(
                [c.sympy_str() for c in self.children]) + ")"

    def inord(self):
        ret = [self]
        for i in range(2):
            if i >= len(self.children):
                ret.append("#")
            else:
                ret.extend(self.children[i].inord())
        return ret

    def depth(self):
        if len(self.children) == 0:
            return 0
        else:
            return 1 + max(c.depth() for c in self.children)

    def size(self):
        if len(self.children) == 0:
            return 1
        else:
            return 1 + sum(c.size() for c in self.children)

    def dump(self, label=True):
        lst = self.inord()
        func = [item.name if isinstance(item, T) else item for item in lst]
        depth = [str(item.depth()) if isinstance(item, T) else item for item in lst]
        nodeNum = []
        n = 0
        numNodes = 0
        for node in func:
            if node != "#":
                nodeNum.append(str(n))
                n += 1
                numNodes += 1
            else:
                nodeNum.append("#")
        vars = [(str(item.varname) if item.varname is not None else "") if isinstance(item, T) else item for item in lst]
        variables = sorted(list(set([var for var in vars if var.startswith("var")])))
        variables = dict(zip(variables, list(range(len(variables)))))
        label = label
        if not label == sympify(self.sympy_str()):
            # print("Wrong", self.sympy_str())
            pass
        assert numNodes == self.size()
        assert isinstance(label, bool)
        ret = {"equation":{
                    "depth": ",".join(depth),
                    "func": ",".join(func),
                    "nodeNum": ",".join(nodeNum),
                    "numNodes": str(numNodes),
                    "vars": ",".join(vars),
                    "variables": variables,
                },
                "label": str(int(label))}
        print(ret)
        return ret

# with open("controlled_generation/axioms_basic.txt", "rt") as f:
#     for line in f:
#         a = parse_equation(line.strip())
#         json.dumps(a.dump())
#         # print(a)
#         # print(json.dumps(a.dump(),sort_keys=True,indent=4))
if __name__ == "__main__":
    print(json.dumps(parse_equation("x * (y + z) == x * y + x * z").dump(), indent=4, sort_keys=True),",")
    print(json.dumps(parse_equation("x * x == x ** 2").dump(), indent=4, sort_keys=True),",")
    print(json.dumps(parse_equation("2 ** -1 * 2 == 1").dump(), indent=4, sort_keys=True),",")
    print(json.dumps(parse_equation("3 ** -1 * 3 == 1").dump(), indent=4, sort_keys=True),",")
    print(json.dumps(parse_equation("4 ** -1 * 4 == 1").dump(), indent=4, sort_keys=True),",")
    print(json.dumps(parse_equation("y ** -1 * y == 1").dump(), indent=4, sort_keys=True),",")