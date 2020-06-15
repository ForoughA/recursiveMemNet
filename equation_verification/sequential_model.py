from collections import OrderedDict

import sys

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from equation_verification.sequential_model_constants import VOCAB, SYMBOL_CLASSES
from equation_verification.constants import UNARY_FNS, BINARY_FNS, NUMBER_ENCODER, \
                                            NUMBER_DECODER, SYMBOL_ENCODER, CONSTANTS
from equation_verification.dataset_loading import BinaryEqnTree


class LSTMchain(torch.nn.Module):
    def __init__(self, num_hidden, dropout):
        super().__init__()

        self.layer = LSTMnode(num_hidden, num_hidden)

        setattr(self, NUMBER_ENCODER, nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(1, num_hidden)),
            ('sgmd1', nn.Sigmoid()),
            ('linear2', nn.Linear(num_hidden, num_hidden)),
            ('sgmd2', nn.Sigmoid())
        ])))
        setattr(self, NUMBER_DECODER, nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(num_hidden, num_hidden)),
            ('sgmd1', nn.Sigmoid()),
            ('linear2', nn.Linear(num_hidden, 1)),
        ])))
        setattr(self, SYMBOL_ENCODER, nn.Embedding(num_embeddings=len(VOCAB),
                                                   embedding_dim=num_hidden))
        self.bias = nn.Parameter(torch.FloatTensor([0]))
        self.num_hidden = num_hidden
        self.dropout = dropout

    def forward(self, tree, hidden, trace=None):
        """ 
            hidden is a tuple of hidden state and memory
            hidden = (h, c)
        """
        # print('modules:', self._modules)
        # if not tree.is_leaf and tree.function_name not in self._modules:
        #     raise AssertionError("Unknown functional node: %s" % tree.function_name)

        nn_block = getattr(self, SYMBOL_ENCODER)
        if tree.is_binary:
            root_value = VOCAB['Binary_'+tree.function_name]
            root_encoded_value = Variable(torch.LongTensor([root_value])[0])
            inp = nn_block(root_encoded_value)
            h, c = self.layer(inp, hidden, dropout=self.dropout)

            parentheses_value_l = VOCAB['parentheses_(']
            left_parentheses = Variable(torch.LongTensor([parentheses_value_l])[0])
            inp = nn_block(left_parentheses)
            h,c = self.layer(inp, (h,c), dropout=self.dropout)

            hl, cl = self(tree.lchild, (h,c), trace=trace.lchild if trace else None)

            comma_value = VOCAB['parentheses_,']
            comma = Variable(torch.LongTensor([comma_value])[0])
            inp = nn_block(comma)
            h, c = self.layer(inp, (hl,cl), dropout=self.dropout)

            hr, cr = self(tree.rchild, (h,c), trace=trace.rchild if trace else None)

            parentheses_value_r = VOCAB['parentheses_)']
            right_parentheses = Variable(torch.LongTensor([parentheses_value_r])[0])
            inp = nn_block(right_parentheses)
            h, c = self.layer(inp, (hr, cr), dropout=self.dropout)

            return h, c

        elif tree.is_unary:

            if tree.function_name in {NUMBER_DECODER, NUMBER_ENCODER,
                                      SYMBOL_ENCODER}:

                hl, _ = self(tree.lchild, hidden, trace=trace.lchild if trace else None)
                nn_block = getattr(self, tree.function_name)
                inp = nn_block(hl)
                h,c = self.layer(inp, hidden, dropout=self.dropout)

            else:
                root_value = VOCAB['Unary_'+tree.function_name]
                root_encoded_value = Variable(torch.LongTensor([root_value])[0])
                inp = nn_block(root_encoded_value)
                h, c = self.layer(inp, hidden, dropout=self.dropout)

                parentheses_value_l = VOCAB['parentheses_(']
                left_parentheses = Variable(torch.LongTensor([parentheses_value_l])[0])
                inp = nn_block(left_parentheses)
                h,c = self.layer(inp, (h,c), dropout=self.dropout)

                hl, cl = self(tree.lchild, (h,c), trace=trace.lchild if trace else None)

                parentheses_value_r = VOCAB['parentheses_)']
                right_parentheses = Variable(torch.LongTensor([parentheses_value_r])[0])
                inp = nn_block(right_parentheses)
                h, c = self.layer(inp, (hl, cl), dropout=self.dropout)

            return h, c
            
        elif tree.is_leaf:
            c = Variable(torch.LongTensor([0] * self.num_hidden))
            if trace:
                trace.output = tree.encoded_value.tolist()
                trace.memory = c.tolist()
            return tree.encoded_value, c
        else:
            raise RuntimeError("Invalid tree:\n%s" % repr(self))

    def compute_batch(self, batch, trace=None):
        record = []
        total_loss = 0
        for tree, label, depth in batch:
            if trace is not None:
                trace_item = eval(repr(tree))
                trace.append(trace_item)
            else:
                trace_item = None

            hl = Variable(torch.FloatTensor([0] * self.num_hidden))
            cl = Variable(torch.FloatTensor([0] * self.num_hidden))
            lchild, _ = self(tree.lchild, (hl,cl), trace=trace_item.lchild if trace else None)
            hr = Variable(torch.FloatTensor([0] * self.num_hidden))
            cr = Variable(torch.FloatTensor([0] * self.num_hidden))
            rchild, _ = self(tree.rchild, (hr, cr), trace=trace_item.rchild if trace else None)

            if tree.is_numeric():
                assert (tree.lchild.is_a_floating_point and tree.rchild.function_name == NUMBER_DECODER) \
                    or (tree.rchild.is_a_floating_point and tree.lchild.function_name == NUMBER_DECODER)
                loss = (lchild - rchild) * (lchild - rchild)
                correct = math.isclose(lchild.item(), rchild.item(), rel_tol=1e-3)
                if trace_item is not None:
                    trace_item.probability = lchild.item()
            else:
                out = torch.cat((Variable(torch.FloatTensor([0])), torch.dot(lchild, rchild).unsqueeze(0) + self.bias), dim=0)
                loss = - F.log_softmax(out)[round(label.item())]
                correct = F.softmax(out)[round(label.item())].item() > 0.5

                if trace_item is not None:
                    trace_item.probability = F.softmax(out)[1].item()
                    trace_item.correct = correct
                    trace_item.bias = self.bias.tolist()
            assert isinstance(correct, bool)
            record.append({
                "ex": tree,
                "label": round(label.item()),
                "loss": loss.item(),
                "correct": correct,
                "depth": depth,
                "score": out[1].item() # WARNING: only works for symbolic data
            })
            total_loss += loss
        return record, total_loss / len(batch)


class LSTMnode(torch.nn.Module):
    def __init__(self, num_input, num_hidden):
        super().__init__()
        self.data = nn.Linear(num_input*2, num_hidden, bias=True)
        self.forget = nn.Linear(num_input*2, num_hidden, bias=True)
        self.output = nn.Linear(num_input*2, num_hidden, bias=True)
        self.input = nn.Linear(num_input*2, num_hidden, bias=True)

    def forward(self, inp, hidden, trace=None, dropout=None):
        """

        Args:
            inp   : (num_hidden,)
            hidden: ((num_hidden,), (num_hidden,))

        Returns:
            (num_hidden,), (num_hidden)
        """
        h, c = hidden
        h = torch.cat((h, inp), dim=0)
        i = F.sigmoid(self.data(h))
        f = F.sigmoid(self.forget(h))
        o = F.sigmoid(self.output(h))
        u = F.tanh(self.input(h))
        if dropout is None:
            c = i * u + f * c
        else:
            c = i * F.dropout(u,p=dropout,training=self.training) + f * c
        h = o * F.tanh(c)
        if trace:
            trace.output = h.tolist()
            trace.memory = c.tolist()
            trace.i = [f.tolist()]
        return h, c




class RNNchain(torch.nn.Module):
    def __init__(self, num_hidden, dropout):
        super().__init__()

        self.layer  = nn.Linear(2*num_hidden, num_hidden)
        self.act1   = nn.Sigmoid()
        self.layer2 = nn.Linear(num_hidden,num_hidden)
        self.act2   = nn.Sigmoid()

        setattr(self, NUMBER_ENCODER, nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(1, num_hidden)),
            ('sgmd1', nn.Sigmoid()),
            ('linear2', nn.Linear(num_hidden, num_hidden)),
            ('sgmd2', nn.Sigmoid())
        ])))
        setattr(self, NUMBER_DECODER, nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(num_hidden, num_hidden)),
            ('sgmd1', nn.Sigmoid()),
            ('linear2', nn.Linear(num_hidden, 1)),
        ])))
        setattr(self, SYMBOL_ENCODER, nn.Embedding(num_embeddings=len(VOCAB),
                                                   embedding_dim=num_hidden))
        self.bias = nn.Parameter(torch.FloatTensor([0]))
        self.num_hidden = num_hidden
        self.dropout = dropout

    def forward(self, tree, hidden, trace=None):
        """ 
            hidden is the hidden state
            hidden = h
        """
        # print('modules:', self._modules)
        # if not tree.is_leaf and tree.function_name not in self._modules:
        #     raise AssertionError("Unknown functional node: %s" % tree.function_name)

        nn_block = getattr(self, SYMBOL_ENCODER)
        if tree.is_binary:
            root_value = VOCAB['Binary_'+tree.function_name]
            root_encoded_value = Variable(torch.LongTensor([root_value])[0])
            inp = nn_block(root_encoded_value)
            inp = torch.cat((inp,hidden),dim=0)
            h = self.act1(F.dropout(self.layer(inp),p=self.dropout,training=self.training))

            parentheses_value_l = VOCAB['parentheses_(']
            left_parentheses = Variable(torch.LongTensor([parentheses_value_l])[0])
            inp = nn_block(left_parentheses)
            inp = torch.cat((inp,h),dim=0)
            h = self.act1(F.dropout(self.layer(inp),p=self.dropout,training=self.training))

            hl = self(tree.lchild, h, trace=trace.lchild if trace else None)

            comma_value = VOCAB['parentheses_,']
            comma = Variable(torch.LongTensor([comma_value])[0])
            inp = nn_block(comma)
            inp = torch.cat((inp,hl),dim=0)
            h = self.act1(F.dropout(self.layer(inp),p=self.dropout,training=self.training))

            hr = self(tree.rchild, h, trace=trace.rchild if trace else None)

            parentheses_value_r = VOCAB['parentheses_)']
            right_parentheses = Variable(torch.LongTensor([parentheses_value_r])[0])
            inp = nn_block(right_parentheses)
            inp = torch.cat((inp,hr),dim=0)
            h = self.act1(F.dropout(self.layer(inp),p=self.dropout,training=self.training))

            return h

        elif tree.is_unary:

            if tree.function_name in {NUMBER_DECODER, NUMBER_ENCODER,
                                      SYMBOL_ENCODER}:

                hl = self(tree.lchild, hidden, trace=trace.lchild if trace else None)
                nn_block = getattr(self, tree.function_name)
                inp = nn_block(hl)
                inp = torch.cat((inp,hidden),dim=0)
                h = self.act1(F.dropout(self.layer(inp),p=self.dropout,training=self.training))

            else:
                root_value = VOCAB['Unary_'+tree.function_name]
                root_encoded_value = Variable(torch.LongTensor([root_value])[0])
                inp = nn_block(root_encoded_value)
                inp = torch.cat((inp,hidden),dim=0)
                h = self.act1(F.dropout(self.layer(inp),p=self.dropout,training=self.training))

                parentheses_value_l = VOCAB['parentheses_(']
                left_parentheses = Variable(torch.LongTensor([parentheses_value_l])[0])
                inp = nn_block(left_parentheses)
                inp = torch.cat((inp,h),dim=0)
                h = self.act1(F.dropout(self.layer(inp),p=self.dropout,training=self.training))

                hl = self(tree.lchild, h, trace=trace.lchild if trace else None)

                parentheses_value_r = VOCAB['parentheses_)']
                right_parentheses = Variable(torch.LongTensor([parentheses_value_r])[0])
                inp = nn_block(right_parentheses)
                inp = torch.cat((inp,hl),dim=0)
                h = self.act1(F.dropout(self.layer(inp),p=self.dropout,training=self.training))

            return h
            
        elif tree.is_leaf:
            if trace:
                trace.output = tree.encoded_value.tolist()
                #trace.memory = c.tolist()
            return tree.encoded_value
        else:
            raise RuntimeError("Invalid tree:\n%s" % repr(self))

    def compute_batch(self, batch, trace=None):
        record = []
        total_loss = 0
        for tree, label, depth in batch:
            if trace is not None:
                trace_item = eval(repr(tree))
                trace.append(trace_item)
            else:
                trace_item = None

            hl = Variable(torch.FloatTensor([0] * self.num_hidden))
            lchild = self(tree.lchild, hl, trace=trace_item.lchild if trace else None)
            lchild = self.act2(F.dropout(self.layer2(lchild),p=self.dropout,training=self.training))
            hr = Variable(torch.FloatTensor([0] * self.num_hidden))
            rchild = self(tree.rchild, hr, trace=trace_item.rchild if trace else None)
            rchild = self.act2(F.dropout(self.layer2(rchild),p=self.dropout,training=self.training))

            if tree.is_numeric():
                assert (tree.lchild.is_a_floating_point and tree.rchild.function_name == NUMBER_DECODER) \
                    or (tree.rchild.is_a_floating_point and tree.lchild.function_name == NUMBER_DECODER)
                loss = (lchild - rchild) * (lchild - rchild)
                correct = math.isclose(lchild.item(), rchild.item(), rel_tol=1e-3)
                if trace_item is not None:
                    trace_item.probability = lchild.item()
            else:
                out = torch.cat((Variable(torch.FloatTensor([0])), torch.dot(lchild, rchild).unsqueeze(0) + self.bias), dim=0)
                loss = - F.log_softmax(out)[round(label.item())]
                correct = F.softmax(out)[round(label.item())].item() > 0.5

                if trace_item is not None:
                    trace_item.probability = F.softmax(out)[1].item()
                    trace_item.correct = correct
                    trace_item.bias = self.bias.tolist()
            assert isinstance(correct, bool)
            record.append({
                "ex": tree,
                "label": round(label.item()),
                "loss": loss.item(),
                "correct": correct,
                "depth": depth,
                "score": out[1].item() # WARNING: only works for symbolic data
            })
            total_loss += loss
        return record, total_loss / len(batch)
