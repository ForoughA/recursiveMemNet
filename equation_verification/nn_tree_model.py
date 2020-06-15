from collections import OrderedDict

import sys

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from equation_verification.constants import VOCAB, SYMBOL_CLASSES, CONSTANTS, BINARY_FNS, UNARY_FNS, \
    NUMBER_ENCODER, SYMBOL_ENCODER, NUMBER_DECODER
from equation_verification.dataset_loading import BinaryEqnTree
from equation_verification.sequential_model import LSTMchain, RNNchain

def build_nnTree(model_class,
                 num_hidden,
                 num_embed,
                 memory_size,
                 share_memory_params,
                 dropout,
                 no_op,
                 stack_type,
                 top_k,
                 verbose,
                 tree_node_activation,
                 stack_node_activation,
                 no_pop,
                 disable_sharing,
                 likeLSTM,
                 gate_push_pop,
                 gate_top_k,
                 normalize_action):
    if model_class == "NNTrees":
        model = NNTrees(num_hidden,
                        num_embed,
                        memory_size,
                        share_memory_params,
                        dropout,
                        tree_node_activation)
    elif model_class == "StackNNTrees":
        model = StackNNTrees(num_hidden,
                             num_embed,
                             memory_size,
                             share_memory_params,
                             dropout,
                             tree_node_activation,
                             stack_node_activation,
                             no_op=no_op,
                             stack_type=stack_type,
                             top_k=top_k,
                             verbose=verbose,
                             disable_sharing=disable_sharing)
    elif model_class == "QueueNNTrees":
        model = QueueNNTrees(num_hidden,
                             num_embed,
                             memory_size,
                             share_memory_params,
                             dropout)
    elif model_class == "LSTMTrees":
        model = LSTMTrees(num_hidden,
                             num_embed,
                             memory_size,
                             share_memory_params,
                             dropout)
    elif model_class == 'StackLSTMTrees':
        model = StackLSTMTrees(num_hidden,
                             num_embed,
                             memory_size,
                             share_memory_params,
                             dropout,
                             stack_node_activation,
                             no_op=no_op,
                             stack_type=stack_type,
                             top_k=top_k,
                             verbose=verbose)
    elif model_class == 'StackNNTreesMem2out':
        model = StackNNTreesMem2out(num_hidden,
                             num_embed,
                             memory_size,
                             dropout,
                             tree_node_activation,
                             stack_node_activation,
                             no_op=no_op,
                             stack_type=stack_type,
                             top_k=top_k,
                             verbose=verbose,
                             no_pop=no_pop,
                             likeLSTM=likeLSTM,
                             gate_push_pop=gate_push_pop,
                             gate_top_k=gate_top_k,
                             normalize_action=normalize_action)
    elif model_class == 'LSTMchain':
        model = LSTMchain(num_hidden,
                          dropout)
    elif model_class == 'RNNchain':
        model = RNNchain(num_hidden,
                         dropout)
        
    else:
        raise ValueError("Unknown model class:%s" %model_class)
    return model

class LSTMTrees(torch.nn.Module):
    def __init__(self, num_hidden, num_embed, memory_size, share_memory_params,
                 dropout):
        super().__init__()

        for name in UNARY_FNS:
            setattr(self, name, UnaryLSTMNode(num_hidden, num_hidden))

        for name in BINARY_FNS:
            setattr(self, name, BinaryLSTMNode(num_input=num_hidden,
                                             num_hidden=num_hidden))

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

    def forward(self, tree, trace=None):
        if not tree.is_leaf and tree.function_name not in self._modules:
            raise AssertionError("Unknown functional node: %s" % tree.function_name)
        if tree.is_binary:
            hl, cl = self(tree.lchild, trace=trace.lchild if trace else None)
            hr, cr = self(tree.rchild, trace=trace.rchild if trace else None)
            nn_block = getattr(self, tree.function_name)
            return nn_block((hl, cl), (hr, cr), trace=trace, dropout=self.dropout)
        elif tree.is_unary:
            hl, cl = self(tree.lchild, trace=trace.lchild if trace else None)
            nn_block = getattr(self, tree.function_name)
            if tree.function_name in {NUMBER_DECODER, NUMBER_ENCODER,
                                      SYMBOL_ENCODER}:
                h = nn_block(hl)
                c = cl
                if trace:
                    trace.output = h.tolist()
                    trace.memory = c.tolist()
            else:
                h, c = nn_block((hl, cl), trace=trace, dropout=self.dropout)
            return h, c
        elif tree.is_leaf:
            c = Variable(torch.FloatTensor([0] * self.num_hidden))
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

            lchild, _ = self(tree.lchild, trace=trace_item.lchild if trace else None)
            rchild, _ = self(tree.rchild, trace=trace_item.rchild if trace else None)

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
                "score": out[1].item() if not tree.is_numeric() else lchild.item() # WARNING: only works for symbolic data
            })
            total_loss += loss
        return record, total_loss / len(batch)


class BinaryLSTMNode(torch.nn.Module):

    def __init__(self, num_input, num_hidden):
        super().__init__()
        self.data_left = nn.Linear(num_input, num_hidden, bias=False)
        self.data_right = nn.Linear(num_input, num_hidden, bias=False)
        self.data_bias = nn.Parameter(torch.FloatTensor([0] * num_hidden))
        self.forget_left_by_left = nn.Linear(num_input, num_hidden, bias=False)
        self.forget_left_by_right = nn.Linear(num_input, num_hidden, bias=False)
        self.forget_right_by_left = nn.Linear(num_input, num_hidden, bias=False)
        self.forget_right_by_right = nn.Linear(num_input, num_hidden, bias=False)
        self.forget_bias_left = nn.Parameter(torch.FloatTensor([0] * num_hidden))
        self.forget_bias_right = nn.Parameter(torch.FloatTensor([0] * num_hidden))
        self.output_left = nn.Linear(num_input, num_hidden, bias=False)
        self.output_right = nn.Linear(num_input, num_hidden, bias=False)
        self.output_bias = nn.Parameter(torch.FloatTensor([0] * num_hidden))
        self.input_left = nn.Linear(num_input, num_hidden, bias=False)
        self.input_right = nn.Linear(num_input, num_hidden, bias=False)
        self.input_bias = nn.Parameter(torch.FloatTensor([0] * num_hidden))

    def forward(self, input_left, input_right, trace=None, dropout=None):
        """

        Args:
            input_left: ((num_hidden,), (num_hidden,))
            input_right: ((num_hidden,), (num_hidden,))

        Returns:
            (num_hidden,), (num_hidden)
        """
        hl, cl = input_left
        hr, cr = input_right
        i = F.sigmoid(self.data_left(hl) + self.data_right(hr) + self.data_bias)
        f_left = F.sigmoid(self.forget_left_by_left(hl) +
                           self.forget_left_by_right(
                               hr) + self.forget_bias_left)
        f_right = F.sigmoid(self.forget_right_by_left(hl) +
                           self.forget_right_by_right(
                               hr) + self.forget_bias_right)
        o = F.sigmoid(self.output_left(hl) + self.output_right(hr) + self.output_bias)
        u = F.tanh(self.input_left(hl) + self.input_right(hr) + self.input_bias)
        if dropout is None:
            c = i * u + f_left * cl + f_right * cr
        else:
            c = i * F.dropout(u,p=dropout,training=self.training) + f_left * cl + f_right * cr
        h = o * F.tanh(c)
        if trace:
            trace.output = h.tolist()
            trace.memory = c.tolist()
            trace.i = [f_left.tolist(), f_right.tolist()]
        return h, c


class UnaryLSTMNode(torch.nn.Module):
    def __init__(self, num_input, num_hidden):
        super().__init__()
        self.data = nn.Linear(num_input, num_hidden, bias=True)
        self.forget = nn.Linear(num_input, num_hidden, bias=True)
        self.output = nn.Linear(num_input, num_hidden, bias=True)
        self.input = nn.Linear(num_input, num_hidden, bias=True)

    def forward(self, inp, trace=None, dropout=None):
        """

        Args:
            inp: ((num_hidden,), (num_hidden,))

        Returns:
            (num_hidden,), (num_hidden)
        """
        h, c = inp
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

class NNTrees(torch.nn.Module):

    def __init__(self, num_hidden, num_embed, memory_size, share_memory_params, 
                 dropout, tree_node_activation):
        super().__init__()

        for name in UNARY_FNS:
            setattr(self, name, UnaryNNNode(num_hidden, num_hidden,
                                            activation=tree_node_activation))

        for name in BINARY_FNS:
            setattr(self, name, BinaryNNNode(num_input=num_hidden,
                                             num_output=num_hidden,
                                             activation=tree_node_activation))

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
        self.dropout = dropout

    def forward(self, tree, trace=None):
        if not tree.is_leaf and tree.function_name not in self._modules:
            raise AssertionError("Unknown functional node: %s" % tree.function_name)
        if tree.is_binary:
            lchild = self(tree.lchild, trace=trace.lchild if trace else None)
            rchild = self(tree.rchild, trace=trace.rchild if trace else None)
            nn_block = getattr(self, tree.function_name)
            return nn_block(lchild, rchild, trace=trace, dropout=self.dropout)
        elif tree.is_unary:
            child = self(tree.lchild, trace=trace.lchild if trace else None)
            nn_block = getattr(self, tree.function_name)
            if tree.function_name in {NUMBER_DECODER, NUMBER_ENCODER, SYMBOL_ENCODER}:
                return nn_block(child)
            return nn_block(child, dropout=self.dropout)
        elif tree.is_leaf:
            if trace:
                trace.output = tree.encoded_value.tolist()
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

            lchild = self(tree.lchild, trace=trace_item.lchild if trace else None)
            rchild = self(tree.rchild, trace=trace_item.rchild if trace else None)

            if tree.is_numeric():
                assert (tree.lchild.is_a_floating_point and tree.rchild.function_name == NUMBER_DECODER) \
                    or (tree.rchild.is_a_floating_point and tree.lchild.function_name == NUMBER_DECODER)
                loss = (lchild - rchild) * (lchild - rchild)
                correct = math.isclose(lchild.item(), rchild.item(), rel_tol=1e-3)
            else:
                out = torch.cat((Variable(torch.FloatTensor([0])), torch.dot(lchild, rchild).unsqueeze(0) + self.bias), dim=0)
                loss = - F.log_softmax(out)[round(label.item())]
                correct = F.softmax(out)[round(label.item())].item() > 0.5

                if trace_item is not None:
                    trace_item.probability = F.softmax(out)[1].item()
                    trace_item.correct = correct
            assert isinstance(correct, bool)
            record.append({
                "ex": tree,
                "label": round(label.item()),
                "loss": loss.item(),
                "correct": correct,
                "depth": depth,
                "score": out[1].item() if not tree.is_numeric() else lchild.item() # WARNING: only works for symbolic data
            })
            total_loss += loss
        return record, total_loss / len(batch)


class BinaryNNNode(torch.nn.Module):
    def __init__(self, num_input, num_output, activation):
        super().__init__()
        self.linear = nn.Linear(num_input * 2, num_output, bias=True)
        if activation == "sigmoid":
            self.activation = F.sigmoid
        elif activation == "tanh":
            self.activation = F.tanh
        else:
            raise ValueError("Unhandled activation: %s" % activation)

    def forward(self, input_left, input_right, trace=None, dropout=None):
        """

        Args:
            input_left: (num_input,)
            input_right: (num_output,)

        Returns:
            (num_output,)
        """
        inp = torch.cat((input_left, input_right), dim=0)
        tmp = self.linear(inp)
        output = self.activation(tmp)
        if dropout is not None:
            output = F.dropout(output,p=dropout,training=self.training)
        else:
            pass
        if trace:
            trace.output = output.tolist()
        return output


class UnaryNNNode(torch.nn.Module):

    def __init__(self, num_input, num_output, activation):
        super().__init__()
        self.linear = nn.Linear(num_input, num_output, bias=True)
        if activation == "sigmoid":
            self.activation = F.sigmoid
        elif activation == "tanh":
            self.activation = F.tanh
        else:
            raise ValueError("Unhandled activation: %s" % activation)

    def forward(self, inp, trace=None, dropout=None):
        """

        Args:
            input: (num_input,)

        Returns:
            (num_output,)
        """
        tmp = self.linear(inp)
        output = self.activation(tmp)
        if dropout is not None:
            output = F.dropout(output,p=dropout,training=self.training)
        else:
            pass
        if trace:
            trace.output = output.tolist()
        return output


class StackNNTrees(torch.nn.Module):

    def __init__(self, num_hidden, num_embed, memory_size, share_memory_params,\
                 dropout, tree_node_activation, stack_node_activation,
                 no_op=False, stack_type='simple', top_k=5, verbose=False, disable_sharing=False):
        super().__init__()

        for name in UNARY_FNS:

            setattr(self, name, UnaryMemoryNNNode(memory_size=memory_size,
                                                  num_input=num_hidden,
                                                  num_output=num_hidden,
                                                  activation=tree_node_activation,
                                                  memory_type=stack_type,
                                                  top_k=top_k))
            if disable_sharing:

                if stack_type == 'full_stack_gated':
                    setattr(self, name+'_stack', UnaryFullStackNNNodeGated(
                        stack_size=memory_size,
                        num_input=num_hidden,
                        activation=stack_node_activation,
                        no_op=no_op, verbose=verbose))

                elif stack_type == 'full_queue_gated':
                    setattr(self, name + '_stack', UnaryFullQueueNNNodeGated(
                        stack_size=memory_size,
                        num_input=num_hidden,
                        activation=stack_node_activation,
                        no_op=no_op,
                        verbose=verbose))
                else:
                    raise AssertionError(
                        'Unknown stack type, please use one of these: "full_queue_gated", '
                        '"full_stack_gated". ')

        for name in BINARY_FNS:
            setattr(self, name, BinaryMemoryNNNode(memory_size=memory_size,
                                                  num_input=num_hidden,
                                                  num_output=num_hidden,
                                                  activation=tree_node_activation,
                                                  memory_type=stack_type,
                                                  top_k=top_k))
            if disable_sharing:
                if stack_type == 'full_stack_gated':
                    setattr(self, name + '_stack', BinaryFullStackNNNodeGated(
                        stack_size=memory_size,
                        num_input=num_hidden,
                        activation=stack_node_activation,
                        no_op=no_op, verbose=verbose))
                elif stack_type == 'full_queue_gated':
                    setattr(self, name + '_stack', BinaryFullQueueNNNodeGated(
                        stack_size=memory_size,
                        num_input=num_hidden,
                        activation=stack_node_activation,
                        no_op=no_op, verbose=verbose))
                else:
                    raise AssertionError(
                        'Unknown stack type, please use one of these: '
                        '"full_queue_gated", '
                        '"full_stack_gated". ')

        setattr(self, NUMBER_ENCODER, nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(1, num_hidden)),
            ('relu1', nn.ReLU()),
            ('linear2', nn.Linear(num_hidden, num_hidden)),
            ('relu2', nn.ReLU())
        ])))
        setattr(self, NUMBER_DECODER, nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(num_hidden, num_hidden)),
            ('relu1', nn.ReLU()),
            ('linear2', nn.Linear(num_hidden, 1)),
        ])))
        setattr(self, SYMBOL_ENCODER, nn.Embedding(num_embeddings=len(VOCAB),
                                   embedding_dim=num_hidden))


        if stack_type == 'simple':

            self.binary_stack = BinaryStackNNNode(stack_size=memory_size,
                                                  num_input=num_hidden,
                                                  activation=stack_node_activation,
                                                  no_op=no_op, verbose=verbose)

            self.unary_stack = UnaryStackNNNode(stack_size=memory_size,
                                                num_input=num_hidden,
                                                activation=stack_node_activation,
                                                no_op=no_op, verbose=verbose)

        elif stack_type == 'simple_gated':

            self.binary_stack = BinaryStackNNNodeGated(stack_size=memory_size,
                                                  num_input=num_hidden,
                                                  activation=stack_node_activation,
                                                  no_op=no_op, verbose=verbose)
            self.unary_stack = UnaryStackNNNodeGated(stack_size=memory_size,
                                                num_input=num_hidden,
                                                activation=stack_node_activation,
                                                no_op=no_op, verbose=verbose)

        elif stack_type == 'nn_stack':

            self.binary_stack = BinaryNNStackNNNode(stack_size=memory_size,
                                                  num_input=num_hidden, no_op=no_op)

            self.unary_stack = UnaryNNStackNNNode(stack_size=memory_size,
                                                num_input=num_hidden, no_op=no_op)

        elif stack_type == 'full_stack':

            self.binary_stack = BinaryFullStackNNNode(stack_size=memory_size,
                                                  num_input=num_hidden, no_op=no_op)

            self.unary_stack = UnaryFullStackNNNode(stack_size=memory_size,
                                                num_input=num_hidden, no_op=no_op) # TODO: replace with Full unary after debugging
        
        elif stack_type == 'full_stack_gated':

            self.binary_stack = BinaryFullStackNNNodeGated(stack_size=memory_size,
                                                  num_input=num_hidden,
                                                  activation=stack_node_activation,
                                                  no_op=no_op, verbose=verbose)
            self.unary_stack = UnaryFullStackNNNodeGated(stack_size=memory_size,
                                                num_input=num_hidden,
                                                activation=stack_node_activation,
                                                no_op=no_op, verbose=verbose)

        elif stack_type == 'add_stack':
            
            self.binary_stack = BinaryAddStackNNNode(stack_size=memory_size,
                                                  num_input=num_hidden,
                                                  activation=stack_node_activation,
                                                  no_op=no_op, verbose=verbose)

            self.unary_stack = UnaryAddStackNNNode(stack_size=memory_size,
                                                num_input=num_hidden,
                                                activation=stack_node_activation,
                                                no_op=no_op, verbose=verbose)
        elif stack_type == 'full_queue_gated':
            self.binary_stack = BinaryFullQueueNNNodeGated(
                stack_size=memory_size,
                num_input=num_hidden,
                activation=stack_node_activation,
                no_op=no_op, verbose=verbose)
            self.unary_stack = UnaryFullQueueNNNodeGated(stack_size=memory_size,
                                                     num_input=num_hidden,
                                                     activation=stack_node_activation,
                                                     no_op=no_op,
                                                     verbose=verbose)
        else:
            raise AssertionError('Unknown stack type, please use one of these: "simple", "nn_stack", "full_stack", "add_stack", "simple_gated", "full_stack_gated". ')

        self.stack_size = memory_size
        self.stack_type = stack_type
        self.num_hidden = num_hidden
        self.verbose = verbose

        self.bias = nn.Parameter(torch.FloatTensor([0]))
        self.disable_sharing = disable_sharing
        print("disable_sharing", self.disable_sharing, file=sys.stderr)
        self.dropout = dropout

    def forward(self, tree, trace=None):
        if not tree.is_leaf and tree.function_name not in self._modules:
            raise AssertionError("Unknown functional node: %s" % tree.function_name)
        if tree.is_binary:
            if self.verbose:
                print('evaluating left child {0} and right child {1} and root node {2}:'\
                      .format(tree.lchild.function_name, tree.rchild.function_name, tree.function_name))
            lchild, lstack = self(tree.lchild, trace=trace.lchild if trace else None)
            rchild, rstack = self(tree.rchild, trace=trace.rchild if trace else None)
            nn_block = getattr(self, tree.function_name)
            stack_block = self.binary_stack if not self.disable_sharing else getattr(self, tree.function_name + "_stack")
            output = nn_block(lchild, rchild, lstack, rstack, trace=trace, dropout=self.dropout)
            output_stack = stack_block(lchild, rchild, lstack, rstack, trace=trace)
            if self.verbose:
                print('output at node {0} is {1}'.format(tree.function_name, output))
                print('output stack at node {0} is {1}'.format(tree.function_name, output_stack))
            return output, output_stack
        elif tree.is_unary:
            if self.verbose:
                print('evaluating child {0} and root node {1}:'\
                      .format(tree.lchild.function_name, tree.function_name))
            child, stack = self(tree.lchild, trace=trace.lchild if trace else None)
            nn_block = getattr(self, tree.function_name)
            if tree.function_name in {NUMBER_DECODER, NUMBER_ENCODER, SYMBOL_ENCODER}:
                output = nn_block(child)
                output_stack = stack
            else:
                stack_block = self.unary_stack if not self.disable_sharing else getattr(self, tree.function_name + "_stack")
                output = nn_block(child, stack, trace=trace, dropout=self.dropout)
                output_stack = stack_block(child, stack, trace=trace)
            
            if self.verbose:
                print('output at node {0} is {1}'.format(tree.function_name, output))
                print('output stack at node {0} is {1}'.format(tree.function_name, output_stack))
            return output, output_stack
        elif tree.is_leaf:
            if self.verbose:
                print('evaluating leaf node {0}:'.format(tree.function_name))
            if self.stack_type == 'full_stack' or self.stack_type == 'full_stack_gated' or self.stack_type == 'full_queue_gated':
                leaf_stack = torch.zeros((self.stack_size,self.num_hidden), requires_grad=True)
            else:
                leaf_stack = torch.zeros((self.stack_size,1), requires_grad=True)
            leaf = tree.encoded_value
            if self.verbose:
                print('output at leaf node {0} is {1}'.format(tree.function_name, leaf))
                print('output stack at leaf node {0} is {1}'.format(tree.function_name, leaf_stack))
            if trace:
                trace.output = leaf.tolist()
                trace.memory = leaf_stack.tolist()
            return leaf, leaf_stack
        else:
            raise RuntimeError("Invalid tree:\n%s" % repr(self))


    def compute_batch(self, batch, trace=None):
        record = []
        total_loss = 0
        for tree, label, depth in batch:
            if self.verbose:
                print('STARTING NEW EQUATION')
                print('EQUATION: {0}'.format(tree.pretty_str()))
                if tree.pretty_str() == 'Integer_3^Integer_0 * Symbol_var_0 = Integer_4 + asec(Integer_0)^Integer_0':
                    print('RAW: {0}'.format(tree.raw['equation']['func']))
                    input('enter')
                print('LABEL: {0}'.format(label))

            if trace is not None:
                trace_item = eval(repr(tree))
                trace.append(trace_item)
            else:
                trace_item = None

            lchild, _ = self(tree.lchild, trace=trace_item.lchild if trace else None)
            rchild, _ = self(tree.rchild, trace=trace_item.rchild if trace else None)
            # thought, do we need to look at the stack at the root?

            if tree.is_numeric():
                assert (tree.lchild.is_a_floating_point and tree.rchild.function_name == NUMBER_DECODER) \
                    or (tree.rchild.is_a_floating_point and tree.lchild.function_name == NUMBER_DECODER)
                loss = (lchild - rchild) * (lchild - rchild)
                correct = math.isclose(lchild.item(), rchild.item(), rel_tol=1e-3)
            else:
                out = torch.cat((Variable(torch.FloatTensor([0])), torch.dot(lchild, rchild).unsqueeze(0) + self.bias), dim=0)
                loss = - F.log_softmax(out)[round(label.item())]
                correct = F.softmax(out)[round(label.item())].item() > 0.5

                if trace_item is not None:
                    trace_item.probability = F.softmax(out)[1].item()
                    trace_item.correct = correct
            assert isinstance(correct, bool)
            record.append({
                "ex": tree,
                "label": round(label.item()),
                "loss": loss.item(),
                "correct": correct,
                "depth": depth,
                "score": out[1].item() if not tree.is_numeric() else lchild.item() # WARNING: only works for symbolic data
            })
            total_loss += loss
        return record, total_loss / len(batch)


class BinaryStackNNNode(torch.nn.Module):

    def __init__(self, stack_size, num_input, activation, no_op=False, verbose=False):
        super().__init__()

        if activation == "sigmoid":
            self.stack_activations = F.sigmoid
        elif activation == "tanh":
            self.stack_activations = F.tanh
        else:
            raise ValueError("Unhandled activation: %s" % activation)
        if no_op:
            self.action = nn.Linear(num_input * 2, 3, bias=True)
            self.no_op_linear = nn.Linear(2, 1, bias=True)
        else:
            self.action = nn.Linear(num_input * 2, 2, bias=True)
        
        self.input_linear = nn.Linear(num_input * 2, 1, bias=True)
        self.pop_linear = nn.Linear(2, 1, bias=True)
        self.push_linear = nn.Linear(2, 1, bias=True)

        self.stack_size = stack_size
        self.no_op = no_op
        self.verbose = verbose

    def forward(self, input_left, input_right, stack_left, stack_right, trace=None):
        """
        Args:
            input_left: (num_input,)
            input_right: (num_output,)
            stack_left: (stack_size,)
            stack_right: (stack_size,)

        Returns:
            [(stack_size,)]
        """

        inp = torch.cat((input_left, input_right), dim=0) # (2*num_input,)
        tmp = self.action(inp)
        action = F.softmax(tmp) # (2,) or (3,)
        if self.verbose:
            print('action at node is {0}'.format(action))

        input_stack = torch.cat((stack_left, stack_right), dim=1) # (stack_size, 2)
        input_stack = torch.cat((input_stack, Variable(torch.FloatTensor([[0,0]]))), dim=0) # (stack_size + 1, 2)
        pop_indices = torch.LongTensor(range(1, self.stack_size+1))
        pop = torch.index_select(input_stack, 0, pop_indices)
        pop = self.pop_linear(pop)
        pop = self.stack_activations(pop) # (stack_size,)

        push_indices = torch.LongTensor(range(0, self.stack_size-1))
        push = torch.index_select(input_stack, 0, push_indices)
        push = self.push_linear(push)
        push_input = self.input_linear(inp).unsqueeze(0)
        push = torch.cat((push_input, push), dim=0)
        push = self.stack_activations(push) # (stack_size,)

        if self.no_op:
            no_op_indices = torch.LongTensor(range(0, self.stack_size))
            no_op = torch.index_select(input_stack, 0, no_op_indices) # (stack_size,2)
            no_op = self.no_op_linear(no_op) # (stack_size,)
            no_op = self.stack_activations(no_op)
            push_pop_cat = torch.cat((push, pop, no_op), dim=1) # (stack_size,3)
        else:
            push_pop_cat = torch.cat((push, pop), dim=1) # (stack_size,2)

        stack = torch.matmul(push_pop_cat, action.unsqueeze(1)) # (stack_size, 1)

        if trace:
            trace.action = action.tolist()
            trace.memory = stack.tolist()
        return stack


class UnaryStackNNNode(torch.nn.Module):

    def __init__(self, stack_size, num_input, activation, no_op=False, verbose=False):
        super().__init__()

        if activation == "sigmoid":
            self.stack_activations = F.sigmoid
        elif activation == "tanh":
            self.stack_activations = F.tanh
        else:
            raise ValueError("Unhandled activation: %s" % activation)
        if no_op:
            self.action = nn.Linear(num_input, 3, bias=True)
            self.no_op_linear = nn.Linear(1, 1, bias=True)
        else:
            self.action = nn.Linear(num_input, 2, bias=True)
        self.input_linear = nn.Linear(num_input, 1, bias=True)
        self.pop_linear = nn.Linear(1, 1, bias=True)
        self.push_linear = nn.Linear(1, 1, bias=True)

        self.stack_size = stack_size
        self.no_op = no_op
        self.verbose = verbose


    def forward(self, inp, stack, trace=None):
        """
        Args:
            inp: (num_input,)
            stack: (stack_size,)

        Returns:
            (stack_size,)
        """

        tmp = self.action(inp)
        action =  F.softmax(tmp) # (2,)
        if self.verbose:
            print('action at node is {0}'.format(action))

        stack = torch.cat((stack, Variable(torch.FloatTensor([[0]]))), dim=0) # (stack_size + 1, 1)
        pop_indices = torch.LongTensor(range(1, self.stack_size+1))
        pop = torch.index_select(stack, 0, pop_indices)
        pop = self.pop_linear(pop)
        pop = self.stack_activations(pop) # (stack_size,)

        push_indices = torch.LongTensor(range(0, self.stack_size-1))
        push = torch.index_select(stack, 0, push_indices)
        push = self.push_linear(push)
        push_input = self.input_linear(inp).unsqueeze(0)
        push = torch.cat((push_input, push), dim=0)
        push = self.stack_activations(push) # (stack_size,)

        if self.no_op:
            no_op_indices = torch.LongTensor(range(0, self.stack_size))
            no_op = torch.index_select(stack, 0, no_op_indices) # (stack_size,)
            no_op = self.no_op_linear(no_op) # (stack_size,)
            no_op = self.stack_activations(no_op)
            push_pop_cat = torch.cat((push, pop, no_op), dim=1) # (stack_size,3)
        else:
            push_pop_cat = torch.cat((push, pop), dim=1) # (stack_size,2)

        stack = torch.matmul(push_pop_cat, action.unsqueeze(1)) # (stack_size, 1)

        if trace:
            trace.action = action.tolist()
            trace.memory = stack.tolist()
        return stack


class BinaryStackNNNodeGated(torch.nn.Module):

    def __init__(self, stack_size, num_input, activation, no_op=False, verbose=False):
        super().__init__()

        if activation == "sigmoid":
            self.stack_activations = F.sigmoid
        elif activation == "tanh":
            self.stack_activations = F.tanh
        else:
            raise ValueError("Unhandled activation: %s" % activation)
        if no_op:
            self.action = nn.Linear(num_input * 2, 3, bias=True)
        else:
            self.action = nn.Linear(num_input * 2, 2, bias=True)
        
        self.gate_linear_l = nn.Linear(num_input*2, 1, bias=True)
        self.gate_linear_r = nn.Linear(num_input*2, 1, bias=True)
        self.input_linear = nn.Linear(num_input * 2, 1, bias=True)

        self.stack_size = stack_size
        self.no_op = no_op
        self.verbose = verbose

    def forward(self, input_left, input_right, stack_left, stack_right, trace=None):
        """
        Args:
            input_left: (num_input,)
            input_right: (num_output,)
            stack_left: (stack_size,)
            stack_right: (stack_size,)

        Returns:
            [(stack_size,)]
        """

        inp = torch.cat((input_left, input_right), dim=0) # (2*num_input,)
        tmp = self.action(inp)
        action = F.softmax(tmp) # (2,) or (3,)
        if self.verbose:
            print('action at node is {0}'.format(action))

        left_gate = self.gate_linear_l(inp)
        left_gate = F.sigmoid(left_gate) # (1,)
        right_gate = self.gate_linear_r(inp)
        right_gate = F.sigmoid(right_gate) # (1,)
        input_stack = left_gate * stack_left + right_gate * stack_right # (stack_size,)
        input_stack = torch.cat((input_stack, torch.zeros((1,1), dtype=torch.float32, requires_grad=True)), dim=0) # (stack_size+1,)

        push_input = self.input_linear(inp).unsqueeze(0)
        push_input = self.stack_activations(push_input)
        
        push_indices = torch.LongTensor(range(0, self.stack_size-1))
        push = torch.index_select(input_stack, 0, push_indices)
        push = torch.cat((push_input, push), dim=0)

        pop_indices = torch.LongTensor(range(1, self.stack_size+1))
        pop = torch.index_select(input_stack, 0, pop_indices)

        if self.no_op:
            no_op_indices = torch.LongTensor(range(0, self.stack_size))
            no_op = torch.index_select(input_stack, 0, no_op_indices) # (stack_size,)
            push_pop_cat = torch.cat((push, pop, no_op), dim=1) # (stack_size,3)
        else:
            push_pop_cat = torch.cat((push, pop), dim=1) # (stack_size,2)

        stack = torch.matmul(push_pop_cat, action.unsqueeze(1)) # (stack_size, 1)

        if trace:
            trace.action = action.tolist()
            trace.memory = stack.tolist()
        return stack

class UnaryStackNNNodeGated(torch.nn.Module):

    def __init__(self, stack_size, num_input, activation, no_op=False, verbose=False):
        super().__init__()

        if activation == "sigmoid":
            self.stack_activations = F.sigmoid
        elif activation == "tanh":
            self.stack_activations = F.tanh
        else:
            raise ValueError("Unhandled activation: %s" % activation)
        if no_op:
            self.action = nn.Linear(num_input, 3, bias=True)
        else:
            self.action = nn.Linear(num_input, 2, bias=True)
        
        self.gate_linear = nn.Linear(num_input, 1, bias=True)
        self.input_linear = nn.Linear(num_input, 1, bias=True)

        self.stack_size = stack_size
        self.no_op = no_op
        self.verbose = verbose


    def forward(self, inp, stack, trace=None):
        """
        Args:
            inp: (num_input,)
            stack: (stack_size,)

        Returns:
            (stack_size,)
        """

        tmp = self.action(inp)
        action =  F.softmax(tmp) # (2,)
        if self.verbose:
            print('action at node is {0}'.format(action))

        gate = self.gate_linear(inp)
        gate = F.sigmoid(gate) # (1,)
        stack = gate * stack
        stack = torch.cat((stack, torch.zeros((1,1), dtype=torch.float32, requires_grad=True)), dim=0) # (stack_size+1,)

        push_input = self.input_linear(inp).unsqueeze(0)
        push_input = self.stack_activations(push_input)

        push_indices = torch.LongTensor(range(0, self.stack_size-1))
        push = torch.index_select(stack, 0, push_indices)
        push = torch.cat((push_input, push), dim=0)

        pop_indices = torch.LongTensor(range(1, self.stack_size+1))
        pop = torch.index_select(stack, 0, pop_indices)


        if self.no_op:
            no_op_indices = torch.LongTensor(range(0, self.stack_size))
            no_op = torch.index_select(stack, 0, no_op_indices) # (stack_size,)
            push_pop_cat = torch.cat((push, pop, no_op), dim=1) # (stack_size,3)
        else:
            push_pop_cat = torch.cat((push, pop), dim=1) # (stack_size,2)

        stack = torch.matmul(push_pop_cat, action.unsqueeze(1)) # (stack_size, 1)

        if trace:
            trace.action = action.tolist()
            trace.memory = stack.tolist()
        return stack


class BinaryNNStackNNNode(torch.nn.Module):

    def __init__(self, stack_size, num_input, no_op=False):
        super().__init__()

        self.activation = F.sigmoid
        # TODO: add a no-action dimension
        if no_op:
            self.action = nn.Linear(num_input * 2, 3, bias=True)
            self.no_op_linear = nn.Linear(2*stack_size, stack_size, bias=True)
        else:
            self.action = nn.Linear(num_input * 2, 2, bias=True)
        self.input_linear = nn.Linear(num_input * 2, 1, bias=True)
        self.pop_linear = nn.Linear(2*(stack_size-1), stack_size, bias=True)
        self.push_linear = nn.Linear(2*(stack_size-1), stack_size-1, bias=True)
        self.stack_activations = F.sigmoid

        self.stack_size = stack_size
        self.no_op = no_op

    def forward(self, input_left, input_right, stack_left, stack_right, trace=None):
        """
        Args:
            input_left: (num_input,)
            input_right: (num_output,)
            stack_left: (stack_size,)
            stack_right: (stack_size,)

        Returns:
            [(stack_size,)]
        """

        inp = torch.cat((input_left, input_right), dim=0) # (2*num_input,)
        tmp = self.action(inp)
        action = F.softmax(tmp) # (2,) or (3,)

        input_stack = torch.cat((stack_left, stack_right), dim=1) # (stack_size,2)
        pop_indices = torch.LongTensor(range(1, self.stack_size))
        pop = torch.index_select(input_stack, 0, pop_indices)
        pop = pop.reshape(-1)
        pop = self.pop_linear(pop).unsqueeze(1)
        pop = self.stack_activations(pop) # (stack_size,)

        push_indices = torch.LongTensor(range(0, self.stack_size-1))
        push = torch.index_select(input_stack, 0, push_indices) # (stack_size-1,)
        push = push.reshape(-1)
        push = self.push_linear(push).unsqueeze(1)
        push_input = self.input_linear(inp).unsqueeze(0)
        push = torch.cat((push_input, push), dim=0)
        push = self.stack_activations(push) # (stack_size,)

        if self.no_op:
            no_op_indices = torch.LongTensor(range(0, self.stack_size))
            no_op = torch.index_select(input_stack, 0, no_op_indices) # (stack_size,2)
            no_op = no_op.reshape(-1)
            no_op = self.no_op_linear(no_op).unsqueeze(1) # (stack_size,)
            no_op = self.stack_activations(no_op)
            push_pop_cat = torch.cat((push, pop, no_op), dim=1) # (stack_size,3)
        else:
            push_pop_cat = torch.cat((push, pop), dim=1) # (stack_size,2)

        stack = torch.matmul(push_pop_cat, action.unsqueeze(1)) # (stack_size, 1)
        if trace:
            trace.action = action.tolist()
            trace.memory = stack.tolist()
        return stack


class UnaryNNStackNNNode(torch.nn.Module):

    def __init__(self, stack_size, num_input, no_op=False):
        super().__init__()

        self.activation = F.sigmoid
        if no_op:
            self.action = nn.Linear(num_input, 3, bias=True)
            self.no_op_linear = nn.Linear(stack_size, stack_size, bias=True)
        else:
            self.action = nn.Linear(num_input, 2, bias=True)
        self.input_linear = nn.Linear(num_input, 1, bias=True)
        self.pop_linear = nn.Linear(stack_size-1, stack_size, bias=True)
        self.push_linear = nn.Linear(stack_size-1, stack_size-1, bias=True)
        self.stack_activations = F.sigmoid

        self.stack_size = stack_size
        self.no_op = no_op


    def forward(self, inp, stack, trace=None):
        """
        Args:
            inp: (num_input,)
            stack: (stack_size,)

        Returns:
            (stack_size,)
        """

        tmp = self.action(inp)
        action =  F.softmax(tmp) # (2,)

        pop_indices = torch.LongTensor(range(1, self.stack_size))
        pop = torch.index_select(stack, 0, pop_indices)
        pop = pop.reshape(-1)
        pop = self.pop_linear(pop).unsqueeze(1)
        pop = self.stack_activations(pop) # (stack_size,)

        push_indices = torch.LongTensor(range(0, self.stack_size-1))
        push = torch.index_select(stack, 0, push_indices)
        push = push.reshape(-1)
        push = self.push_linear(push).unsqueeze(1)
        push_input = self.input_linear(inp).unsqueeze(0)
        push = torch.cat((push_input, push), dim=0)
        push = self.stack_activations(push) # (stack_size,)

        if self.no_op:
            no_op_indices = torch.LongTensor(range(0, self.stack_size))
            no_op = torch.index_select(stack, 0, no_op_indices) # (stack_size,)
            no_op = no_op.reshape(-1)
            no_op = self.no_op_linear(no_op).unsqueeze(1) # (stack_size,)
            no_op = self.stack_activations(no_op)
            push_pop_cat = torch.cat((push, pop, no_op), dim=1) # (stack_size,3)
        else:
            push_pop_cat = torch.cat((push, pop), dim=1) # (stack_size,2)

        stack = torch.matmul(push_pop_cat, action.unsqueeze(1)) # (stack_size, 1)

        if trace:
            trace.action = action.tolist()
            trace.memory = stack.tolist()
        return stack


class BinaryFullStackNNNode(torch.nn.Module):

    def __init__(self, stack_size, num_input, no_op=False):
        super().__init__()


        self.activation = F.sigmoid
        if no_op:
            self.action = nn.Linear(num_input * 2, 3, bias=True)
            self.no_op_linear = nn.Linear(num_input*2, num_input, bias=True)
        else:
            self.action = nn.Linear(num_input * 2, 2, bias=True)
        self.input_linear = nn.Linear(num_input * 2, num_input, bias=True)
        self.pop_linear   = nn.Linear(num_input * 2, num_input, bias=True)
        self.push_linear  = nn.Linear(num_input * 2, num_input, bias=True)
        self.stack_activations = F.sigmoid

        self.stack_size = stack_size
        self.num_input  = num_input
        self.no_op      = no_op

    def forward(self, input_left, input_right, stack_left, stack_right, trace=None):
        """
        Args:
            input_left: (num_input,)
            input_right: (num_output,)
            stack_left: (stack_size,)
            stack_right: (stack_size,)

        Returns:
            [(stack_size,)]
        """

        inp = torch.cat((input_left, input_right), dim=0) # (2*num_input,)
        tmp = self.action(inp)
        action = F.softmax(tmp) # (2,) or (3,)

        input_stack = torch.cat((stack_left, stack_right), dim=1) # (stack_size, 2*num_input)
        input_stack = torch.cat((input_stack, torch.zeros((1,2*self.num_input), dtype=torch.float32, requires_grad=True)), dim=0) # (stack_size + 1, 2*num_input)
        pop_indices = torch.LongTensor(range(1, self.stack_size+1))
        pop = torch.index_select(input_stack, 0, pop_indices)
        pop = self.pop_linear(pop)
        pop = self.stack_activations(pop) # (stack_size,num_input)

        push_indices = torch.LongTensor(range(0, self.stack_size-1))
        push = torch.index_select(input_stack, 0, push_indices)
        push = self.push_linear(push)
        push_input = self.input_linear(inp).unsqueeze(0)
        push = torch.cat((push_input, push), dim=0)
        push = self.stack_activations(push) # (stack_size,input)

        if self.no_op:
            no_op_indices = torch.LongTensor(range(0, self.stack_size))
            no_op = torch.index_select(input_stack, 0, no_op_indices)  # (stack_size, 2*num_input)
            no_op = self.no_op_linear(no_op)  # (stack_size,num_input)
            no_op = self.stack_activations(no_op)
            push_pop_cat = torch.cat((push, pop, no_op), dim=1)  # (stack_size,3*num_input)
            stack = push_pop_cat[:,           0 : self.num_input]   * action[0] + \
                    push_pop_cat[:,   self.num_input : 2*self.num_input] * action[1] + \
                    push_pop_cat[:, self.num_input*2 : 3*self.num_input] * action[2]  # (stack_size, num_input)
        else:
            push_pop_cat = torch.cat((push, pop), dim=1)  # (stack_size, 2*num_input)
            stack = push_pop_cat[:, 0: self.num_input] * action[0] + \
                    push_pop_cat[:, self.num_input: 2 * self.num_input] * action[1]  # (stack_size, num_input)

        if trace:
            trace.action = action.tolist()
            trace.memory = stack.tolist()
        return stack

class UnaryFullStackNNNode(torch.nn.Module):

    def __init__(self, stack_size, num_input, no_op=False):
        super().__init__()

        self.activation = F.sigmoid
        if no_op:
            self.action = nn.Linear(num_input, 3, bias=True)
            self.no_op_linear = nn.Linear(num_input, num_input, bias=True)
        else:
            self.action = nn.Linear(num_input, 2, bias=True)
        self.input_linear = nn.Linear(num_input, num_input, bias=True)
        self.pop_linear = nn.Linear(num_input, num_input, bias=True)
        self.push_linear = nn.Linear(num_input, num_input, bias=True)
        self.stack_activations = F.sigmoid

        self.stack_size = stack_size
        self.num_input  = num_input
        self.no_op      = no_op


    def forward(self, inp, stack, trace=None):
        """
        Args:
            inp: (num_input,)
            stack: (stack_size,num_input)

        Returns:
            (stack_size, num_input)
        """

        tmp = self.action(inp)
        action =  F.softmax(tmp) # (2,)

        stack = torch.cat((stack, torch.zeros((1,self.num_input), dtype=torch.float32, requires_grad=True)), dim=0) # (stack_size + 1, num_input)
        pop_indices = torch.LongTensor(range(1, self.stack_size+1))
        pop = torch.index_select(stack, 0, pop_indices)
        pop = self.pop_linear(pop)
        pop = self.stack_activations(pop) # (stack_size,)

        push_indices = torch.LongTensor(range(0, self.stack_size-1))
        push = torch.index_select(stack, 0, push_indices)
        push = self.push_linear(push)
        push_input = self.input_linear(inp).unsqueeze(0)
        push = torch.cat((push_input, push), dim=0)
        push = self.stack_activations(push) # (stack_size,)

        if self.no_op:
            no_op_indices = torch.LongTensor(range(0, self.stack_size))
            no_op = torch.index_select(stack, 0, no_op_indices) # (stack_size,)
            no_op = self.no_op_linear(no_op) # (stack_size,)
            no_op = self.stack_activations(no_op)
            push_pop_cat = torch.cat((push, pop, no_op), dim=1) # (stack_size,3*num_input)
            stack = push_pop_cat[:,           0 : self.num_input]   * action[0] + \
                    push_pop_cat[:,   self.num_input : 2*self.num_input] * action[1] + \
                    push_pop_cat[:, self.num_input*2 : 3*self.num_input] * action[2]  # (stack_size, num_input)
        else:
            push_pop_cat = torch.cat((push, pop), dim=1) # (stack_size,2)
            stack = push_pop_cat[:, 0: self.num_input] * action[0] + \
                    push_pop_cat[:, self.num_input: 2 * self.num_input] * action[1]  # (stack_size, num_input)

        # stack = torch.matmul(push_pop_cat, action.unsqueeze(1)) # (stack_size, 1)

        if trace:
            trace.action = action.tolist()
            trace.memory = stack.tolist()
        return stack


class BinaryFullStackNNNodeGated(torch.nn.Module):

    def __init__(self, stack_size, num_input, activation, no_op=False, verbose=False, 
                 no_pop=False, likeLSTM=False, gate_push_pop=False, normalize_action=False):
        super().__init__()

        assert no_op==False or no_pop==False, 'at least one of these should be False'


        if activation == "sigmoid":
            self.stack_activations = F.sigmoid
        elif activation == "tanh":
            self.stack_activations = F.tanh
        else:
            raise ValueError("Unhandled activation: %s" % activation)
        
        if gate_push_pop == False:
            if no_op:
                self.action = nn.Linear(num_input * 2, 3, bias=True)
            elif no_pop or not no_op:
                self.action = nn.Linear(num_input * 2, 2, bias=True)
        else:
            self.push_gate_linear = nn.Linear(num_input * 2, num_input, bias=True)
            if no_pop == False:
                self.pop_gate_linear = nn.Linear(num_input * 2, num_input, bias=True)
            if no_op or no_pop:
                self.no_op_gate_linear = nn.Linear(num_input * 2, num_input, bias=True) 


        self.gate_linear_l = nn.Linear(num_input*2, num_input, bias=True)
        self.gate_linear_r = nn.Linear(num_input*2, num_input, bias=True)
        self.input_linear = nn.Linear(num_input*2, num_input, bias=True)
        if likeLSTM:
            self.data_linear = nn.Linear(num_input*2, num_input, bias=True)

        self.stack_size = stack_size
        self.num_input  = num_input
        self.no_op      = no_op
        self.verbose = verbose
        self.no_pop = no_pop
        self.likeLSTM = likeLSTM
        self.gate_push_pop = gate_push_pop
        self.normalize_action = normalize_action


    def forward(self, input_left, input_right, stack_left, stack_right, trace=None, dropout=None):
        """
        Args:
            input_left: (num_input,)
            input_right: (num_output,)
            stack_left: (stack_size,num_input)
            stack_right: (stack_size,num_input)

        Returns:
            [(stack_size,num_input)]
        """

        inp = torch.cat((input_left, input_right), dim=0) # (2*num_input,)
        if self.gate_push_pop == False:
            tmp = self.action(inp)
            action = F.softmax(tmp) # (2,) or (3,)
            if self.verbose:
                print('action at node is {0}'.format(action))
        else:
            tmp_push = self.push_gate_linear(inp)
            push_gate = F.sigmoid(tmp_push)
            if self.no_pop == False:
                tmp_pop = self.pop_gate_linear(inp)
                pop_gate = F.sigmoid(tmp_pop)
            if self.no_op or self.no_pop:
                tmp_no_op = self.no_op_gate_linear(inp)
                no_op_gate = F.sigmoid(tmp_no_op)

        left_gate = self.gate_linear_l(inp)
        left_gate = F.sigmoid(left_gate) # (1,)
        right_gate = self.gate_linear_r(inp)
        right_gate = F.sigmoid(right_gate) # (1,)

        input_stack = left_gate * stack_left + right_gate * stack_right # (stack_size, num_input)
        input_stack = torch.cat((input_stack, torch.zeros((1,self.num_input), dtype=torch.float32, requires_grad=True)), dim=0) # (stack_size+1, num_input)

        push_input = self.input_linear(inp).unsqueeze(0) # (1,num_input)
        push_input = self.stack_activations(push_input)
        if dropout is not None:
            push_input = F.dropout(push_input, p=dropout, training=self.training)
        if self.likeLSTM:
            data_gate = self.data_linear(inp)
            data_gate = F.sigmoid(data_gate)
            push_input = data_gate * push_input

        push_indices = torch.LongTensor(range(0, self.stack_size-1))
        push = torch.index_select(input_stack, 0, push_indices)
        push = torch.cat((push_input, push), dim=0) # (stack_size, num_input)

        if self.no_pop == False:
            pop_indices = torch.LongTensor(range(1, self.stack_size+1))
            pop = torch.index_select(input_stack, 0, pop_indices)  # (stack_size,num_input)

        no_op_indices = torch.LongTensor(range(0, self.stack_size))
        no_op = torch.index_select(input_stack, 0, no_op_indices)  # (stack_size, num_input)

        if self.gate_push_pop==False:
            if self.no_op:
                stack = action[0] * push + action[1] * pop  + action[2] * no_op
            elif self.no_pop:
                stack = action[0] * push + action[1] * no_op
            else:
                stack = action[0] * push + action[1] * pop
        else:
            if self.no_op:
                if self.normalize_action:
                    tmpAction = torch.cat((push_gate.unsqueeze(0), pop_gate.unsqueeze(0), no_op_gate.unsqueeze(0)), dim=0)
                    tmpAction = F.softmax(tmpAction, dim=0)
                    push_gate = tmpAction[0]
                    pop_gate = tmpAction[1]
                    no_op_gate = tmpAction[2]
                stack = push_gate * push + pop_gate * pop + no_op_gate * no_op
            elif self.no_pop:
                if self.normalize_action:
                    tmpAction = torch.cat((push_gate.unsqueeze(0), no_op_gate.unsqueeze(0)), dim=0)
                    tmpAction = F.softmax(tmpAction, dim=0)
                    push_gate = tmpAction[0]
                    no_op_gate = tmpAction[1]
                stack = push_gate * push + no_op_gate * no_op
            else:
                if self.normalize_action:
                    tmpAction = torch.cat((push_gate.unsqueeze(0), pop_gate.unsqueeze(0)), dim=0)
                    tmpAction = F.softmax(tmpAction, dim=0)
                    push_gate = tmpAction[0]
                    pop_gate = tmpAction[1]
                stack = push_gate * push + pop_gate * pop

        if trace:
            if self.gate_push_pop == False:
                trace.action = action.tolist()
            else:
                trace.action = []
                trace.action.append(push_gate.tolist())
                if self.no_pop == False:
                    trace.action.append(pop_gate.tolist())
                if self.no_op == True:
                    trace.action.append(no_op_gate.tolist())
            trace.memory = stack.tolist()
            trace.i = [left_gate.tolist(), right_gate.tolist()]
        return stack


class UnaryFullStackNNNodeGated(torch.nn.Module):

    def __init__(self, stack_size, num_input, activation, no_op=False, 
                 verbose=False, no_pop=False, likeLSTM=False, 
                 gate_push_pop=False, normalize_action=False):
        super().__init__()

        assert no_op==False or no_pop==False, 'at least one of these should be False'

        if activation == "sigmoid":
            self.stack_activations = F.sigmoid
        elif activation == "tanh":
            self.stack_activations = F.tanh
        else:
            raise ValueError("Unhandled activation: %s" % activation)

        if gate_push_pop == False:
            if no_op:
                self.action = nn.Linear(num_input, 3, bias=True)
            elif no_pop or not no_op:
                self.action = nn.Linear(num_input, 2, bias=True)
        else:
            self.push_gate_linear = nn.Linear(num_input, num_input, bias=True)
            if no_pop == False:
                self.pop_gate_linear = nn.Linear(num_input, num_input, bias=True)
            if no_op or no_pop:
                self.no_op_gate_linear = nn.Linear(num_input, num_input, bias=True)
                

        self.gate_linear = nn.Linear(num_input, num_input, bias=True)
        self.input_linear = nn.Linear(num_input, num_input, bias=True)
        if likeLSTM:
            self.data_linear = nn.Linear(num_input, num_input, bias=True)

        self.stack_size = stack_size
        self.num_input  = num_input
        self.no_op      = no_op
        self.verbose = verbose
        self.no_pop = no_pop
        self.likeLSTM = likeLSTM
        self.gate_push_pop = gate_push_pop
        self.normalize_action = normalize_action


    def forward(self, inp, stack, trace=None, dropout=None):
        """
        Args:
            inp: (num_input,)
            stack: (stack_size,num_input)

        Returns:
            (stack_size, num_input)
        """

        if self.gate_push_pop == False:
            tmp = self.action(inp)
            action =  F.softmax(tmp) # (2,)
            if self.verbose:
                print('action at node is {0}'.format(action))
        else:
            tmp_push = self.push_gate_linear(inp)
            push_gate = F.sigmoid(tmp_push)
            if self.no_pop == False:
                tmp_pop = self.pop_gate_linear(inp)
                pop_gate = F.sigmoid(tmp_pop)
            if self.no_op or self.no_pop:
                tmp_no_op = self.no_op_gate_linear(inp)
                no_op_gate = F.sigmoid(tmp_no_op)

        gate = self.gate_linear(inp)
        gate = F.sigmoid(gate) # (1,)
        stack = gate * stack
        stack = torch.cat((stack, torch.zeros((1,self.num_input), dtype=torch.float32, requires_grad=True)), dim=0) # (stack_size + 1, num_input)

        push_input = self.input_linear(inp).unsqueeze(0) # (1,num_input)
        push_input = self.stack_activations(push_input)
        if dropout is not None:
            push_input = F.dropout(push_input, p=dropout, training=self.training)
        if self.likeLSTM:
            data_gate = self.data_linear(inp)
            data_gate = F.sigmoid(data_gate)
            push_input = data_gate * push_input

        push_indices = torch.LongTensor(range(0, self.stack_size-1))
        push = torch.index_select(stack, 0, push_indices)
        push = torch.cat((push_input, push), dim=0)

        if self.no_pop == False:
            pop_indices = torch.LongTensor(range(1, self.stack_size+1))
            pop = torch.index_select(stack, 0, pop_indices)

        no_op_indices = torch.LongTensor(range(0, self.stack_size))
        no_op = torch.index_select(stack, 0, no_op_indices) # (stack_size,num_input)

        if self.gate_push_pop == False:
            if self.no_op:
                stack = action[0] * push + action[1] * pop + action[2] * no_op
            elif self.no_pop:
                stack = action[0] * push + action[1] * no_op
            else:
                stack = action[0] * push + action[1] * pop
        else:
            if self.no_op:
                if self.normalize_action:
                    tmpAction = torch.cat((push_gate.unsqueeze(0), pop_gate.unsqueeze(0), no_op_gate.unsqueeze(0)), dim=0)
                    tmpAction = F.softmax(tmpAction, dim=0)
                    push_gate = tmpAction[0]
                    pop_gate = tmpAction[1]
                    no_op_gate = tmpAction[2]
                stack = push_gate * push + pop_gate * pop + no_op_gate * no_op
            elif self.no_pop:
                if self.normalize_action:
                    tmpAction = torch.cat((push_gate.unsqueeze(0), no_op_gate.unsqueeze(0)), dim=0)
                    tmpAction = F.softmax(tmpAction, dim=0)
                    push_gate = tmpAction[0]
                    no_op_gate = tmpAction[1]
                stack = push_gate * push + no_op_gate * no_op
            else:
                if self.normalize_action:
                    tmpAction = torch.cat((push_gate.unsqueeze(0), pop_gate.unsqueeze(0)), dim=0)
                    tmpAction = F.softmax(tmpAction, dim=0)
                    push_gate = tmpAction[0]
                    pop_gate = tmpAction[1]
                stack = push_gate * push + pop_gate * pop

        if trace:
            if self.gate_push_pop == False:
                trace.action = action.tolist()
            else:
                trace.action = []
                trace.action.append(push_gate.tolist())
                if self.no_pop == False:
                    trace.action.append(pop_gate.tolist())
                if self.no_op == True:
                    trace.action.append(no_op_gate.tolist())
            trace.memory = stack.tolist()
            trace.i = [gate.tolist()]
        return stack


class BinaryAddStackNNNode(torch.nn.Module):

    def __init__(self, stack_size, num_input, activation, no_op=False, verbose=False):
        super().__init__()

        if activation == "sigmoid":
            self.stack_activations = F.sigmoid
        elif activation == "tanh":
            self.stack_activations = F.tanh
        else:
            raise ValueError("Unhandled activation: %s" % activation)
        if no_op:
            self.action = nn.Linear(num_input * 2, 3, bias=True)
        else:
            self.action = nn.Linear(num_input * 2, 2, bias=True)
        self.input_linear = nn.Linear(num_input * 2, 1, bias=True)

        self.stack_size = stack_size
        self.no_op = no_op
        self.verbose = verbose

    def forward(self, input_left, input_right, stack_left, stack_right, trace=None):
        """
        Args:
            input_left: (num_input,)
            input_right: (num_output,)
            stack_left: (stack_size,)
            stack_right: (stack_size,)

        Returns:
            [(stack_size,)]
        """

        inp = torch.cat((input_left, input_right), dim=0) # (2*num_input,)
        tmp = self.action(inp)
        action = F.softmax(tmp) # (2,) or (3,)
        if self.verbose:
            print('action at node is {0}'.format(action))

        input_stack = stack_left + stack_right # (stack_size, 1)
        input_stack = torch.cat((input_stack, Variable(torch.FloatTensor([[0]]))), dim=0) # (stack_size + 1, 2)
        pop_indices = torch.LongTensor(range(1, self.stack_size+1))
        pop = torch.index_select(input_stack, 0, pop_indices)
        # QUESTION: add activations regardless?
        # pop = self.stack_activations(pop) # (stack_size,)

        push_indices = torch.LongTensor(range(0, self.stack_size-1))
        push = torch.index_select(input_stack, 0, push_indices)
        push_input = self.input_linear(inp).unsqueeze(0)# (1,)
        push_input = self.stack_activations(push_input) 
        push = torch.cat((push_input, push), dim=0)
        # push = self.stack_activations(push) # (stack_size,)

        if self.no_op:
            no_op_indices = torch.LongTensor(range(0, self.stack_size))
            no_op = torch.index_select(input_stack, 0, no_op_indices) # (stack_size,1)
            # no_op = self.stack_activations(no_op)
            push_pop_cat = torch.cat((push, pop, no_op), dim=1) # (stack_size,3)
        else:
            push_pop_cat = torch.cat((push, pop), dim=1) # (stack_size,2)

        stack = torch.matmul(push_pop_cat, action.unsqueeze(1)) # (stack_size, 1)

        if trace:
            trace.action = action.tolist()
            trace.memory = stack.tolist()
        return stack

class UnaryAddStackNNNode(torch.nn.Module):

    def __init__(self, stack_size, num_input, activation, no_op=False, verbose=False):
        super().__init__()

        if activation == "sigmoid":
            self.stack_activations = F.sigmoid
        elif activation == "tanh":
            self.stack_activations = F.tanh
        else:
            raise ValueError("Unhandled activation: %s" % activation)
        if no_op:
            self.action = nn.Linear(num_input, 3, bias=True)
        else:
            self.action = nn.Linear(num_input, 2, bias=True)
        self.input_linear = nn.Linear(num_input, 1, bias=True)

        self.stack_size = stack_size
        self.no_op = no_op
        self.verbose = verbose


    def forward(self, inp, stack, trace=None):
        """
        Args:
            inp: (num_input,)
            stack: (stack_size,)

        Returns:
            (stack_size,)
        """

        tmp = self.action(inp)
        action =  F.softmax(tmp) # (2,)
        if self.verbose:
            print('action at node is {0}'.format(action))

        stack = torch.cat((stack, Variable(torch.FloatTensor([[0]]))), dim=0) # (stack_size + 1, 1)
        pop_indices = torch.LongTensor(range(1, self.stack_size+1))
        pop = torch.index_select(stack, 0, pop_indices)
        # pop = self.stack_activations(pop) # (stack_size,)

        push_indices = torch.LongTensor(range(0, self.stack_size-1))
        push = torch.index_select(stack, 0, push_indices)
        push_input = self.input_linear(inp).unsqueeze(0)
        push_input = self.stack_activations(push_input)
        push = torch.cat((push_input, push), dim=0)
        # push = self.stack_activations(push) # (stack_size,)

        if self.no_op:
            no_op_indices = torch.LongTensor(range(0, self.stack_size))
            no_op = torch.index_select(stack, 0, no_op_indices) # (stack_size,)
            # no_op = self.stack_activations(no_op)
            push_pop_cat = torch.cat((push, pop, no_op), dim=1) # (stack_size,3)
        else:
            push_pop_cat = torch.cat((push, pop), dim=1) # (stack_size,2)

        stack = torch.matmul(push_pop_cat, action.unsqueeze(1)) # (stack_size, 1)

        if trace:
            trace.action = action.tolist()
            trace.memory = stack.tolist()
        return stack



class QueueNNTrees(torch.nn.Module):

    def __init__(self, num_hidden, num_embed, memory_size, share_memory_params,
                 dropout):
        super().__init__()

        for name in UNARY_FNS:
            setattr(self, name, UnaryMemoryNNNode(memory_size=memory_size,
                                                  num_input=num_hidden,
                                                  num_output=num_hidden))

        for name in BINARY_FNS:
            setattr(self, name, BinaryMemoryNNNode(memory_size=memory_size,
                                                  num_input=num_hidden,
                                                  num_output=num_hidden))

        setattr(self, NUMBER_ENCODER, nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(1, num_hidden)),
            ('relu1', nn.ReLU()),
            ('linear2', nn.Linear(num_hidden, num_hidden)),
            ('relu2', nn.ReLU())
        ])))
        setattr(self, NUMBER_DECODER, nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(num_hidden, num_hidden)),
            ('relu1', nn.ReLU()),
            ('linear2', nn.Linear(num_hidden, 1)),
        ])))
        setattr(self, SYMBOL_ENCODER, nn.Embedding(num_embeddings=len(VOCAB),
                                   embedding_dim=num_hidden))

        self.binary_queue = BinaryQueueNNNode(queue_size=memory_size,
                                              num_input=num_hidden)

        self.unary_queue = UnaryQueueNNNode(queue_size=memory_size,
                                            num_input=num_hidden)

        self.queue_size = memory_size

        self.bias = nn.Parameter(torch.FloatTensor([0]))

    def forward(self, tree, trace=None):
        if not tree.is_leaf and tree.function_name not in self._modules:
            raise AssertionError("Unknown functional node: %s" % tree.function_name)
        if tree.is_binary:
            lchild, lqueue = self(tree.lchild, trace=trace.lchild if trace else None)
            rchild, rqueue = self(tree.rchild, trace=trace.rchild if trace else None)
            nn_block = getattr(self, tree.function_name)
            queue_block = self.binary_queue
            output = nn_block(lchild, rchild, lqueue, rqueue, trace=trace)
            output_queue = queue_block(lchild, rchild, lqueue, rqueue, trace=trace)
            return output, output_queue
        elif tree.is_unary:
            child, queue = self(tree.lchild, trace=trace.lchild if trace else None)
            nn_block = getattr(self, tree.function_name)
            queue_block = self.unary_queue
            if tree.function_name in {NUMBER_DECODER, NUMBER_ENCODER, SYMBOL_ENCODER}:
                output = nn_block(child)
                output_queue = queue
            else:
                output = nn_block(child, queue, trace=trace)
                output_queue = queue_block(child, queue, trace=trace)
            return output, output_queue
        elif tree.is_leaf:
            leaf_queue = Variable(-1 * torch.ones((self.queue_size, 1)))
            leaf = tree.encoded_value
            if trace:
                trace.output = leaf.tolist()
                trace.memory = leaf_queue.tolist()
            return leaf, leaf_queue
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

            lchild, _ = self(tree.lchild, trace=trace_item.lchild if trace else None)
            rchild, _ = self(tree.rchild, trace=trace_item.rchild if trace else None)


            if tree.is_numeric():
                assert (tree.lchild.is_a_floating_point and tree.rchild.function_name == NUMBER_DECODER) \
                    or (tree.rchild.is_a_floating_point and tree.lchild.function_name == NUMBER_DECODER)
                loss = (lchild - rchild) * (lchild - rchild)
                correct = math.isclose(lchild.item(), rchild.item(), rel_tol=1e-3)
            else:
                out = torch.cat((Variable(torch.FloatTensor([0])), torch.dot(lchild, rchild).unsqueeze(0) + self.bias), dim=0)
                loss = - F.log_softmax(out)[round(label.item())]
                correct = F.softmax(out)[round(label.item())].item() > 0.5

                if trace_item is not None:
                    trace_item.probability = F.softmax(out)[1].item()
                    trace_item.correct = correct
            assert isinstance(correct, bool)
            record.append({
                "ex": tree,
                "label": round(label.item()),
                "loss": loss.item(),
                "correct": correct,
                "depth": depth,
                "score": out[1].item() if not tree.is_numeric() else lchild.item() # WARNING: only works for symbolic data
            })
            total_loss += loss
        return record, total_loss / len(batch)

class BinaryQueueNNNode(torch.nn.Module):

    def __init__(self, queue_size, num_input):
        super().__init__()

        self.activation = F.sigmoid
        # TODO: add a no-action dimension
        self.action = nn.Linear(num_input * 2, 2, bias=True)
        self.input_linear = nn.Linear(num_input * 2, 1, bias=True)
        self.pop_linear = nn.Linear(2, 1, bias=True)
        self.push_linear = nn.Linear(2, 1, bias=True)
        self.queue_activations = F.sigmoid

        self.queue_size = queue_size

    def forward(self, input_left, input_right, queue_left, queue_right, trace=None):
        """
        Args:
            input_left: (num_input,)
            input_right: (num_output,)
            queue_left: (queue_size,)
            queue_right: (queue_size,)

        Returns:
            [(queue_size,)]
        """

        inp = torch.cat((input_left, input_right), dim=0) # (2*num_input,)
        tmp = self.action(inp)
        action =  F.softmax(tmp) # (2,)

        input_queue = torch.cat((queue_left, queue_right), dim=1) # (queue_size, 2)
        input_queue = torch.cat((input_queue, Variable(torch.FloatTensor([[-1,-1]]))), dim=0) # (queue_size + 1, 2)
        pop_indices = torch.LongTensor(range(1, self.queue_size+1))
        pop = torch.index_select(input_queue, 0, pop_indices)
        pop = self.pop_linear(pop)
        pop = self.queue_activations(pop) # (queue_size,)

        push_indices = torch.LongTensor(range(0, self.queue_size-1))
        push = torch.index_select(input_queue, 0, push_indices)
        push = self.push_linear(push)
        push_input = self.input_linear(inp).unsqueeze(0)
        push = torch.cat((push_input, push), dim=0)
        push = self.queue_activations(push) # (queue_size,)

        push_pop_cat = torch.cat((push, pop), dim=1) # (queue_size,2)
        queue = torch.matmul(push_pop_cat, action.unsqueeze(1)) # (queue_size, 1)

        if trace:
            trace.action = action.tolist()
            trace.memory = queue.tolist()
        return queue


class UnaryQueueNNNode(torch.nn.Module):

    def __init__(self, queue_size, num_input):
        super().__init__()

        self.activation = F.sigmoid
        # TODO: add a no-action dimension
        self.action = nn.Linear(num_input, 2, bias=True)
        self.input_linear = nn.Linear(num_input, 1, bias=True)
        self.deq_linear = nn.Linear(1, 1, bias=True)
        self.enq_linear = nn.Linear(1, 1, bias=True)
        self.queue_activations = F.sigmoid

        self.queue_size = queue_size


    def forward(self, inp, queue, trace=None):
        """
        Args:
            inp: (num_input,)
            queue: (queue_size,)

        Returns:
            (queue_size,)
        """

        tmp = self.action(inp)
        action =  F.softmax(tmp) # (2,)

        queue = torch.cat((queue, Variable(torch.FloatTensor([[-1]]))), dim=0) # (queue_size + 1, 1)
        deq_indices = torch.LongTensor(range(1, self.queue_size+1))
        deq = torch.index_select(queue, 0, deq_indices)
        deq = self.deq_linear(deq)
        deq = self.queue_activations(deq) # (queue_size,)

        enq_indices = torch.LongTensor(range(1, self.queue_size))
        enq = torch.index_select(queue, 0, enq_indices)
        enq = self.enq_linear(enq)
        enq_input = self.input_linear(inp).unsqueeze(0)
        enq = torch.cat((enq, enq_input), dim=0)
        enq = self.queue_activations(enq) # (queue_size,)

        enq_deq_cat = torch.cat((enq, deq), dim=1) # (queue_size,2)
        queue = torch.matmul(enq_deq_cat, action.unsqueeze(1)) # (queue_size, 1)

        if trace:
            trace.action = action.tolist()
            trace.memory = queue.tolist()
        return queue


class BinaryMemoryNNNode(torch.nn.Module):
    def __init__(self, memory_size, num_input, num_output, activation, memory_type='simple', top_k=5):
        super().__init__()

        self.linear = nn.Linear(num_input * 2, num_output)
        if memory_type == 'full_stack' or memory_type == 'full_stack_gated' or memory_type == "full_queue_gated":
            # if full_stack just take the top? we can also sum the top k, or learn an attention vector
            self.memory_linear = nn.Linear(num_input * 2, num_output)
        else:
            self.memory_linear = nn.Linear(top_k * 2, num_output)
        if activation == "sigmoid":
            self.activation = F.sigmoid
        elif activation == "tanh":
            self.activation = F.tanh
        else:
            raise ValueError("Unhandled activation: %s" % activation)
        self.top_k = top_k
        self.memory_type = memory_type

    def forward(self, input_left, input_right, memory_left, memory_right, trace=None, dropout=None):
        """
        Args:
            input_left: (num_input,)
            input_right: (num_output,)
            memory_left: (memory_size,)
            memory_right: (memory_size,)

        Returns:
            (num_output,)
        """
        inp = torch.cat((input_left, input_right), dim=0)
        if self.memory_type == 'full_stack' or self.memory_type == 'full_stack_gated' or self.memory_type == "full_queue_gated":
            memory = torch.cat((memory_left, memory_right), dim=1) # (stack_size, 2*num_input)
            memory = memory[0,:].reshape(-1) # (1,2*num_input) # TODO: currently choosing only the top. but consider choosing top-k after debugging
        else:
            # memory = torch.cat((memory_left.squeeze(1), memory_right.squeeze(1)),
            #                dim=0)
            memory = torch.cat((memory_left.squeeze(1)[:self.top_k], memory_right.squeeze(1)[:self.top_k]),
                           dim=0) # (top_k * 2, )

        tmp1 = self.linear(inp)
        tmp2 = self.memory_linear(memory)
        output = self.activation(tmp1 + tmp2)
        if dropout is not None:
            output = F.dropout(output,p=dropout,training=self.training)
        else:
            pass
        if trace:
            trace.output = output.tolist()
        return output

class UnaryMemoryNNNode(torch.nn.Module):

    def __init__(self, memory_size, num_input, num_output, activation, memory_type='simple', top_k=5):
        super().__init__()

        self.linear = nn.Linear(num_input, num_output)
        if memory_type == 'full_stack' or memory_type == 'full_stack_gated' or memory_type == "full_queue_gated":
            # if full_stack just take the top? we can also sum the top k, or learn an attention vector
            self.memory_linear = nn.Linear(num_input, num_output)
        else:
            self.memory_linear = nn.Linear(top_k, num_output)
        if activation == "sigmoid":
            self.activation = F.sigmoid
        elif activation == "tanh":
            self.activation = F.tanh
        else:
            raise ValueError("Unhandled activation: %s" % activation)
        self.top_k = top_k
        self.memory_type = memory_type

    def forward(self, inp, memory, trace=None, dropout=None):
        """
        Args:
            inp: (num_input,)
            memory: (memory_size,) or (memory_size, num_input)

        Returns:
            (num_output,) 
        """

        if self.memory_type == 'full_stack' or self.memory_type == 'full_stack_gated' or self.memory_type == 'full_queue_gated':
            memory = memory[0,:].unsqueeze(1) # (num_input, ) # TODO: currently choosing only the top. but consider choosing top-k after debugging
        else:
            memory = memory[:self.top_k] # (top_k, )

        tmp1 = self.linear(inp)
        tmp2 = self.memory_linear(memory.squeeze(1))
        output = self.activation(tmp1 + tmp2)
        if dropout is not None:
            output = F.dropout(output,p=dropout,training=self.training)
        else:
            pass
        if trace:
            trace.output = output.tolist()
        return output


class StackLSTMTrees(torch.nn.Module):
    def __init__(self, num_hidden, num_embed, memory_size, share_memory_params,
                 dropout, stack_node_activation,
                 no_op=False, stack_type='simple', top_k=5, verbose=False):
        super().__init__()

        for name in UNARY_FNS:
            setattr(self, name, UnaryMemoryLSTMNode(memory_size=memory_size,
                                                     num_input=num_hidden,
                                                     num_hidden=num_hidden,
                                                     activation=stack_node_activation,
                                                     memory_type=stack_type,
                                                     top_k=top_k))

        for name in BINARY_FNS:
            setattr(self, name, BinaryMemoryLSTMNode(memory_size=memory_size,
                                                     num_input=num_hidden,
                                                     num_hidden=num_hidden,
                                                     activation=stack_node_activation,
                                                     memory_type=stack_type,
                                                     top_k=top_k))

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


        if stack_type == 'simple':

            self.binary_stack_h = BinaryStackNNNode(stack_size=memory_size,
                                                  num_input=num_hidden,
                                                  activation=stack_node_activation,
                                                  no_op=no_op, verbose=verbose)
            self.binary_stack_c = BinaryStackNNNode(stack_size=memory_size,
                                                  num_input=num_hidden,
                                                  activation=stack_node_activation,
                                                  no_op=no_op, verbose=verbose)

            self.unary_stack_h = UnaryStackNNNode(stack_size=memory_size,
                                                num_input=num_hidden,
                                                activation=stack_node_activation,
                                                no_op=no_op, verbose=verbose)
            self.unary_stack_c = UnaryStackNNNode(stack_size=memory_size,
                                                num_input=num_hidden,
                                                activation=stack_node_activation,
                                                no_op=no_op, verbose=verbose)

        self.bias = nn.Parameter(torch.FloatTensor([0]))
        self.num_hidden = num_hidden

        self.stack_size = memory_size
        self.stack_type = stack_type
        self.verbose = verbose

    def forward(self, tree, trace=None):
        if not tree.is_leaf and tree.function_name not in self._modules:
            raise AssertionError("Unknown functional node: %s" % tree.function_name)
        if tree.is_binary:
            hl, hlm, cl, clm = self(tree.lchild, trace=trace.lchild if trace else None)
            hr, hrm, cr, crm = self(tree.rchild, trace=trace.rchild if trace else None)
            nn_block = getattr(self, tree.function_name)
            stack_block_h = self.binary_stack_h
            stack_block_c = self.binary_stack_c
            output_h, output_c = nn_block((hl, hlm, cl, clm), (hr, hrm, cr, crm), trace=trace)
            output_h_stack = stack_block_h(hl, hr, hlm, hrm, trace=trace)
            output_c_stack = stack_block_c(cl, cr, clm, crm, trace=trace)
            return output_h, output_h_stack, output_c, output_c_stack
        elif tree.is_unary:
            hl, hlm, cl, clm = self(tree.lchild, trace=trace.lchild if trace else None)
            nn_block = getattr(self, tree.function_name)
            if tree.function_name in {NUMBER_DECODER, NUMBER_ENCODER,
                                      SYMBOL_ENCODER}:
                # TODO: do we need to input hm here? how do we get stack here?
                output_h = nn_block(hl)
                output_c = cl
                output_h_stack = hlm
                output_c_stack = clm
            else:
                stack_block_h = self.unary_stack_h
                stack_block_c = self.unary_stack_c
                output_h, output_c = nn_block((hl, hlm, cl, clm), trace=trace)
                output_h_stack = stack_block_h(hl, hlm, trace=trace)
                output_c_stack = stack_block_c(cl, clm, trace=trace)
            return output_h, output_h_stack, output_c, output_c_stack
        elif tree.is_leaf:
            output_c = Variable(torch.FloatTensor([0] * self.num_hidden))
            output_h = tree.encoded_value
            output_c_stack = torch.zeros((self.stack_size,1), requires_grad=True)
            output_h_stack = torch.zeros((self.stack_size,1), requires_grad=True)
            if trace:
                trace.output = tree.encoded_value.tolist()
                trace.memory = c.tolist()
            return output_h, output_h_stack, output_c, output_c_stack
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

            lchild, _, _, _ = self(tree.lchild, trace=trace_item.lchild if trace else None)
            rchild, _, _, _ = self(tree.rchild, trace=trace_item.rchild if trace else None)

            if tree.is_numeric():
                assert (tree.lchild.is_a_floating_point and tree.rchild.function_name == NUMBER_DECODER) \
                    or (tree.rchild.is_a_floating_point and tree.lchild.function_name == NUMBER_DECODER)
                loss = (lchild - rchild) * (lchild - rchild)
                correct = math.isclose(lchild.item(), rchild.item(), rel_tol=1e-3)
            else:
                out = torch.cat((Variable(torch.FloatTensor([0])), torch.dot(lchild, rchild).unsqueeze(0) + self.bias), dim=0)
                loss = - F.log_softmax(out)[round(label.item())]
                correct = F.softmax(out)[round(label.item())].item() > 0.5

                if trace_item is not None:
                    trace_item.probability = F.softmax(out)[1].item()
            assert isinstance(correct, bool)
            record.append({
                "ex": tree,
                "label": round(label.item()),
                "loss": loss.item(),
                "correct": correct,
                "depth": depth,
                "score": out[1].item() if not tree.is_numeric() else lchild.item() # WARNING: only works for symbolic data
            })
            total_loss += loss
        return record, total_loss / len(batch)


class BinaryMemoryLSTMNode(torch.nn.Module):

    def __init__(self, memory_size, num_input, num_hidden, activation, memory_type='simple', top_k=5):
        super().__init__()
        self.data_left = nn.Linear(num_input, num_hidden, bias=False)
        self.data_right = nn.Linear(num_input, num_hidden, bias=False)
        self.data_bias = nn.Parameter(torch.FloatTensor([0] * num_hidden))
        self.forget_left_by_left = nn.Linear(num_input, num_hidden, bias=False)
        self.forget_left_by_right = nn.Linear(num_input, num_hidden, bias=False)
        self.forget_right_by_left = nn.Linear(num_input, num_hidden, bias=False)
        self.forget_right_by_right = nn.Linear(num_input, num_hidden, bias=False)
        self.forget_bias_left = nn.Parameter(torch.FloatTensor([0] * num_hidden))
        self.forget_bias_right = nn.Parameter(torch.FloatTensor([0] * num_hidden))
        self.output_left = nn.Linear(num_input, num_hidden, bias=False)
        self.output_right = nn.Linear(num_input, num_hidden, bias=False)
        self.output_bias = nn.Parameter(torch.FloatTensor([0] * num_hidden))
        self.input_left = nn.Linear(num_input, num_hidden, bias=False)
        self.input_right = nn.Linear(num_input, num_hidden, bias=False)
        self.input_bias = nn.Parameter(torch.FloatTensor([0] * num_hidden))

        self.c_memory_linear = nn.Linear(top_k * 2, num_hidden)
        self.h_memory_linear = nn.Linear(top_k * 2, num_hidden)

        if activation == "sigmoid":
            self.activation = F.sigmoid
        elif activation == "tanh":
            self.activation = F.tanh
        else:
            raise ValueError("Unhandled activation: %s" % activation)
        self.top_k = top_k
        self.memory_type = memory_type



    def forward(self, input_left, input_right, trace=None):
        """

        Args:
            input_left: ((num_hidden,), (num_stack,), (num_hidden,), (num_stack,))
            input_right: ((num_hidden,), (num_stack,), (num_hidden,), (num_stack,))

        Returns:
            (num_hidden,), (num_hidden)
        """
        hl, hlm, cl, clm = input_left
        hr, hrm, cr, crm = input_right

        hm = torch.cat((hlm.squeeze(1)[:self.top_k] ,hrm.squeeze(1)[:self.top_k]), dim=0) # (stack_size, 2)
        cm = torch.cat((clm.squeeze(1)[:self.top_k] ,crm.squeeze(1)[:self.top_k]), dim=0) # (stack_size, 2)
        hm_out = self.h_memory_linear(hm)
        hm_out = self.activation(hm_out)
        cm_out = self.c_memory_linear(cm)
        cm_out = self.activation(cm_out)

        i = F.sigmoid(self.data_left(hl) + self.data_right(hr) + self.data_bias)
        f_left = F.sigmoid(self.forget_left_by_left(hl) +
                           self.forget_left_by_right(
                               hr) + self.forget_bias_left)
        f_right = F.sigmoid(self.forget_right_by_left(hl) +
                           self.forget_right_by_right(
                               hr) + self.forget_bias_right)
        o = F.sigmoid(self.output_left(hl) + self.output_right(hr) + self.output_bias)
        u = F.tanh(self.input_left(hl) + self.input_right(hr) + self.input_bias)

        c = i * u + f_left * cl + f_right * cr + cm_out
        h = o * F.tanh(c) + hm_out

        return h, c


class UnaryMemoryLSTMNode(torch.nn.Module):
    def __init__(self, memory_size, num_input, num_hidden, activation, memory_type='simple', top_k=5):
        super().__init__()
        self.data = nn.Linear(num_input, num_hidden, bias=True)
        self.forget = nn.Linear(num_input, num_hidden, bias=True)
        self.output = nn.Linear(num_input, num_hidden, bias=True)
        self.input = nn.Linear(num_input, num_hidden, bias=True)

        self.c_memory_linear = nn.Linear(top_k, num_hidden)
        self.h_memory_linear = nn.Linear(top_k, num_hidden)

        if activation == "sigmoid":
            self.activation = F.sigmoid
        elif activation == "tanh":
            self.activation = F.tanh
        else:
            raise ValueError("Unhandled activation: %s" % activation)
        self.top_k = top_k
        self.memory_type = memory_type

    def forward(self, inp, trace=None):
        """

        Args:
            inp: ((num_hidden,), (num_stack,), (num_hidden,), (num_stack,))

        Returns:
            (num_hidden,), (num_hidden)
        """
        h, hm, c, cm = inp

        hm = hm.squeeze(1)[:self.top_k]
        hm_out = self.h_memory_linear(hm)
        hm_out = self.activation(hm_out)
        cm = cm.squeeze(1)[:self.top_k]
        cm_out = self.c_memory_linear(cm)
        cm_out = self.activation(cm_out)

        i = F.sigmoid(self.data(h))
        f = F.sigmoid(self.forget(h))
        o = F.sigmoid(self.output(h))
        u = F.tanh(self.input(h))
        c = i * u + f * c + cm_out
        h = o * F.tanh(c) + hm_out
        return h, c



class BinaryStackLSTMNode(torch.nn.Module):

    def __init__(self, stack_size, num_input, activation, no_op=False, verbose=False):
        super().__init__()

        if activation == "sigmoid":
            self.stack_activations = F.sigmoid
        elif activation == "tanh":
            self.stack_activations = F.tanh
        else:
            raise ValueError("Unhandled activation: %s" % activation)
        if no_op:
            self.action_c = nn.Linear(num_input * 2, 3, bias=True)
            self.action_h = nn.Linear(num_input * 2, 3, bias=True)
            self.no_op_linear_c = nn.Linear(2, 1, bias=True)
            self.no_op_linear_h = nn.Linear(2, 1, bias=True)
        else:
            self.action_c = nn.Linear(num_input * 2, 2, bias=True)
            self.action_h = nn.Linear(num_input * 2, 2, bias=True)

        self.input_linear_c = nn.Linear(num_input * 2, 1, bias=True)
        self.pop_linear_c = nn.Linear(2, 1, bias=True)
        self.push_linear_c = nn.Linear(2, 1, bias=True)

        self.input_linear_h = nn.Linear(num_input * 2, 1, bias=True)
        self.pop_linear_h = nn.Linear(2, 1, bias=True)
        self.push_linear_h = nn.Linear(2, 1, bias=True)

        self.stack_size = stack_size
        self.no_op = no_op
        self.verbose = verbose

    def forward(self, input_left, input_right, trace=None):
        """
        Args:
            input_left_h: (num_input,)
            input_right_h: (num_output,)
            input_left_c: (num_input,)
            input_right_c: (num_output,)
            stack_left_h: (stack_size,)
            stack_right_h: (stack_size,)
            stack_left_c: (stack_size,)
            stack_right_c: (stack_size,)

        Returns:
            [(stack_size,)]
            [(stack_size,)]
        """

        hl, hlm, cl, clm = input_left
        hr, hrm, cr, crm = input_right

        inp_h = torch.cat((hl, hr), dim=0) # (2*num_input,)
        tmp = self.action(inp_h)
        action_h = F.softmax(tmp) # (2,) or (3,)
        inp_c = torch.cat((cl, cr), dim=0) # (2*num_input,)
        tmp = self.action(inp_c)
        action_c = F.softmax(tmp) # (2,) or (3,)
        if self.verbose:
            print('action_h at node is {0}'.format(action_h))
            print('action_c at node is {0}'.format(action_c))

        pop_indices = torch.LongTensor(range(1, self.stack_size+1))
        stack_h = torch.cat((hlm, hrm), dim=1) # (stack_size, 2)
        stack_h = torch.cat((stack_h, Variable(torch.FloatTensor([[0,0]]))), dim=0) # (stack_size + 1, 2)
        pop_h = torch.index_select(stack_h, 0, pop_indices)
        pop_h = self.pop_linear_h(pop_h)
        pop_h = self.stack_activations(pop_h) # (stack_size,)
        stack_c = torch.cat((clm, crm), dim=1) # (stack_size, 2)
        stack_c = torch.cat((stack_c, Variable(torch.FloatTensor([[0,0]]))), dim=0) # (stack_size + 1, 2)
        pop_c = torch.index_select(stack_c, 0, pop_indices)
        pop_c = self.pop_linear_c(pop_c)
        pop_c = self.stack_activations(pop_c) # (stack_size,)

        push_indices = torch.LongTensor(range(0, self.stack_size-1))
        push_h = torch.index_select(stack_h, 0, push_indices)
        push_h = self.push_linear_h(push_h)
        push_input_h = self.input_linear_h(inp_h).unsqueeze(0)
        push_h = torch.cat((push_input_h, push_h), dim=0)
        push_h = self.stack_activations(push_h) # (stack_size,)
        push_c = torch.index_select(stack_c, 0, push_indices)
        push_c = self.push_linear_c(push_c)
        push_input_c = self.input_linear_c(inp_c).unsqueeze(0)
        push_c = torch.cat((push_input_c, push_c), dim=0)
        push_c = self.stack_activations(push_c) # (stack_size,)

        if self.no_op:
            no_op_indices = torch.LongTensor(range(0, self.stack_size))
            no_op_h = torch.index_select(stack_h, 0, no_op_indices) # (stack_size,2)
            no_op_h = self.no_op_linear_h(no_op_h) # (stack_size,)
            no_op_h = self.stack_activations(no_op_h)
            push_pop_cat_h = torch.cat((push_h, pop_h, no_op_h), dim=1) # (stack_size,3)

            no_op_c = torch.index_select(stack_c, 0, no_op_indices) # (stack_size,2)
            no_op_c = self.no_op_linear_c(no_op_c) # (stack_size,)
            no_op_c = self.stack_activations(no_op_c)
            push_pop_cat_c = torch.cat((push_c, pop_c, no_op_c), dim=1) # (stack_size,3)
        else:
            push_pop_cat_h = torch.cat((push_h, pop_h), dim=1) # (stack_size,2)
            push_pop_cat_c = torch.cat((push_c, pop_c), dim=1) # (stack_size,2)

        stack_h = torch.matmul(push_pop_cat_h, action_h.unsqueeze(1)) # (stack_size, 1)
        stack_c = torch.matmul(push_pop_cat_c, action_c.unsqueeze(1)) # (stack_size, 1)

        if trace:
            trace.action = action.tolist()
            trace.memory = stack.tolist()
        return stack_h, stack_c


class UnaryStackLSTMNode(torch.nn.Module):

    def __init__(self, stack_size, num_input, activation, no_op=False, verbose=False):
        super().__init__()

        if activation == "sigmoid":
            self.stack_activations = F.sigmoid
        elif activation == "tanh":
            self.stack_activations = F.tanh
        else:
            raise ValueError("Unhandled activation: %s" % activation)
        if no_op:
            self.action_h = nn.Linear(num_input, 3, bias=True)
            self.action_c = nn.Linear(num_input, 3, bias=True)
            self.no_op_linear_h = nn.Linear(1, 1, bias=True)
            self.no_op_linear_c = nn.Linear(1, 1, bias=True)
        else:
            self.action_h = nn.Linear(num_input, 2, bias=True)
            self.action_c = nn.Linear(num_input, 2, bias=True)
        
        self.input_linear_h = nn.Linear(num_input, 1, bias=True)
        self.pop_linear_h = nn.Linear(1, 1, bias=True)
        self.push_linear_h = nn.Linear(1, 1, bias=True)
        
        self.input_linear_c = nn.Linear(num_input, 1, bias=True)
        self.pop_linear_c = nn.Linear(1, 1, bias=True)
        self.push_linear_c = nn.Linear(1, 1, bias=True)

        self.stack_size = stack_size
        self.no_op = no_op
        self.verbose = verbose


    def forward(self, inp, trace=None):
        """
        Args:
            inp_c: (num_input,)
            inp_h: (num_input,)
            stack_c: (stack_size,)
            stack_h: (stack_size,)

        Returns:
            (stack_size,)
            (stack_size,)
        """

        inp_h, stack_h, inp_c, stack_c = inp

        tmp = self.action_h(inp_h)
        action_h =  F.softmax(tmp) # (2,)
        tmp = self.action_c(inp_c)
        action_c =  F.softmax(tmp) # (2,)
        if self.verbose:
            print('action_h at node is {0}'.format(action_h))
            print('action_c at node is {0}'.format(action_c))

        pop_indices = torch.LongTensor(range(1, self.stack_size+1))
        stack_h = torch.cat((stack_h, Variable(torch.FloatTensor([[0]]))), dim=0) # (stack_size + 1, 1)
        pop_h = torch.index_select(stack_h, 0, pop_indices)
        pop_h = self.pop_linear_h(pop_h)
        pop_h = self.stack_activations(pop_h) # (stack_size,)
        stack_c = torch.cat((stack_c, Variable(torch.FloatTensor([[0]]))), dim=0) # (stack_size + 1, 1)
        pop_c = torch.index_select(stack_c, 0, pop_indices)
        pop_c = self.pop_linear_c(pop_c)
        pop_c = self.stack_activations(pop_c) # (stack_size,)

        push_indices = torch.LongTensor(range(0, self.stack_size-1))
        push_h = torch.index_select(stack_h, 0, push_indices)
        push_h = self.push_linear_h(push_h)
        push_input_h = self.input_linear_h(inp_h).unsqueeze(0)
        push_h = torch.cat((push_input_h, push_h), dim=0)
        push_h = self.stack_activations(push_h) # (stack_size,)
        push_c = torch.index_select(stack_c, 0, push_indices)
        push_c = self.push_linear_c(push_c)
        push_input_c = self.input_linear_c(inp_c).unsqueeze(0)
        push_c = torch.cat((push_input_c, push_c), dim=0)
        push_c = self.stack_activations(push_c) # (stack_size,)

        if self.no_op:
            no_op_indices = torch.LongTensor(range(0, self.stack_size))
            no_op_h = torch.index_select(stack_h, 0, no_op_indices) # (stack_size,)
            no_op_h = self.no_op_linear_h(no_op_h) # (stack_size,)
            no_op_h = self.stack_activations(no_op_h)
            push_pop_cat_h = torch.cat((push_h, pop_h, no_op_h), dim=1) # (stack_size,3)
            no_op_c = torch.index_select(stack_c, 0, no_op_indices) # (stack_size,)
            no_op_c = self.no_op_linear_c(no_op_c) # (stack_size,)
            no_op_c = self.stack_activations(no_op_c)
            push_pop_cat_c = torch.cat((push_c, pop_c, no_op_c), dim=1) # (stack_size,3)
        else:
            push_pop_cat_h = torch.cat((push_h, pop_h), dim=1) # (stack_size,2)
            push_pop_cat_c = torch.cat((push_c, pop_c), dim=1) # (stack_size,2)

        stack_h = torch.matmul(push_pop_cat_h, action_h.unsqueeze(1)) # (stack_size, 1)
        stack_c = torch.matmul(push_pop_cat_c, action_c.unsqueeze(1)) # (stack_size, 1)

        if trace:
            trace.action = action.tolist()
            trace.memory = stack.tolist()
        return stack_h, stack_c




class StackNNTreesMem2out(torch.nn.Module):

    def __init__(self, num_hidden, num_embed, memory_size, \
                 dropout, tree_node_activation, stack_node_activation,
                 no_op=False, stack_type='stack', top_k=5, verbose=False, 
                 no_pop=False, likeLSTM=False, gate_push_pop=False, 
                 gate_top_k=False, normalize_action=False):
        super().__init__()

        for name in UNARY_FNS:

            setattr(self, name, UnaryMem2outNNNode(memory_size=memory_size,
                                                  num_input=num_hidden,
                                                  num_output=num_hidden,
                                                  activation=tree_node_activation,
                                                  top_k=top_k,
                                                  gate_top_k=gate_top_k))

            if stack_type == 'stack':
                setattr(self, name+'_stack', UnaryFullStackNNNodeGated(stack_size=memory_size,
                                                num_input=num_hidden,
                                                activation=stack_node_activation,
                                                no_op=no_op, verbose=verbose,
                                                no_pop=no_pop,
                                                likeLSTM=likeLSTM,
                                                gate_push_pop=gate_push_pop,
                                                normalize_action=normalize_action))

            elif stack_type == 'queue':
                setattr(self, name+'_stack', UnaryFullQueueNNNodeGated(stack_size=memory_size,
                                                num_input=num_hidden,
                                                activation=stack_node_activation,
                                                no_op=no_op, verbose=verbose)) # TODO: change name to queue later and case in forward
            else:
                raise AssertionError('unknown stack_type, choose one of these: "stack", "queue"')

        for name in BINARY_FNS:
            setattr(self, name, BinaryMem2outNNNode(memory_size=memory_size,
                                                  num_input=num_hidden,
                                                  num_output=num_hidden,
                                                  activation=tree_node_activation,
                                                  top_k=top_k,
                                                  gate_top_k=gate_top_k))

            if stack_type == 'stack':

                setattr(self, name+'_stack', BinaryFullStackNNNodeGated(stack_size=memory_size,
                                                  num_input=num_hidden,
                                                  activation=stack_node_activation,
                                                  no_op=no_op, verbose=verbose,
                                                  no_pop=no_pop,
                                                  likeLSTM=likeLSTM,
                                                  gate_push_pop=gate_push_pop,
                                                  normalize_action=normalize_action))
            elif stack_type == 'queue':
                setattr(self, name+'_stack', BinaryFullQueueNNNodeGated(stack_size=memory_size,
                                                  num_input=num_hidden,
                                                  activation=stack_node_activation,
                                                  no_op=no_op, verbose=verbose)) # TODO: change name to queue later and case in forward


        setattr(self, NUMBER_ENCODER, nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(1, num_hidden)),
            ('relu1', nn.Sigmoid()),
            ('linear2', nn.Linear(num_hidden, num_hidden)),
            ('relu2', nn.Sigmoid())
        ])))
        setattr(self, NUMBER_DECODER, nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(num_hidden, num_hidden)),
            ('relu1', nn.Sigmoid()),
            ('linear2', nn.Linear(num_hidden, 1)),
        ])))
        setattr(self, SYMBOL_ENCODER, nn.Embedding(num_embeddings=len(VOCAB),
                                   embedding_dim=num_hidden))


        self.stack_size = memory_size
        self.num_hidden = num_hidden
        self.verbose = verbose
        self.stack_type = stack_type

        self.bias = nn.Parameter(torch.FloatTensor([0]))
        self.dropout = dropout

    def forward(self, tree, trace=None):
        if not tree.is_leaf and tree.function_name not in self._modules:
            raise AssertionError("Unknown functional node: %s" % tree.function_name)
        if tree.is_binary:
            if self.verbose:
                print('evaluating left child {0} and right child {1} and root node {2}:'\
                      .format(tree.lchild.function_name, tree.rchild.function_name, tree.function_name))
            lchild, lstack = self(tree.lchild, trace=trace.lchild if trace else None)
            rchild, rstack = self(tree.rchild, trace=trace.rchild if trace else None)
            nn_block = getattr(self, tree.function_name)
            stack_block = getattr(self, tree.function_name+'_stack')
            output_stack = stack_block(lchild, rchild, lstack, rstack, trace=trace, dropout=self.dropout)
            output = nn_block(lchild, rchild, output_stack, trace=trace)
            if self.verbose:
                print('output at node {0} is {1}'.format(tree.function_name, output))
                print('output stack at node {0} is {1}'.format(tree.function_name, output_stack))
            return output, output_stack
        elif tree.is_unary:
            if self.verbose:
                print('evaluating child {0} and root node {1}:'\
                      .format(tree.lchild.function_name, tree.function_name))
            child, stack = self(tree.lchild, trace=trace.lchild if trace else None)
            nn_block = getattr(self, tree.function_name)
            if tree.function_name in {NUMBER_DECODER, NUMBER_ENCODER, SYMBOL_ENCODER}:
                output = nn_block(child)
                output_stack = stack
                if trace:
                    trace.output = output.tolist()
                    trace.memory = stack.tolist()
            else:
                stack_block = getattr(self, tree.function_name + '_stack')
                output_stack = stack_block(child, stack, trace=trace, dropout=self.dropout)
                output = nn_block(child, output_stack, trace=trace)

            if self.verbose:
                print('output at node {0} is {1}'.format(tree.function_name, output))
                print('output stack at node {0} is {1}'.format(tree.function_name, output_stack))
            return output, output_stack
        elif tree.is_leaf:
            if self.verbose:
                print('evaluating leaf node {0}:'.format(tree.function_name))
            leaf_stack = torch.zeros((self.stack_size,self.num_hidden), requires_grad=True)
            leaf = tree.encoded_value
            if self.verbose:
                print('output at leaf node {0} is {1}'.format(tree.function_name, leaf))
                print('output stack at leaf node {0} is {1}'.format(tree.function_name, leaf_stack))
            if trace:
                trace.output = leaf.tolist()
                trace.memory = leaf_stack.tolist()
            return leaf, leaf_stack
        else:
            raise RuntimeError("Invalid tree:\n%s" % repr(self))


    def compute_batch(self, batch, trace=None):
        record = []
        total_loss = 0
        for tree, label, depth in batch:
            if self.verbose:
                print('STARTING NEW EQUATION')
                print('EQUATION: {0}'.format(tree.pretty_str()))
                if tree.pretty_str() == 'Integer_3^Integer_0 * Symbol_var_0 = Integer_4 + asec(Integer_0)^Integer_0':
                    print('RAW: {0}'.format(tree.raw['equation']['func']))
                    input('enter')
                print('LABEL: {0}'.format(label))

            if trace is not None:
                trace_item = eval(repr(tree))
                trace.append(trace_item)
            else:
                trace_item = None

            lchild, _ = self(tree.lchild, trace=trace_item.lchild if trace else None)
            rchild, _ = self(tree.rchild, trace=trace_item.rchild if trace else None)
            # thought, do we need to look at the stack at the root?

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
                "score": out[1].item() if not tree.is_numeric() else lchild.item() # WARNING: only works for symbolic data
            })
            total_loss += loss
        return record, total_loss / len(batch)





class BinaryMem2outNNNode(torch.nn.Module):
    def __init__(self, memory_size, num_input, num_output, activation, top_k=5, gate_top_k=False):
        super().__init__()

        if top_k == 1 and gate_top_k:
            raise AssertionError('gating top-k is only supported with top_k>1')

        self.o_linear = nn.Linear(num_input * 2, num_output, bias=True)

        if activation == "sigmoid":
            self.activation = F.sigmoid
        elif activation == "tanh":
            self.activation = F.tanh
        else:
            raise ValueError("Unhandled activation: %s" % activation)
        
        self.top_k = top_k
        self.gate_top_k = gate_top_k
        self.num_input = num_input

        if top_k > 1:
            if self.gate_top_k == False:
                self.pos_linear = nn.Linear(num_input * 2, top_k, bias=True)
            else:
                for k in range(1,self.top_k+1):
                    setattr(self, 'pos_linear_{0}'.format(k), 
                            nn.Linear(num_input*2, num_input, bias=True))

    def forward(self, input_left, input_right, node_memory, trace=None):
        """
        Args:
            input_left: (num_input,)
            input_right: (num_output,)
            memory_left: (memory_size,)
            memory_right: (memory_size,)

        Returns:
            (num_output,)
        """

        inp = torch.cat((input_left, input_right), dim=0)
        o = self.o_linear(inp)
        o = F.sigmoid(o) # (num_output,)

        if self.top_k > 1: 
            if self.gate_top_k==False:
                pos_gate = self.pos_linear(inp)
                pos_gate = F.sigmoid(pos_gate) # (topk,)
                top_indices = torch.LongTensor(range(0, self.top_k))
                memory = torch.index_select(node_memory, 0, top_indices) # (top_k, num_input)
                memory = torch.matmul(pos_gate.unsqueeze(0), memory).reshape(-1) # (num_output,)
            else:
                memory = torch.zeros((1,self.num_input), dtype=torch.float32, requires_grad=False)
                for k in range(1,self.top_k+1):
                    # IS THIS GOING TO BE REALLY SLOW?
                    nn_block = getattr(self, 'pos_linear_{0}'.format(k))
                    pos_gate = nn_block(inp)
                    pos_gate = F.sigmoid(pos_gate)
                    mem_row = node_memory[k-1,:]
                    memory = memory + pos_gate * mem_row
                memory = memory.reshape(-1)
        else:
            memory = node_memory[0,:].reshape(-1) # (num_output,) 

        output = self.activation(memory)
        output = o * output

        if trace:
            trace.output = output.tolist()
        return output

class UnaryMem2outNNNode(torch.nn.Module):

    def __init__(self, memory_size, num_input, num_output, activation, top_k=5, gate_top_k=False):
        super().__init__()

        if top_k == 1 and gate_top_k:
            raise AssertionError('gating top-k is only supported with top_k>1')
        
        self.o_linear = nn.Linear(num_input, num_output, bias=True)

        if activation == "sigmoid":
            self.activation = F.sigmoid
        elif activation == "tanh":
            self.activation = F.tanh
        else:
            raise ValueError("Unhandled activation: %s" % activation)

        self.top_k = top_k
        self.gate_top_k = gate_top_k
        self.num_input = num_input

        if self.top_k > 1 :
            if self.gate_top_k==False:
                self.pos_linear = nn.Linear(num_input, top_k, bias=True)
            else:
                for k in range(1,self.top_k+1):
                    setattr(self, 'pos_linear_{0}'.format(k), 
                            nn.Linear(num_input, num_input, bias=True))


    def forward(self, inp, node_memory, trace=None):
        """
        Args:
            inp: (num_input,)
            memory: (memory_size,) or (memory_size, num_input)

        Returns:
            (num_output,) 
        """

        o = self.o_linear(inp)
        o = F.sigmoid(o) # (1,)

        if self.top_k > 1: 
            if self.gate_top_k==False:
                pos_gate = self.pos_linear(inp)
                pos_gate = F.sigmoid(pos_gate) # (topk,)
                top_indices = torch.LongTensor(range(0, self.top_k))
                memory = torch.index_select(node_memory, 0, top_indices) # (top_k, num_input)
                memory = torch.matmul(pos_gate.unsqueeze(0), memory).reshape(-1) # (num_output,)
            else:
                memory = torch.zeros((1,self.num_input), dtype=torch.float32, requires_grad=False)
                for k in range(1,self.top_k+1):
                    # IS THIS GOING TO BE REALLY SLOW?
                    nn_block = getattr(self, 'pos_linear_{0}'.format(k))
                    pos_gate = nn_block(inp)
                    pos_gate = F.sigmoid(pos_gate)
                    mem_row = node_memory[k-1,:]
                    memory = memory + pos_gate * mem_row
                memory = memory.reshape(-1)

        else:
            memory = node_memory[0,:].reshape(-1) # (num_input,) # TODO: consider getting the top-k

        output = self.activation(memory)
        output = o * output

        if trace:
            trace.output = output.tolist()
        return output


class BinaryFullQueueNNNodeGated(torch.nn.Module):

    def __init__(self, stack_size, num_input, activation, no_op=False, verbose=False):
        super().__init__()


        if activation == "sigmoid":
            self.stack_activations = F.sigmoid
        elif activation == "tanh":
            self.stack_activations = F.tanh
        else:
            raise ValueError("Unhandled activation: %s" % activation)
        
        if no_op:
            self.action = nn.Linear(num_input * 2, 3, bias=True)
        else:
            self.action = nn.Linear(num_input * 2, 2, bias=True)

        self.gate_linear_l = nn.Linear(num_input*2, num_input, bias=True)
        self.gate_linear_r = nn.Linear(num_input*2, num_input, bias=True)
        self.input_linear = nn.Linear(num_input*2, num_input, bias=True)

        self.stack_size = stack_size
        self.num_input  = num_input
        self.no_op      = no_op
        self.verbose = verbose


    def forward(self, input_left, input_right, stack_left, stack_right, trace=None):
        """
        Args:
            input_left: (num_input,)
            input_right: (num_output,)
            stack_left: (stack_size,num_input)
            stack_right: (stack_size,num_input)

        Returns:
            [(stack_size,num_input)]
        """

        inp = torch.cat((input_left, input_right), dim=0) # (2*num_input,)
        tmp = self.action(inp)
        action = F.softmax(tmp) # (2,) or (3,)
        if self.verbose:
            print('action at node is {0}'.format(action))

        left_gate = self.gate_linear_l(inp)
        left_gate = F.sigmoid(left_gate) # (1,)
        right_gate = self.gate_linear_r(inp)
        right_gate = F.sigmoid(right_gate) # (1,)

        input_stack = left_gate * stack_left + right_gate * stack_right # (stack_size, num_input)


        push_input = self.input_linear(inp).unsqueeze(0) # (1,num_input)
        push_input = self.stack_activations(push_input)

        queue_indices = torch.LongTensor(range(1, self.stack_size))
        queue = torch.index_select(input_stack, 0, queue_indices)
        push = torch.cat((queue, push_input), dim=0)
        pop = torch.cat((queue, torch.zeros((1,self.num_input), dtype=torch.float32, requires_grad=True)), dim=0)


        if self.no_op:
            no_op_indices = torch.LongTensor(range(0, self.stack_size))
            no_op = torch.index_select(input_stack, 0, no_op_indices)  # (stack_size, num_input)
            stack = action[0] * push + action[1] * pop  + action[2] * no_op
        else:
            push_pop_cat = torch.cat((push, pop), dim=1)  # (stack_size, 2*num_input)
            stack = action[0] * push + action[1] * pop

        if trace:
            trace.action = action.tolist()
            trace.memory = stack.tolist()
        return stack


class UnaryFullQueueNNNodeGated(torch.nn.Module):

    def __init__(self, stack_size, num_input, activation, no_op=False, verbose=False):
        super().__init__()

        if activation == "sigmoid":
            self.stack_activations = F.sigmoid
        elif activation == "tanh":
            self.stack_activations = F.tanh
        else:
            raise ValueError("Unhandled activation: %s" % activation)

        if no_op:
            self.action = nn.Linear(num_input, 3, bias=True)
        else:
            self.action = nn.Linear(num_input, 2, bias=True)

        self.gate_linear = nn.Linear(num_input, num_input, bias=True)
        self.input_linear = nn.Linear(num_input, num_input, bias=True)

        self.stack_size = stack_size
        self.num_input  = num_input
        self.no_op      = no_op
        self.verbose = verbose


    def forward(self, inp, stack, trace=None):
        """
        Args:
            inp: (num_input,)
            stack: (stack_size,num_input)

        Returns:
            (stack_size, num_input)
        """

        tmp = self.action(inp)
        action =  F.softmax(tmp) # (2,)
        if self.verbose:
            print('action at node is {0}'.format(action))

        gate = self.gate_linear(inp)
        gate = F.sigmoid(gate) # (1,)
        stack = gate * stack

        push_input = self.input_linear(inp).unsqueeze(0) # (1,num_input)
        push_input = self.stack_activations(push_input)

        queue_indices = torch.LongTensor(range(1, self.stack_size))
        queue = torch.index_select(stack, 0, queue_indices)
        push = torch.cat((queue, push_input), dim=0)
        pop = torch.cat((queue, torch.zeros((1,self.num_input), dtype=torch.float32, requires_grad=True)), dim=0)

        if self.no_op:
            no_op_indices = torch.LongTensor(range(0, self.stack_size))
            no_op = torch.index_select(stack, 0, no_op_indices) # (stack_size,num_input)
            stack = action[0] * push + action[1] * pop  + action[2] * no_op
        else:
            push_pop_cat = torch.cat((push, pop), dim=1) # (stack_size,2)
            stack = action[0] * push + action[1] * pop

        if trace:
            trace.action = action.tolist()
            trace.memory = stack.tolist()
        return stack

