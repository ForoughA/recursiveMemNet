import argparse
import pickle
import traceback

import torch
import random
import json

from equation_verification.dataset_loading import load_equation_tree_examples, \
    load_single_equation_tree_example, sequential_sampler, \
    build_equation_tree_examples_list
from equation_verification.nn_tree_model import build_nnTree
from optimizers import build_optimizer
from parse import parse_equation
from equation_verification.constants import UNARY_FNS


class nnTreeEquationVerificationExperiment:

    def __init__(self):
        """
        Defines and parses hyper-parameters, conditions of the experiment,
        logging and save/loading options from the commandline.
        """
        parser = argparse.ArgumentParser(
            description="Train tree-LSTM on generated equalities",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # define and parse hyper-parameters of model/training from commandline
        parser.add_argument('--num-hidden', type=int, default=50,
                            help='hidden layer size')
        parser.add_argument('--num-embed', type=int, default=50,
                            help='embedding layer size')
        parser.add_argument('--memory-size', type=int, default=5,
                            help='max size of the stack/queue')
        parser.add_argument('--lr', type=float, default=0.001,
                            help='initial learning rate')
        parser.add_argument('--optimizer', type=str, default='adam',
                            help='the optimizer type')
        parser.add_argument('--mom', type=float, default=0.2,
                            help='momentum for sgd')
        parser.add_argument('--wd', type=float, default=0.00001,
                            help='weight decay for sgd')
        parser.add_argument('--beta1', type=float, default=0.9,
                            help='beta 1 for optimizer')
        parser.add_argument('--beta2', type=float, default=0.999,
                            help='beta 2 for optimizer')
        parser.add_argument('--dropout', type=float, default=0.2,
                            help='dropout probability (1.0 - keep probability)')

        # define and parse conditions of the experiment from commandline
        parser.add_argument('--model-class', type=str, default="NNTrees",
                            help='the classname of the model to run')
        parser.add_argument('--train-path', type=str, default=None,
                            help='path to training examples')
        parser.add_argument('--unify-one-zero', type=eval, default=True,
                            help='whether to unify ones and zeros to integer')
        parser.add_argument('--validation-path', type=str, default=None,
                            help='path to validation examples')
        parser.add_argument('--test-path', type=str, default=None,
                            help='path to test examples')
        parser.add_argument('--num-epochs', type=int, default=100,
                            help='max num of epochs')
        parser.add_argument('--seed', type=int, default=12093,
                            help='max num of epochs')
        parser.add_argument('--batch-size', type=int, default=1,
                            help='the batch size')
        parser.add_argument('--share-memory-params', default=False,
                            action='store_true',
                            help='whether to allow weight sharing for memory '
                                 'operations')
        parser.add_argument('--disable-sharing', default=False,
                            action='store_true',
                            help='whether to not allow weight sharing for memory '
                                 'operations')
        parser.add_argument('--no-op', default=False,
                            action='store_true',
                            help='whether to add no operation to stack push '
                                 'and pop')
        parser.add_argument('--no-pop', default=False,
                            action='store_true',
                            help='whether to add just push and no-op')
        parser.add_argument('--stack-type', type=str, default='simple',
                            help='choose the stack type. options are: simple, nn_stack, full_stack, add_stack, simple_gated, full_stack_gated')
        parser.add_argument('--likeLSTM', default=False,
                            action='store_true',
                            help='whether to make the mem2out stack tree completely like an LSTM+stack')
        parser.add_argument('--gate-push-pop', default=False,
                            action='store_true',
                            help='whether to make the push pop action a gate rather than a number')
        parser.add_argument('--normalize-action', default=False,
                            action='store_true',
                            help='whether to normalize push pop weight to 1 before pushing and poping')
        parser.add_argument('--gate-top-k', default=False,
                            action='store_true',
                            help='whether to gate the top-k instead of weighted average')
        parser.add_argument('--top-k', type=int, default=5,
                            help='the top-k stack elements will be used for computing the output'
                                 'select k. NOTE: k is in range(0,memory_size)')
        parser.add_argument('--numeric', default=False,
                            action='store_true',
                            help='whether to train on numeric equations')
        parser.add_argument('--fast', default=False,
                            action='store_true',
                            help='whether to evaluate only on 20% training') #TODO: This is not used by any code rn
        parser.add_argument('--verbose', default=False,
                            action='store_true',
                            help='whether to print execusion trace outputs')
        parser.add_argument('--curriculum', default=None,
                            help='what type of cirriculum (depth/func)')
        parser.add_argument('--switch-epoch', default=None, type=int,
                            help='epoch to switch over (2stage curriculum)')
        parser.add_argument('--curriculum-depth', default=None, type=int,
                            help='max depth in curriculum')
        parser.add_argument('--eval-depth', nargs="+", type=int, default=None,
                            help='list of depth to evaluate on')
        parser.add_argument('--tree-node-activation', type=str, default='sigmoid',
                            help='choose the activation for tree-node')
        parser.add_argument('--stack-node-activation', type=str,
                            default='sigmoid',
                            help='choose the activation for stack-node')


        # define and parse logging and save/loading options of the experiment
        parser.add_argument('--model-prefix', type=str, default=None,
                            help='path to save/load model')
        parser.add_argument('--result-path', type=str, default=None,
                            help='path to save results')
        parser.add_argument('--load-epoch', type=int, default=None,
                            help='load from epoch')
        parser.add_argument('--evaluate-only', action="store_true", default=False,
                            help='evaluate only')
        parser.add_argument('--trace-path', type=str, default=None,
                            help='path to save traces')
        parser.add_argument('--disp-epoch', type=int, default=1,
                            help='show progress for every n epochs')
        parser.add_argument('--checkpoint-every-n-epochs', type=int, default=5,
                            help='save model for every n epochs')
        parser.add_argument('--interactive', action='store_true', default=False,
                            help='interactive evaluation')

        self.args = parser.parse_args()

    def aggregate_statistics(self, record, depth=None):
        # if len(record) == 0:
        #     raise ValueError("empty record")
        if depth is not None:
            record = [item for item in record if item["depth"]==depth]
            if len(record) == 0:
                raise ValueError("empty record at depth %d" % depth)
        symbolic_true_positive = 0
        symbolic_true_negative = 0
        symbolic_false_positive = 0
        symbolic_false_negative = 0
        symbolic_loss = 0
        symbolic_count = 0
        numeric_true_positive = 0
        numeric_true_negative = 0
        numeric_false_positive = 0
        numeric_false_negative = 0
        numeric_loss = 0
        numeric_count = 0
        for item in record:
            if not item["ex"].is_numeric():
                if item["correct"] and (item["label"] == 1):
                    symbolic_true_positive += 1
                elif item["correct"] and (item["label"] == 0):
                    symbolic_true_negative += 1
                elif not item["correct"] and (item["label"] == 1):
                    symbolic_false_negative += 1
                elif not item["correct"] and (item["label"] == 0):
                    symbolic_false_positive += 1
                else:
                    assert False
                symbolic_loss += item["loss"]
                symbolic_count += 1

            else:
                if item["correct"] and (item["label"] == 1):
                    numeric_true_positive += 1
                elif item["correct"] and (item["label"] == 0):
                    numeric_true_negative += 1
                elif not item["correct"] and (item["label"] == 1):
                    numeric_false_negative += 1
                elif not item["correct"] and (item["label"] == 0):
                    numeric_false_positive += 1
                else:
                    assert False

                numeric_loss += item["loss"]
                numeric_count += 1
        assert symbolic_count == (symbolic_true_positive + symbolic_false_positive
                                  + symbolic_true_negative + symbolic_false_negative)
        symbolic_accuracy = (symbolic_true_positive + symbolic_true_negative) / symbolic_count if symbolic_count != 0 else 0
        symbolic_precision = symbolic_true_positive / (symbolic_true_positive + symbolic_false_positive) if (symbolic_true_positive + symbolic_false_positive) != 0 else 0
        symbolic_recall = symbolic_true_positive / (symbolic_true_positive + symbolic_false_negative) if (symbolic_true_positive + symbolic_false_negative) != 0 else 0
        symbolic_f1 = 2 * (symbolic_precision * symbolic_recall) / (symbolic_precision + symbolic_recall) if (symbolic_precision + symbolic_recall) != 0 else 0

        assert numeric_count == (
        numeric_true_positive + numeric_false_positive
        + numeric_true_negative + numeric_false_negative)

        numeric_accuracy = (
                            numeric_true_positive + numeric_true_negative) / \
                            numeric_count if numeric_count != 0 else 0
        numeric_precision = numeric_true_positive / (
        numeric_true_positive + numeric_false_positive) if (
                                                             numeric_true_positive + numeric_false_positive) != 0 else 0
        numeric_recall = numeric_true_positive / (
        numeric_true_positive + numeric_false_negative) if (
                                                             numeric_true_positive + numeric_false_negative) != 0 else 0
        numeric_f1 = 2 * (numeric_precision * numeric_recall) / (
        numeric_precision + numeric_recall) if (
                                             numeric_precision + numeric_recall) != 0 else 0

        return {
                "sym_loss_avg": symbolic_loss / symbolic_count if symbolic_count != 0 else 0,
                "sym_acc": symbolic_accuracy,
                "sym_precision": symbolic_precision,
                "sym_recall": symbolic_recall,
                "sym_f1": symbolic_f1,
                "sym_count": symbolic_count,
                "num_loss_avg": numeric_loss / numeric_count if numeric_count != 0 else 0,
                "num_acc": numeric_accuracy,
                "num_precision": numeric_precision,
                "num_recall": numeric_recall,
                "num_f1": numeric_f1,
                "num_count": numeric_count
            }

    def train(self, model, optimizer, batch):
        model.train()
        record, loss = model.compute_batch(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return record, loss.item()

    def eval(self, model, loader, trace=None):
        model.eval()
        aggregate_record = []
        aggregate_loss = 0
        for batch in loader:
            record, loss = model.compute_batch(batch, trace=trace)
            aggregate_record.extend(record)
            aggregate_loss += loss.item()
        return aggregate_record, (aggregate_loss / len(aggregate_record)) if len(aggregate_record) != 0 else 0

    def eval_and_log(self, model, loader, name, epoch, path, trace=None):
        if trace is not None:
            trace[name] = []
        record, loss = self.eval(model, loader, trace=trace[name] if trace is not None else None)

        # log
        depths = sorted(list(set(item["depth"] for item in record)))
        stats = self.aggregate_statistics(record)
        print("epoch={} set={} loss={} depth=all sym_loss={} "
              "sym_acc={} sym_prec={} sym_rec={} sym_f1={} "
              "sym_count={}".format(
            epoch, name, loss,
            stats["sym_loss_avg"], stats["sym_acc"],
            stats["sym_precision"], stats["sym_recall"],
            stats["sym_f1"], stats["sym_count"]
        ))
        print("epoch={} set={} loss={} depth=all num_loss={} "
              "num_acc={} num_prec={} num_rec={} num_f1={} "
              "num_count={}".format(
            epoch, name, loss,
            stats["num_loss_avg"], stats["num_acc"],
            stats["num_precision"], stats["num_recall"],
            stats["num_f1"], stats["num_count"]
        ))
        with open(path, "a") as fout:
            log_entry = {"type": "eval", "epoch":epoch, "set":name, "stats": {"all": stats}}
            for depth in depths:
                stats_d = self.aggregate_statistics(record, depth=depth)
                print("epoch={} set={} loss={} depth={} sym_loss={} "
                      "sym_acc={} sym_prec={} sym_rec={} sym_f1={} "
                      "sym_count={}".format(
                    epoch, name, None, depth,
                    stats_d["sym_loss_avg"], stats_d["sym_acc"],
                    stats_d["sym_precision"], stats_d["sym_recall"],
                    stats_d["sym_f1"], stats_d["sym_count"]
                ))
                print("epoch={} set={} loss={} depth={} num_loss={} "
                      "num_acc={} num_prec={} num_rec={} num_f1={} "
                      "num_count={}".format(
                    epoch, name, None, depth,
                    stats_d["num_loss_avg"], stats_d["num_acc"],
                    stats_d["num_precision"], stats_d["num_recall"],
                    stats_d["num_f1"], stats_d["num_count"]
                ))
                log_entry["stats"][depth] = stats_d
            fout.write(json.dumps(log_entry) + "\n")

    def save_model(self, epoch, model, optimizer):
        model_path = "%s-model-%d.checkpoint" % (self.args.model_prefix, epoch)
        torch.save(model.state_dict(), model_path)
        optim_path = "%s-optim-%d.checkpoint"% (self.args.model_prefix, epoch)
        torch.save(optimizer.state_dict(), optim_path)

    def load_model_and_optim(self, epoch, model, optimizer):
        model_path = "%s-model-%d.checkpoint" % (self.args.model_prefix, epoch)
        optim_path = "%s-optim-%d.checkpoint"% (self.args.model_prefix, epoch)
        model.load_state_dict(torch.load(model_path))
        optimizer.load_state_dict(torch.load(optim_path))

    def run(self):
        """
        Main routine for running this experiment.
        """
        torch.manual_seed(self.args.seed)
        random.seed(self.args.seed)
        if not self.args.interactive:
            train_loader, train_eval_loader, validation_loader, test_loader = load_equation_tree_examples(
                self.args.train_path,
                self.args.validation_path,
                self.args.test_path,
                batch_size=self.args.batch_size,
                numeric=self.args.numeric,
                eval_depth=self.args.eval_depth,
                unify_one_zero=self.args.unify_one_zero
            )

            if self.args.curriculum == "depth":
                easy_train_loader, easy_train_eval_loader, easy_validation_loader, easy_test_loader = load_equation_tree_examples(
                    self.args.train_path,
                    self.args.validation_path,
                    self.args.test_path,
                    batch_size=self.args.batch_size,
                    numeric=self.args.numeric,
                    eval_depth=self.args.eval_depth,
                    unify_one_zero=self.args.unify_one_zero,
                    filter=lambda x: x.depth <= self.args.curriculum_depth
                )
            if self.args.curriculum == "func":
                prohibited = set(UNARY_FNS)
                easy_train_loader, easy_train_eval_loader, \
                easy_validation_loader, easy_test_loader = \
                    load_equation_tree_examples(
                    self.args.train_path,
                    self.args.validation_path,
                    self.args.test_path,
                    batch_size=self.args.batch_size,
                    numeric=self.args.numeric,
                    eval_depth=self.args.eval_depth,
                    unify_one_zero=self.args.unify_one_zero,
                    filter=lambda x: x.all(lambda n:n.function_name not in prohibited)
                )


        model = build_nnTree(self.args.model_class,
                             self.args.num_hidden,
                             self.args.num_embed,
                             self.args.memory_size,
                             self.args.share_memory_params,
                             self.args.dropout,
                             self.args.no_op,
                             self.args.stack_type,
                             self.args.top_k,
                             self.args.verbose,
                             self.args.tree_node_activation,
                             self.args.stack_node_activation,
                             self.args.no_pop,
                             self.args.disable_sharing,
                             self.args.likeLSTM,
                             self.args.gate_push_pop,
                             self.args.gate_top_k,
                             self.args.normalize_action)

        totParams = 0
        for p in model.parameters():
            totParams += p.numel()
        # print('total num params:', totParams)
        #

        optimizer = build_optimizer((param for param in model.parameters() if
                                        param.requires_grad),
                                    self.args.optimizer,
                                    self.args.lr,
                                    self.args.mom,
                                    self.args.wd,
                                    self.args.beta1,
                                    self.args.beta2)

        if self.args.load_epoch is None or self.args.evaluate_only is False:
            with open(self.args.result_path, "wt") as _: # TODO: if fine-tuning option is added this should be fixed
                pass


        batch_counter = 0
        if self.args.interactive:
            self.load_model_and_optim(self.args.load_epoch, model, optimizer)
            self.interactive(model)
        elif self.args.evaluate_only:
            trace = dict()
            self.load_model_and_optim(self.args.load_epoch, model, optimizer)
            self.eval_and_log(model, train_eval_loader, "train", self.args.load_epoch, "tmp", trace=trace)
            self.eval_and_log(model, validation_loader, "validation", self.args.load_epoch, "tmp", trace=trace)
            self.eval_and_log(model, test_loader, "test", self.args.load_epoch, "tmp", trace=trace)
            if self.args.trace_path:
                with open(self.args.trace_path, "wb") as f:
                    f.write(pickle.dumps(trace))
        elif self.args.curriculum is not None:
            for epoch in range(1, self.args.num_epochs + 1):
                if epoch <= self.args.switch_epoch:
                    curriculum_loader = easy_train_loader
                else:
                    curriculum_loader = train_loader
                for batch in curriculum_loader:
                    _, loss = self.train(model, optimizer, batch)
                    batch_counter += 1
                    print("iter %d loss=%f" % (batch_counter, loss))
                    with open(self.args.result_path, "a") as fout:
                        fout.write(json.dumps(
                            {"type": "train", "iter": batch_counter,
                             "set": "train", "loss": loss}) + "\n")
                if epoch % self.args.disp_epoch is 0:
                    self.eval_and_log(model, easy_train_loader, "easy_train", epoch,
                                      self.args.result_path)
                    self.eval_and_log(model, easy_validation_loader, "easy_validation",
                                      epoch, self.args.result_path)
                    self.eval_and_log(model, easy_test_loader, "easy_test", epoch,
                                      self.args.result_path)
                    self.eval_and_log(model, train_eval_loader, "train", epoch,
                                      self.args.result_path)
                    self.eval_and_log(model, validation_loader, "validation",
                                      epoch, self.args.result_path)
                    self.eval_and_log(model, test_loader, "test", epoch,
                                      self.args.result_path)
                if epoch % self.args.checkpoint_every_n_epochs == 0:
                    self.save_model(epoch, model, optimizer)
        else:
            for epoch in range(1, self.args.num_epochs + 1):
                for batch in train_loader:
                    _, loss = self.train(model, optimizer, batch)
                    batch_counter += 1
                    print("iter %d loss=%f" % (batch_counter, loss))
                    with open(self.args.result_path, "a") as fout:
                        fout.write(json.dumps(
                            {"type": "train", "iter": batch_counter,
                             "set": "train", "loss": loss}) + "\n")
                if epoch % self.args.disp_epoch is 0:
                    # self.eval_and_log(model, train_eval_loader, "train", epoch,
                    #                   self.args.result_path)
                    self.eval_and_log(model, validation_loader, "validation",
                                      epoch, self.args.result_path)
                    self.eval_and_log(model, test_loader, "test", epoch,
                                      self.args.result_path)
                if epoch % self.args.checkpoint_every_n_epochs == 0:
                    self.save_model(epoch, model, optimizer)

    def interactive(self, model):
        with open(self.args.train_path, "rt") as fin:
            train_json = json.loads(fin.read())
            train_trio = build_equation_tree_examples_list(train_json,
                                                           numeric=self.args.numeric,
                                                           unify_one_zero=self.args.unify_one_zero)
            train = set(str(tree) for tree,_,_ in train_trio)
        gtrace = []
        exit = False
        while True:
            while True:
                try:
                    s = input("enter an equation:\n")
                    if s == "exit":
                        exit = True
                        break
                    equation = parse_equation(s)
                    break
                except Exception:
                    traceback.print_exc()
            if exit:
                break
            trio = load_single_equation_tree_example(equation.dump())
            tree = trio[0]
            tree_r = load_single_equation_tree_example(equation.dump())[0]
            tree_r.lchild, tree_r.rchild = tree_r.rchild, tree_r.lchild
            print((str(tree) in train) or (str(tree_r) in train))
            loader = sequential_sampler(trios=[trio],batch_size=1)
            trace = []
            record, loss = self.eval(model, loader, trace=trace)
            print(trace[0].probability)
            gtrace.append(trace[0])
        if self.args.trace_path:
            with open(self.args.trace_path, "wb") as f:
                f.write(pickle.dumps({"train":gtrace}))


if __name__ == "__main__":
    # train-path as input
    nnTreeEquationVerificationExperiment().run()
