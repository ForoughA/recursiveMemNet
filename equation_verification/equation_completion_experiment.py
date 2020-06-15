import argparse
import math
import pickle
import traceback
from collections import defaultdict

import torch
import random
import json

from equation_verification.dataset_loading import load_equation_tree_examples, \
    load_single_equation_tree_example, sequential_sampler, \
    build_equation_tree_examples_list, load_equation_completion_batch, \
    extract_candidates, load_equation_completion_blank_example
from equation_verification.nn_tree_model import build_nnTree
from optimizers import build_optimizer
from parse import parse_equation
from equation_verification.constants import UNARY_FNS

NCHOICES = 0
CUT = 100000

class EquationCompletionExperiment:

    def __init__(self):
        """
        Hyperparameters defined here should match nn_tree_experiment.py exactly,
        reasoning being that this experiment relies on a model trained using
        nn_tree_experiment.py, so all the configurations should be the same.
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
        parser.add_argument('--dropout', type=float, default=None,
                            help='dropout probability (1.0 - keep probability)')

        # define and parse conditions of the experiment from commandline
        parser.add_argument('--model-class', type=str, default="NNTrees",
                            help='the classname of the model to run')
        parser.add_argument('--train-path', type=str, default=None,
                            help='path to training examples')
        parser.add_argument('--candidate-path', type=str, default=None,
                            help='path to equation completion candidate answers')
        parser.add_argument('--unify-one-zero', type=eval, default=True,
                            help='whether to unify ones and zeros to integer')
        parser.add_argument('--validation-path', type=str, default=None, nargs="+",
                            help='path(s) to validation examples, if multiple files supplied, will evaluate each individually')
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
        parser.add_argument('--cut', type=int, default=1000_000,
                            help='cut after this many examples')

        self.args = parser.parse_args()
        with open(self.args.candidate_path) as cddf:
            cddjson = json.loads(cddf.read())
            self.candidate_trees = [tup[0] for tup in build_equation_tree_examples_list(cddjson, unify_one_zero=self.args.unify_one_zero)]
            self.candidate_classes = defaultdict(list)
            for i,tree in enumerate(self.candidate_trees):
                self.candidate_classes[tree.cls].append(i)

    def load_model_and_optim(self, epoch, model, optimizer):
        model_path = "%s-model-%d.checkpoint" % (self.args.model_prefix, epoch)
        optim_path = "%s-optim-%d.checkpoint"% (self.args.model_prefix, epoch)
        model.load_state_dict(torch.load(model_path))
        optimizer.load_state_dict(torch.load(optim_path))

    def eval(self, model, loader, trace=None):
        model.eval()
        aggregate_record = []
        aggregate_loss = 0
        for batch in loader:
            record, loss = model.compute_batch(batch, trace=trace)
            aggregate_record.extend(record)
            aggregate_loss += loss.item()
        return aggregate_record, (aggregate_loss / len(aggregate_record)) if len(aggregate_record) != 0 else 0

    def aggregate_statistics(self, record, depth=None):
        assert len(record) % NCHOICES == 0
        if depth is not None:
            record = [item for item in record if item["depth"]==depth]
            if len(record) == 0:
                raise ValueError("empty record at depth %d" % depth)
        result = {}
        ranks = []
        for i in range(0,len(record), NCHOICES):
            scores = [item["score"] for item in record[i:i+NCHOICES]]
            labels = [item["label"] for item in record[i:i+NCHOICES]]
            true_ex = record[i+labels.index(1)]["ex"]
            indices = list(range(NCHOICES))
            assert sum(labels) == 1 # should only be one correct
            sorted_scores = sorted(list(zip(scores,labels,indices)), key=lambda x:x[0], reverse=True)
            print("-------- -------- -------- -------- -------- --------")
            print(true_ex.pretty_str())
            for i,(score, label, ind) in enumerate(sorted_scores):
                if label == 1:
                    ranks.append(i+1)
                if i < 20:
                    print(score, 1/(1+math.exp(score)),self.candidate_trees[ind].pretty_str())
            print("rank", ranks[-1])
        for i in range(10):
            name = f"top{i}_acc"
            val = len([rank for rank in ranks if rank <= i+1]) / len(ranks)
            result[name] = val
        result["mrr"] = sum([1/rank for rank in ranks]) / len(ranks)
        return result

    def log(self, record, path, epoch, name):
        depths = sorted(list(set(item["depth"] for item in record)))
        stats = self.aggregate_statistics(record)
        with open(path, "a") as fout:
            log_entry = {"type": "eval", "epoch":epoch, "set":name, "stats": {"all": stats}}
            for depth in depths:
                stats_d = self.aggregate_statistics(record, depth=depth)
                log_entry["stats"][depth] = stats_d
            fout.write(json.dumps(log_entry) + "\n")

    def aggregate_statistics_rank(self, rank_record, depth=None):
        if depth is not None:
            rank_record = [item for item in rank_record if item["depth"]==depth]
            if len(rank_record) == 0:
                raise ValueError("empty rank_record at depth %d" % depth)
        result = {}
        for ranktype in rank_record[0]["ranks"].keys():
            ranks = [item["ranks"][ranktype] for item in rank_record]
            for i in range(10):
                name = f"top{i+1}_acc_{ranktype}"
                val = len([rank for rank in ranks if rank <= i+1]) / len(ranks)
                result[name] = val
            result[f"mrr_{ranktype}"] = sum([1/rank for rank in ranks]) / len(ranks)
        result["count"] = len(rank_record)
        return result

    def log_rank(self, ranks, path, epoch, name):
        depths = sorted(list(set(item["depth"] for item in ranks)))
        stats = self.aggregate_statistics_rank(ranks)
        with open(path, "a") as fout:
            log_entry = {"type": "eval", "epoch":epoch, "set":name, "stats": {"all": stats},"raw":ranks}
            for depth in depths:
                stats_d = self.aggregate_statistics_rank(ranks, depth=depth)
                log_entry["stats"][depth] = stats_d
            fout.write(json.dumps(log_entry) + "\n")

    def run(self):
        """
        Main routine for running this experiment.
        """
        torch.manual_seed(self.args.seed)
        random.seed(self.args.seed)
        global NCHOICES
        with open(self.args.candidate_path) as cddf:
            cddjson = json.loads(cddf.read())
            candidates = extract_candidates(cddjson)
            NCHOICES = 0
            for group in cddjson:
                NCHOICES += len(group)
            assert NCHOICES == len(candidates)
        print(f"{NCHOICES} number of candidates for equation completion")

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
        print('total num params:', totParams)

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


        if self.args.evaluate_only:
            self.load_model_and_optim(self.args.load_epoch, model, optimizer)
            with open(self.args.test_path,"rt") as f:
                groups = json.loads(f.read())
                # record_agg = []
                rank_agg = []
                done_counter = 0
                for group in groups:
                    for example in group:
                        if done_counter >= self.args.cut:
                            break
                        if example["blankNodeNum"] == "0":
                            continue
                        if "Number" in example["equation"]["func"]:
                            continue
                        done_counter += 1
                        print(done_counter)
                        blank, lbbb, ddd = load_equation_completion_blank_example(example)
                        test_loader = load_equation_completion_batch(
                            [[example]],
                            batch_size=self.args.batch_size,
                            numeric=self.args.numeric,
                            eval_depth=self.args.eval_depth,
                            unify_one_zero=self.args.unify_one_zero,
                            equation_completion=True,
                            candidates=candidates
                        )
                        record, _ = self.eval(model, test_loader, trace=None)
                        assert len(record) == NCHOICES
                        # record_agg.extend([{"score":item["score"],
                        #                     "label":item["label"],
                        #                     "depth":item["depth"],
                        #                     "ex":item["ex"] if item["label"] else None} for item in record])
                        # computing statistics
                        ranks = self.compute_statistic_single(blank, record)
                        rank_agg.append(ranks)
                    if done_counter >= self.args.cut:
                        break
            # self.log(record_agg, self.args.result_path, self.args.load_epoch, "test")
            self.log_rank(rank_agg, self.args.result_path, self.args.load_epoch, "test")
        else:
            print("doing nothing, please set flag --evaluate-only")

    def compute_statistic_single(self, blank, record):
        scores = [item["score"] for item in record]
        labels = [item["label"] for item in record]
        indices = list(range(NCHOICES))
        classes = [tree.cls for tree in self.candidate_trees]
        assert sum(labels) == 1 # should only be one correct
        sorted_scores = sorted(list(zip(scores,labels,indices,classes)), key=lambda x:x[0], reverse=True)
        print("-------- -------- -------- -------- -------- --------")
        print(blank.pretty_str())
        raw_rank = None
        true_class = None
        for i,(score, label, ind, cls) in enumerate(sorted_scores):
            if label == 1:
                raw_rank = i+1
                true_class = cls
                print(score, math.exp(score)/(1+math.exp(score)), cls, f"[{self.candidate_trees[ind].pretty_str()}]")
            if i < 20:
                print(score, math.exp(score)/(1+math.exp(score)), cls,self.candidate_trees[ind].pretty_str())
        print("raw_rank", raw_rank)
        class_rank = None
        for i,(score, label, ind, cls) in enumerate(sorted_scores):
            if class_rank is None and cls == true_class:
                class_rank = i+1
                print(score, math.exp(score)/(1+math.exp(score)), cls, f"[{self.candidate_trees[ind].pretty_str()}]")
            if i < 20:
                print(score, math.exp(score)/(1+math.exp(score)), cls, self.candidate_trees[ind].pretty_str())
        print("class_rank", class_rank)
        collapse_scores = []
        seen = set()
        for i,(score, label, ind, cls) in enumerate(sorted_scores):
            if cls not in seen:
                collapse_scores.append((score, label, ind, cls))
                seen.add(cls)
            else:
                continue
        collapse_rank = None
        for i,(score, label, ind, cls) in enumerate(collapse_scores):
            if cls == true_class:
                collapse_rank = i+1
                print(score, math.exp(score)/(1+math.exp(score)), cls, f"[{self.candidate_trees[ind].pretty_str()}]")
            if i < 20:
                print(score, math.exp(score)/(1+math.exp(score)), cls,self.candidate_trees[ind].pretty_str())
        print("collapse_rank", collapse_rank)
        random_collapse_ranks = []
        for sample_s in range(10):
            random_collapse_inds = set(random.choice(cands) for cands in self.candidate_classes.values())
            random_collapse_scores = [(s,l,ind,c) for (s,l,ind,c) in sorted_scores if ind in random_collapse_inds]
            random_collapse_rank = None
            for i,(score, label, ind, cls) in enumerate(random_collapse_scores):
                if cls == true_class:
                    random_collapse_rank = i+1
                    print(score, math.exp(score)/(1+math.exp(score)), cls, f"[{self.candidate_trees[ind].pretty_str()}]")
                if i < 20:
                    print(score, math.exp(score)/(1+math.exp(score)), cls,self.candidate_trees[ind].pretty_str())
            random_collapse_ranks.append(random_collapse_rank)
        random_collapse_rank = sum(random_collapse_ranks) / len(random_collapse_ranks)
        random_collapse_rank_std = (sum([(r-random_collapse_rank) ** 2 for r in random_collapse_ranks]))**0.5
        print("random_collapse_rank", random_collapse_rank)
        print()
        print("raw_rank", raw_rank)
        print("class_rank", class_rank)
        print("collapse_rank", collapse_rank)
        print("random_collapse_rank", random_collapse_rank)
        bins = [0] * 20
        unit = 1/len(bins)
        for (score, label, ind, cls) in sorted_scores:
            bin_idx = math.floor((math.exp(score)/(1+math.exp(score)) / unit))
            if bin_idx >= len(bins):
                assert bin_idx == len(bins)
                bin_idx = bin_idx - 1
            bins[bin_idx] += 1
        maxcnt = max(bins)
        normalized_bins = [cnt/maxcnt for cnt in bins]
        print("1.0")
        for ncnt in reversed(normalized_bins):
            print(max(math.ceil(ncnt), math.ceil(ncnt * 60) )* "+")
        normalized_bins = [cnt/NCHOICES for cnt in bins]
        print("0.0")
        print("1.0")
        for ncnt in reversed(normalized_bins):
            print(max(math.ceil(ncnt), math.ceil(ncnt * 60) )* "+")
        print("0.0")
        return {"ranks":{"raw":raw_rank, "class":class_rank, "collapse":collapse_rank, "random_collapse":random_collapse_rank},"depth":blank.depth,"scores":sorted_scores,"random_collapse_std":random_collapse_rank_std}


if __name__ == "__main__":
    # train-path as input
    EquationCompletionExperiment().run()