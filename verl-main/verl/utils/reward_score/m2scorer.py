#!/usr/bin/python
import os
# This file is part of the NUS M2 scorer.
# The NUS M2 scorer is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# The NUS M2 scorer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# file: m2scorer.py
# 
# score a system's output against a gold reference 
#
# Usage: m2scorer.py [OPTIONS] proposed_sentences source_gold
# where
#  proposed_sentences   -   system output, sentence per line
#  source_gold          -   source sentences with gold token edits
# OPTIONS
#   -v    --verbose             -  print verbose output
#   --very_verbose              -  print lots of verbose output
#   --max_unchanged_words N     -  Maximum unchanged words when extracting edits. Default 2."
#   --beta B                    -  Beta value for F-measure. Default 0.5."
#   --ignore_whitespace_casing  -  Ignore edits that only affect whitespace and caseing. Default no."
#

import sys
from verl.utils.reward_score import levenshtein

def load_annotation(golden_txt):
    source_sentences = []
    gold_edits = []

    # 替换原本从文件读取的逻辑，这里我们直接使用传入的 string_list
    puffer_lines = golden_txt  # 假设传入的是按行分割的字符串列表

    # 模拟 paragraphs 函数的行为（如果需要）
    # 假设 paragraphs 是把连续的段落合并成一组（根据空行分隔），你可以用如下方式模拟：
    items = []
    current_item = []
    for line in puffer_lines:
        if line.strip() == '':  # 空行表示段落结束
            if current_item:
                items.append(current_item)
                current_item = []
        else:
            current_item.append(line)
    if current_item:  # 添加最后一个段落
        items.append(current_item)
    for item in items:
        item_lines = item  # 不再调用 splitlines(False)，因为我们已经有了每一行
        sentence = [line[2:].strip() for line in item_lines if line.startswith('S ')]
        assert sentence != []
        annotations = {}
        for line in item_lines[1:]:
            if line.startswith('I ') or line.startswith('S '):
                continue
            assert line.startswith('A ')
            line = line[2:]
            fields = line.split('|||')
            start_offset = int(fields[0].split()[0])
            end_offset = int(fields[0].split()[1])
            etype = fields[1]
            if etype == 'noop':
                start_offset = -1
                end_offset = -1
            corrections = [c.strip() if c != '-NONE-' else '' for c in fields[2].split('||')]
            original = ' '.join(' '.join(sentence).split()[start_offset:end_offset])
            annotator = int(fields[5])
            if annotator not in annotations:
                annotations[annotator] = []
            annotations[annotator].append((start_offset, end_offset, original, corrections))
        tok_offset = 0
        for this_sentence in sentence:
            tok_offset += len(this_sentence.split())
            source_sentences.append(this_sentence)
            this_edits = {}
            for annotator, annotation in annotations.items():
                this_edits[annotator] = [edit for edit in annotation if edit[0] <= tok_offset and edit[1] <= tok_offset and edit[0] >= 0 and edit[1] >= 0]
            if len(this_edits) == 0:
                this_edits[0] = []
            gold_edits.append(this_edits)
    return (source_sentences, gold_edits)


def print_usage():
    print >> sys.stderr, "Usage: m2scorer.py [OPTIONS] proposed_sentences gold_source"
    print >> sys.stderr, "where"
    print >> sys.stderr, "  proposed_sentences   -   system output, sentence per line"
    print >> sys.stderr, "  source_gold          -   source sentences with gold token edits"
    print >> sys.stderr, "OPTIONS"
    print >> sys.stderr, "  -v    --verbose                   -  print verbose output"
    print >> sys.stderr, "        --very_verbose              -  print lots of verbose output"
    print >> sys.stderr, "        --max_unchanged_words N     -  Maximum unchanged words when extraction edit. Default 2."
    print >> sys.stderr, "        --beta B                    -  Beta value for F-measure. Default 0.5."
    print >> sys.stderr, "        --ignore_whitespace_casing  -  Ignore edits that only affect whitespace and caseing. Default no."


def cal_m2scorer(system_txt, gold_txt):
    max_unchanged_words=2
    beta = 0.5
    ignore_whitespace_casing= False
    verbose = False
    very_verbose = False
    # opts, args = getopt(sys.argv[1:], "v", ["max_unchanged_words=", "beta=", "verbose", "ignore_whitespace_casing", "very_verbose"])
    # for o, v in opts:
    #     if o in ('-v', '--verbose'):
    #         verbose = True
    #     elif o == '--very_verbose':
    #         very_verbose = True
    #     elif o == '--max_unchanged_words':
    #         max_unchanged_words = int(v)
    #     elif o == '--beta':
    #         beta = float(v)
    #     elif o == '--ignore_whitespace_casing':
    #         ignore_whitespace_casing = True
    #     else:
    #         print >> sys.stderr, "Unknown option :", o
    #         print_usage()
    #         sys.exit(-1)

    # starting point
    # if len(args) != 2:
    #     print_usage()
    #     sys.exit(-1)
    #
    # system_file = args[0]
    # gold_file = args[1]

    # load source sentences and gold edits
    source_sentences, gold_edits = load_annotation(gold_txt)

    # load system hypotheses
    # system_sentences = [line.decode("utf8").strip() for line in fin.readlines()]
    system_sentences = system_txt

    p, r, f1 = levenshtein.batch_multi_pre_rec_f1(system_sentences, source_sentences, gold_edits, max_unchanged_words, beta, ignore_whitespace_casing, verbose, very_verbose)

    # print("Precision   : %.4f" % p)
    # print("Recall      : %.4f" % r)
    # print("F_%.1f       : %.4f" % (beta, f1))
    return p, r, f1

# if __name__ == '__main__':
#     system_txt = ["In meiner Familie ist der Vater sehr wichtig ."]
#     gold_source = """S In meiner Familian , Vater ist sehr wichtig .
#     A 2 3|||WS|||Familie|||REQUIRED|||-NONE-|||0
#     A 3 4|||ZS|||-NONE-|||REQUIRED|||-NONE-|||0
#     A 4 4|||StV-|||ist|||REQUIRED|||-NONE-|||0
#     A 4 4|||Det|||mein|||REQUIRED|||-NONE-|||0
#     A 5 6|||StV|||-NONE-|||REQUIRED|||-NONE-|||0"""
#     gold_source = gold_source.split("\n")
#
#     p, r, f1 = cal_m2scorer(system_txt, gold_source)
#
#     print(p)
