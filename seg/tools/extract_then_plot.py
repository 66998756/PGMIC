# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
"""Modified from https://github.com/open-
mmlab/mmdetection/blob/master/tools/analysis_tools/analyze_logs.py."""
import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

import glob
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='extract training history to csv file')
    parser.add_argument(
        'work_dirs',
        type=str,
        nargs='+',
        help='path of train history work dirs')
    parser.add_argument(
        '--keys',
        type=str,
        default='IoU',
        help='the metric that you want to extract, IoU, Acc(predict), acc(model) or loss')
    parser.add_argument(
        '--filter',
        type=str,
        nargs='+',
        help='the metric that you want to extract, for example "bick" "road"')
    
    # for the target history
    parser.add_argument('--title', type=str, help='title of figure')
    parser.add_argument(
        '--legend',
        type=str,
        nargs='+',
        default=None,
        help='legend of each plot')
    parser.add_argument(
        '--backend', type=str, default=None, help='backend of plt')
    parser.add_argument(
        '--style', type=str, default='dark', help='style of plt')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()
    return args


def load_json_logs(json_logs):
    # load and convert json_logs to log_dict, key is epoch, value is a sub dict
    # keys of sub dict is different metrics
    # value of sub dict is a list of corresponding values of all iterations
    first = True
    pre_iter = 0
    log_dicts = dict()

    json_logs = glob.glob(json_logs[0] + "/*.json")[0]
    with open(json_logs, 'r') as log_file:
        for line in log_file:
            log = json.loads(line.strip())
            # skip lines without `epoch` field
            if 'iter' not in log:
                continue
            iter = log.pop('iter')
            # a bug in original mic, when validation
            # iter item will set to 500
            if iter == 500:
                iter = pre_iter + 50
                if first:
                    first = False
                    continue
                if iter not in log_dicts:
                    log_dicts[iter] = defaultdict(list)
                for k, v in log.items():
                    log_dicts[iter][k].append(v)
            pre_iter = iter
    return log_dicts


def convert_csv(log_dicts, args):
    output_path = args.work_dirs[0]

    csv_dict = {}
    for iters, scores in log_dicts.items():
        iter_dict = {}
        for score in scores.keys():
            if args.keys in score:
                iter_dict[score] = scores[score][0]

        csv_dict[iters] = iter_dict

    csv_df = pd.DataFrame(csv_dict).transpose()
    print(csv_df)
    csv_df.to_csv(output_path + '/valid_history.csv', index_label='Iteration')

    return csv_df


def plot_curve(args):
    converted_csv = pd.read_csv(args.work_dirs[0] + '/valid_history.csv')
    
    x = converted_csv['Iteration']
    overall = 'm' + args.keys
    overall_y = converted_csv[overall]

    fig, axs = plt.subplots(1, 2, figsize=(9 * 2, 5 * 1), squeeze=False)

    axs[0][0].plot(x, overall_y, label=overall)
    # axs[0][0].set_xticklabels(x, rotation=30)
    axs[0][0].legend()
    axs[0][0].set_title('{} curve'.format(overall))

    if args.filter is None:
        for k in converted_csv.keys():
            if k != overall and k != 'Iteration':
                axs[0][1].plot(x, converted_csv[k], label=k)
    else:
        for target in args.filter:
            for k in converted_csv.keys():
                if target in k:
                    axs[0][1].plot(x, converted_csv[k], label=k)
    
    # axs[0][1].set_xticklabels(x, rotation=30)
    axs[0][1].legend(ncol=4)
    if args.filter is None:
        axs[0][1].set_ylim([-0.02 * 5, 1.05])
    else:    
        axs[0][1].set_ylim([-0.02 * int(len(args.filter)/4), 1.05])
    axs[0][1].set_title('{} curve'.format(args.keys))

    plt.savefig(
        args.work_dirs[0] + args.keys + "_curve.png")
    plt.close()


def main():
    args = parse_args()
    work_dirs = args.work_dirs
    log_dicts = load_json_logs(work_dirs)
    convert_csv(log_dicts, args)
    plot_curve(args)


if __name__ == '__main__':
    main()
