
import argparse
import glob
import matplotlib.pyplot as plt

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Print the learning curve')
    parser.add_argument(
        'work_dirs',
        type=str,
        nargs='+',
        help='path of target trained exp')
    
    parser.add_argument(
        '--ordered_title',
        type=str,
        nargs='+',
        default=['Baseline', 'Ours'],
        help='the metric that you want to plot')
    
    parser.add_argument(
        '--keys',
        type=str,
        nargs='+',
        default=['mIoU'],
        help='the metric that you want to plot')
    
    parser.add_argument('--xlim', type=float, nargs='+', default=[20000, 80000])
    parser.add_argument('--ylim', type=float, nargs='+', default=[0.72, 0.78])
    
    parser.add_argument('--out', type=str, default=None)

    args = parser.parse_args()

    return args


def compare_curve(csv_file, args):
    datasets = {}
    paired_y = {}
    for exp_name, csv_path in csv_file.items():
        exp_csv = pd.read_csv(csv_path[0])

        x = exp_csv['Iteration']
        y = {}
        paired_y[exp_name] = []
        for key in args.keys:
            y[key] = exp_csv[key]

        datasets[exp_name] = {
            'x': x,
            'y': y
        }
    
    for idx, (exp_name, datas) in enumerate(datasets.items()):
        x = datas['x']
        for data in datas['y'].keys():
            plt.plot(x, datas['y'][data], label=args.ordered_title[idx] + '_' + data)
            paired_y[exp_name] = datas['y'][data]
    print(datasets)
    index = 0
    for idx, value in enumerate(x):
        if value == 40000:
            index = idx
            break
    
    # print(paired_y)
    plt.axvline(x[index], linestyle='--', color='gray')
    for key in paired_y.keys():
        plt.fill_between(x[index:], paired_y[key][index:], paired_y[key][index:], 
                         color='lightyellow', hatch='/', edgecolor='gray')
    plt.legend()
    plt.title("IoU Curve")
    plt.xlabel('Iteration')
    plt.ylabel('IoU')
    plt.xlim(args.xlim)
    plt.ylim(args.ylim)
    print(f'save curve to: {args.out}')
    plt.savefig(args.out)
    plt.cla()


def main():
    args = parse_args()

    csv_file = {}
    work_dirs = args.work_dirs
    for work_dir in work_dirs:
        csv_path = glob.glob(work_dir + '/*.csv')
        assert csv_path != None
        csv_file[work_dir.split('/')[-2]] = csv_path
    # print(csv_file)
    compare_curve(csv_file, args)


if __name__ == '__main__':
    main()