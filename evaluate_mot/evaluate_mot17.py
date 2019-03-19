import motmetrics as mm
import glob
import os
from motmetrics.apps.eval_motchallenge import compare_dataframes
from collections import OrderedDict
from pathlib import Path


def evaluate_mot17(
        ground_truth_path,
        test_path
):
    """Evaluate Mot17 by the filename

    Args:
        ground_truth_path: gt files directory
        test_path: directry of the tracker's result

    Returns: The dataframe of evaluate frames.

    """

    fmt = 'mot16'

    gt = read_gt_mot17(ground_truth_path)


    ts = read_test_mot17(test_path)

    summary = get_summary_mot17(gt, ts)

    return summary


def read_gt_mot17(ground_truth_path):
    """
    Read the ground truth ``gt.txt`` from the specified directories.

    Args:
        ground_truth_path: The ground truth directory.

    Returns: The ground truth data with the standard format. Its type is OrderedDict.

    """
    fmt = 'mot16'
    gtfiles = glob.glob(os.path.join(
        ground_truth_path,
        '*/gt/gt.txt'
    ))
    gt = OrderedDict(
        [
            (
                Path(f).parts[-3],
                mm.io.loadtxt(
                    f,
                    fmt=fmt,
                    min_confidence=1
                )
            )
            for f in gtfiles
        ]
    )
    return gt


def read_test_mot17(test_path):
    """
    Read test result from the specified directory *test_path*.

    Args:
        test_path:the directory which contains the data to be evaluated.

    Returns: The test result of your tracker with the standard format. Its type is OrderedDict.

    """
    fmt = 'mot16'
    tsfiles = [
        f for f in glob.glob(
            os.path.join(
                test_path,
                '*.txt'
            ))
        if os.path.basename(f).startswith('MOT17')
    ]

    ts = OrderedDict([
        (
            os.path.splitext(Path(f).parts[-1])[0],
            mm.io.loadtxt(f, fmt=fmt)
        )
        for f in tsfiles
    ])
    return ts


def read_test_mot17_single_file(test_path):
    fmt = 'mot16'
    return OrderedDict([
        (
            os.path.splitext(Path(test_path).parts[-1])[0],
            mm.io.loadtxt(test_path, fmt=fmt)
        )
    ])

def get_summary_mot17(gt, ts):
    """
    Get the summary by the ground truth (gt) data and test data (ts).
    Args:
        gt: ground truth data which is the OrderedDict
        ts: data to be evaluated whose type is also the OrderedDict

    Returns: A pandas dataframe format result.

    """
    mh = mm.metrics.create()
    accs, names = compare_dataframes(gt, ts)

    summary = mh.compute_many(
        accs,
        names=names,
        metrics=mm.metrics.motchallenge_metrics,
        generate_overall=True
    )

    mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )

    return summary


def convert_to_standard_format():
    pass


if __name__ == "__main__":
    summary = evaluate_mot17(
        ground_truth_path='./ground_truth/train',
        test_path='./result/0190304-ssj-sst-mean/3_2_0_2_5_5',
    )

