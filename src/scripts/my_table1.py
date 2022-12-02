import os
import argparse

job_dict = {
    "categories": [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ],
    "folds": [0, 1, 2, 3, 4],
    "mode": ["--npca"],
    "thresholds": [0.99],
    "archs": ["efficientnet-b4"],
    "args":  "--model gaussian --max_nb_epochs 0 --batch_size 8 --extract_blocks 0 1 2 3 4 5 6 7 8 --select_space 2_3",
}


def load_jobs(grid, logfolder_path, gpu):  # jobs = load_jobs(job_dict, args.logpath, args.gpu)
    # Read Arguments from File

    global job_version  # for some reason nonlocal does not work
    job_version = 0

    def job_category_fold(category, fold, mode, threshold, arch):
        job = grid.copy()
        global job_version
        if mode is not None and threshold is not None:
            job["args"] = (
                    " "
                    + job["args"]
                    + " --category {} --logpath {} --version {} {} --variance_threshold {} --arch {}".format(
                category,
                logfolder_path,
                job_version,
                mode,
                threshold,
                arch,
            )
            )
        else:
            job["args"] = (
                    " "
                    + job["args"]
                    + " --category {} --logpath {} --version {} --arch {}".format(
                category, logfolder_path, job_version, arch
            )
            )
        if gpu:
            job["args"] = job["args"] + " --gpus 0"
        job_version += 1
        if fold is not None:
            job["args"] += " --fold {}".format(fold)
        return job

    # cross-product the grid
    jobs = [
        job_category_fold(category, fold, mode, threshold, arch)
        for category in grid["categories"]
        for fold in grid["folds"]
        for mode in grid["mode"]
        for threshold in grid["thresholds"]
        for arch in grid["archs"]
    ]

    return jobs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logpath",
        type=str,
        default=os.getcwd(),
        help="The path where logs should go",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Whether or not to use GPU acceleration",
    )

    args = parser.parse_args()

    jobs = load_jobs(job_dict, args.logpath, args.gpu)

    for job in jobs:
        print("python -m src.common.trainer" + job["args"])
        os.system("python -m src.common.trainer" + job["args"])

# aggregate results
    from src.scripts.results import aggregate_results
    import pandas as pd
    import numpy as np

    def enumerate(xs, start=0, step=1):
        for x in xs:
            yield (start, x)
            start += step

    df = aggregate_results(args.logpath)
    df["mode"] = " "
    str2bool = {"True": True, "False": False}
    df.loc[df["npca"].map(str2bool), "mode"] = "npca"
    df.loc[df["pca"].map(str2bool), "mode"] = "pca"

    df = df.groupby(["arch", "mode", "variance_threshold", "category"]).mean()
    df = df.reset_index(level=(2, 3))
    df.to_csv("./b4_1_9_2_3_original.csv")

