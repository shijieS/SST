import os
import pandas as pd
from pathlib import Path

def mot17_post_processing(file_name):
    """
    MOT17 Post Processing contains:

    - Remove small tracks.
    - Merge Tracks.

    Args:
        file_name: The output result files

    Returns: None

    """
    data = pd.read_csv(file_name, header=None, sep=' ', dtype=int)

    data_group = data.groupby(1)
    select_track = []
    i = 1
    for track_id, frame_boxes in data_group:
        if len(frame_boxes) > 10:
            select_track += [frame_boxes]
            frame_boxes.loc[:, 1] = i
            frame_boxes.loc[:, 0] = frame_boxes.loc[:, 0]
            i += 1

    data_selected = pd.concat(select_track)

    # save_path = os.path.join(
    #     *Path(file_name).parts[:-1],
    #     os.path.splitext(Path(file_name).parts[-1])[0]
    #     +'_converted'
    #     +os.path.splitext(Path(file_name).parts[-1])[1]
    # )

    data_selected.sort_values(0).to_csv(path_or_buf=file_name, index=False, sep=' ', header=False)

    return file_name



if __name__ == "__main__":
    mot17_post_processing('result/Demo/MOT17-10-FRCNN.txt')