import numpy as np

_dtype = np.dtype([("x", np.uint16), ("y", np.uint16), ("p", np.bool_), ("ts", np.uint64)])


class DVSSpikeTrain(np.recarray):
    """Common type for event based vision datasets"""

    __name__ = "SparseVisionSpikeTrain"

    def __new__(cls, nb_of_spikes, *args, width=-1, height=-1, duration=-1, time_scale=1e-6, **nargs):
        obj = super(DVSSpikeTrain, cls).__new__(cls, nb_of_spikes, dtype=_dtype, *args, **nargs)
        obj.width = width
        obj.height = height
        obj.duration = duration
        obj.time_scale = time_scale  # dt duration in seconds

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        self.width = getattr(obj, "width", None)
        self.height = getattr(obj, "height", None)
        self.duration = getattr(obj, "duration", None)
        self.time_scale = getattr(obj, "time_scale", None)

def readAERFile(filename: str) -> DVSSpikeTrain:
    """Function adapted from https://github.com/gorchard/event-Python/blob/master/eventvision.py
    for reading AER files from N-MNIST and N-Caltech 101
    """

    raw_data = np.fromfile(filename, dtype=np.uint8).astype(np.uint32)
    # print(f"J: {raw_data.shape}")

    all_x = raw_data[0::5]
    all_y = raw_data[1::5]
    all_p = np.right_shift(raw_data[2::5], 7)
    all_ts = (
        np.left_shift(raw_data[2::5] & 127, 16)
        | np.left_shift(raw_data[3::5], 8)
        | raw_data[4::5]
    )

    # Process time stamp overflow events
    time_increment = 2 ** 13
    overflow_indices = np.where(all_y == 240)[0]
    for overflow_index in overflow_indices:
        all_ts[overflow_index:] += time_increment

    # Everything else is a proper td spike
    td_indices = np.where(all_y != 240)[0]

    data = DVSSpikeTrain(td_indices.size)

    data.x = all_x[td_indices]
    data.y = all_y[td_indices]
    data.ts = all_ts[td_indices]
    data.p = all_p[td_indices]
    return data