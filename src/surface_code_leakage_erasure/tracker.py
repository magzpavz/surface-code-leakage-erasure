from typing import Any
from collections import defaultdict

class MeasurementTracker:
    # tracks which measurement happened on which item
    # used when building circuit to construct detectors
    def __init__(self):
        self.measurement_list = []
        self.history: defaultdict[int, list[int]] = defaultdict(list)
        self.t = 0
        self.rounds = 0

    def __eq__(self, other):
        return (type(self) == type(other) and self.t == other.t and
                self.history == other.history and self.rounds == other.rounds)

    def __repr__(self):
       return f"MeasurementTracker(t={self.t}, rounds={self.rounds})"

    def add_round(self):
        self.rounds += 1

    def add_measurement(self, item: int):
        self.measurement_list.append(item)
        self.history[item].append(self.t)
        self.t += 1

    def add_measurement_list(self, item_list : list[int]):
        self.measurement_list.extend(item_list)
        history = self.history  # local ref
        t = self.t
        # Pre-compute time indices
        time_indices = range(t, t + len(item_list))
        for item, time_idx in zip(item_list, time_indices):
            history[item].append(time_idx)
        self.t = t + len(item_list)

    def get_curr_meas(self, item: int):
        return self.history[item][-1] - self.t

    def get_prev_meas(self, item: int):
        return self.history[item][-2] - self.t

    def get_all_meas(self, item: int):
        return [idx - self.t for idx in self.history[item]]
