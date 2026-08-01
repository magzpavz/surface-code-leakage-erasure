def describe_detector(circuit, detector_idx: int):
    """ Given a stim circuit and a detector index, return the list of measurements that are compared to determine if the detector is triggered.

    Args:
        circuit: a stim.Circuit object
        detector_idx: the index of the detector to be described. The first detector has index 0.

    Returns:
        A list of tuples, (qubit index, round number), for each measurement that is compared to determine if the detector is triggered. Round number is determined by the number of resets in the circuit before the measurement.
    """

    round_count = -1
    detector_count = -1
    just_reset = False
    measurement_list = []
    for line in circuit:
        if line.name == 'R' and not just_reset:
            # avoid double-counting when there are consecutive resets
            round_count += 1
            just_reset = True
            continue
        elif line.name == 'DETECTOR':
            detector_count += 1
            if detector_count == detector_idx:
                compared_measurements =[
                    measurement_list[targ.value]
                    for targ in line.targets_copy()]
                break
        elif line.name == "M":
            measurement_list.extend(
                (targ.value, round_count) for targ in line.targets_copy())

        just_reset = False

    return compared_measurements