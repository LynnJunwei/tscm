# -*- coding: utf-8 -*-
from typing import Literal
import pandas as pd


def get_bigger_group(local_sensation: pd.Series) -> Literal["warm", "cold"]:
    """
    Determine which sensation group (warm or cold / positive or negative) is bigger group. The group including
    larger number of sensations will be considered as bigger group. If the numbers are the same, bigger group is
    warm / positive sensation group.

    Returns:
        Name of bigger group which is either "warm" or "cold".
    """
    return "warm" if sum(local_sensation.gt(0)) >= sum(local_sensation.lt(0)) else "cold"


def get_sensation_level(local_sensation: pd.Series, bigger_group: Literal["warm", "cold"]) -> Literal["high", "low"]:
    """
    Determine the level of sensation for input local sensation group.

    Args:
        local_sensation : A series containing thermal sensations for various body parts.
        bigger_group : Name of bigger group which is either "warm" or "cold".

    Returns:
        The level of sensation for input local sensation group.
    """
    if bigger_group == "warm":
        local_sensation_descending = local_sensation.sort_values(ascending=False)
        second_max_index = 1 if not are_hands_feet_most_extreme(local_sensation_descending) else 2
        second_max = local_sensation_descending.iloc[second_max_index]
        return 'high' if second_max >= 2 else 'low'
    if bigger_group == "cold":
        local_sensation_ascending = local_sensation.sort_values(ascending=True)
        second_min_index = 1 if not are_hands_feet_most_extreme(local_sensation_ascending) else 2
        second_min = local_sensation_ascending.iloc[second_min_index]
        return 'high' if second_min <= -2 else 'low'


def are_hands_feet_most_extreme(local_sensation_sorted: pd.Series) -> bool:
    """
    Determine if the most extreme sensations in a given sorted sensation series are from the hands or feet.

    This function examines the input series of sensations to determine if the most extreme sensations
    come from the hands or feet. The input series should be sorted in advanced.
    For sensations in ascending order, this function will determine whether hands or feet sensations are
    maximum. For those in descending order, it will determine whether hands or feet sensations are minimum.

    Args:
        local_sensation_sorted : A series containing sorted thermal sensations for various body parts.

    Returns:
        Return True if the most extreme sensations (either the two largest or two smallest)
        correspond to both hands or both feet. Otherwise, return False.
    """
    if len(local_sensation_sorted) < 2:
        return False

    body_parts_sorted = local_sensation_sorted.index
    are_hands_most_extreme = body_parts_sorted[0].endswith("Hand") and body_parts_sorted[1].endswith("Hand")
    are_feet_most_extreme = body_parts_sorted[0].endswith("Foot") and body_parts_sorted[1].endswith("Foot")
    return are_hands_most_extreme or are_feet_most_extreme

