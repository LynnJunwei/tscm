import pandas as pd
import numpy as np

from tscm.config import OverallComfortConfig


class OverallComfortCalculator:
    """
    A class for calculating overall comfort based on local comfort.
    """
    def __init__(self,
                 local_comfort: pd.Series,
                 overall_comfort_config: OverallComfortConfig = OverallComfortConfig()):
        """
        Args:
            local_comfort: A series of local comfort.
            overall_comfort_config: Configuration of overall comfort calculation. Refer to class OverallComfortConfig.
        """
        self.local_comfort = local_comfort
        self.config = overall_comfort_config

    def get_overall_comfort(self) -> float:
        """
        Calculate the overall comfort based on local comfort.

        Returns:
            A float of overall comfort.
        """

        local_comfort_sorted = self.local_comfort.sort_values(ascending=True)
        if self.config.is_controlled or self.config.is_transient:
            if self.are_hands_feet_lowest():
                overall_comfort = np.mean(local_comfort_sorted.iloc[0] +
                                          local_comfort_sorted.iloc[2] +
                                          local_comfort_sorted.iloc[-1])
            else:
                overall_comfort = np.mean(local_comfort_sorted.iloc[0] +
                                          local_comfort_sorted.iloc[1] +
                                          local_comfort_sorted.iloc[-1])
        else:
            if self.are_hands_feet_lowest():
                overall_comfort = np.mean(local_comfort_sorted.iloc[0] +
                                          local_comfort_sorted.iloc[2])
            else:
                overall_comfort = np.mean(local_comfort_sorted.iloc[0] +
                                          local_comfort_sorted.iloc[1])

        return overall_comfort

    def are_hands_feet_lowest(self) -> bool:
        """
        Determine if the most extreme comfort in a given sorted comfort series are from the hands or feet.
        """
        if len(self.local_comfort) < 2:
            return False
        local_comfort_sorted = self.local_comfort.sort_values(ascending=True)

        body_parts_sorted = local_comfort_sorted.index
        are_hands_most_lowest = body_parts_sorted[0].endswith("Hand") and body_parts_sorted[1].endswith("Hand")
        are_feet_most_lowest = body_parts_sorted[0].endswith("Foot") and body_parts_sorted[1].endswith("Foot")
        return are_hands_most_lowest or are_feet_most_lowest

    @property
    def overall_comfort(self):
        return self.get_overall_comfort()