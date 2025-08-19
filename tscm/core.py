# -*- coding: utf-8 -*-
from typing import Optional
import multiprocessing as mp
import warnings

import pandas as pd

from .sensation.local import LocalSensationCalculator
from .sensation.overall import OverallSensationCalculator
from .comfort.local import LocalComfortCalculator
from .comfort.overall import OverallComfortCalculator
from .config import HumanConfig, LocalSensationConfig, OverallSensationConfig, LocalComfortConfig, OverallComfortConfig


class LocalSensationModel:
    """A class for calculating local sensation based on skin temperature data."""
    def __init__(self,
                 skin_temperature: pd.DataFrame,
                 delta_skin_temperature: Optional[pd.DataFrame] = None,
                 delta_core_temperature: Optional[pd.Series] = None,
                 human_config: HumanConfig = HumanConfig(),
                 local_sensation_config: LocalSensationConfig = LocalSensationConfig()):
        """
        Args:
            skin_temperature:
                A dataframe of measured or simulated skin temperatures with timestamp.
            delta_skin_temperature:
                A series of derivative of skin temperatures with timestamp.
                Only needed and used for dynamic local sensation.
            delta_core_temperature:
                A series of derivative of core temperatures with timestamp.
                Only needed and used for dynamic local sensation.
            human_config:
                Configuration of human object. Refer to class HumanConfig.
            local_sensation_config:
                Configuration of local sensation calculation. Refer to class LocalSensationConfig.
        """
        self.human_config = human_config
        self.local_sensation_config = local_sensation_config

        self.skin_temperature = skin_temperature
        self.delta_skin_temperature = delta_skin_temperature
        self.delta_core_temperature = delta_core_temperature
        self.local_sensation = pd.DataFrame().reindex_like(self.skin_temperature)

        if self.local_sensation_config.dynamic:
            if self.delta_core_temperature is None or self.delta_skin_temperature is None:
                self.local_sensation_config.dynamic = False
                warnings.warn('No input of delta core/skin temperature.'
                              'Dynamic influence on local sensations will not be calculated', RuntimeWarning)

    def run(self, num_cores: int = 2):
        """
        Start local sensation calculation.
        Results are saved in attribute local_sensation.

        Args:
            num_cores:
                Number of cpu cores used to run calculation. Default is 2.
        """
        pool = mp.Pool(num_cores)
        q = mp.Manager().Queue()
        tasks = []
        for i in self.skin_temperature.index:
            task = pool.apply_async(self._sub_run, args=(i, q,))
            tasks.append(task)
        for task in tasks:
            task.get()
        pool.close()
        pool.join()

        for _ in self.skin_temperature.index:
            i, local_sensation_i = q.get()
            self.local_sensation.loc[i, :] = local_sensation_i

    def _sub_run(self, i, q):
        """
        Calculate local sensation for each iteration.

        Args:
            i: Index of current iteration.
            q: Multiprocessing queue to collect results.
        Returns:
             Results are put into queue including:
             i: Index of current iteration.
             local_sensation_i: A series of local sensations of current iteration.
        """
        local_sensation_model = LocalSensationCalculator(
            skin_temperature=self.skin_temperature.loc[i, :],
            delta_skin_temperature=self.delta_skin_temperature.loc[i, :],
            delta_core_temperature=self.delta_core_temperature.loc[i],
            human_config=self.human_config,
            local_sensation_config=self.local_sensation_config
        )
        local_sensation_i = local_sensation_model.local_sensation
        q.put((i, local_sensation_i))


class OverallSensationModel:
    """A class for calculating overall sensation based on local sensation data."""
    def __init__(self,
                 local_sensation: pd.DataFrame,
                 overall_sensation_config: OverallSensationConfig = OverallSensationConfig()):
        """
        Args:
            local_sensation:
                A dataframe of local sensations with timestamp.
            overall_sensation_config:
                Configuration of overall sensation calculation. Refer to class OverallSensationConfig.
        """
        self.local_sensation = local_sensation
        self.overall_sensation_config = overall_sensation_config
        self.overall_sensation = pd.Series(index=self.local_sensation.index)
        self.model_num = pd.Series(index=self.local_sensation.index)

    def run(self, num_cores: int = 2):
        """
        Start overall sensation calculation.
        Results are saved in attributes overall_sensation and model_num.

        Args:
            num_cores:
                Number of cpu cores used to run calculation. Default is 2.
        """
        pool = mp.Pool(num_cores)
        q = mp.Manager().Queue()
        tasks = []
        for i in self.local_sensation.index:
            task = pool.apply_async(self._sub_run, args=(i, q,))
            tasks.append(task)
        for task in tasks:
            task.get()
        pool.close()
        pool.join()

        for _ in self.local_sensation.index:
            i, overall_sensation_i, model_num_i = q.get()
            self.overall_sensation.loc[i] = overall_sensation_i
            self.model_num.loc[i] = model_num_i

    def _sub_run(self, i, q):
        """
        Calculate overall sensation for each iteration.

        Args:
            i: Index of current iteration.
            q: Multiprocessing queue to collect results.
        Returns:
             Results are put into queue including:
             i: Index of current iteration.
             overall_sensation_i: Value of overall sensation of current iteration.
             model_num_i: Model number of current iteration.
        """
        overall_sensation_model = OverallSensationCalculator(
            local_sensation=self.local_sensation.loc[i, :],
            overall_sensation_config=self.overall_sensation_config
        )
        overall_sensation_i = overall_sensation_model.overall_sensation
        model_num_i = overall_sensation_model.model_num
        q.put((i, overall_sensation_i, model_num_i))


class LocalComfortModel:
    """A class for calculating local comfort based on local sensation and overall sensation data."""
    def __init__(self,
                 local_sensation: pd.DataFrame,
                 overall_sensation: pd.Series,
                 local_comfort_config: LocalComfortConfig = LocalComfortConfig()):
        """
        Args:
            local_sensation:
                A dataframe of local sensations with timestamp.
            overall_sensation:
                A series of overall sensations with timestamp.
            local_comfort_config:
                Configuration of local comfort calculation. Refer to class LocalComfortConfig.
        """
        self.local_sensation = local_sensation
        self.overall_sensation = overall_sensation
        self.local_comfort_config = local_comfort_config
        self.local_comfort = pd.DataFrame().reindex_like(self.local_sensation)

    def run(self, num_cores: int = 2):
        """
        Start local comfort calculation.
        Results are saved in attribute local_comfort.

        Args:
            num_cores:
                Number of cpu cores used to run calculation. Default is 2.
        """
        pool = mp.Pool(num_cores)
        q = mp.Manager().Queue()
        tasks = []
        for i in self.local_sensation.index:
            task = pool.apply_async(self._sub_run, args=(i, q,))
            tasks.append(task)
        for task in tasks:
            task.get()
        pool.close()
        pool.join()

        for _ in self.local_sensation.index:
            i, local_comfort_i = q.get()
            self.local_comfort.loc[i, :] = local_comfort_i

    def _sub_run(self, i, q):
        """
        Calculate local comfort for each iteration.

        Args:
            i: Index of current iteration.
            q: Multiprocessing queue to collect results.
        Returns:
             Results are put into queue including:
             i: Index of current iteration.
             local_comfort_i: A series of local comfort of current iteration.
        """
        local_comfort_model = LocalComfortCalculator(
            local_sensation=self.local_sensation.loc[i, :],
            overall_sensation=self.overall_sensation.loc[i],
            local_comfort_config=self.local_comfort_config
        )
        local_comfort_i = local_comfort_model.local_comfort
        q.put((i, local_comfort_i))


class OverallComfortModel:
    """A class for calculating overall comfort based on local comfort data."""
    def __init__(self,
                 local_comfort: pd.DataFrame,
                 overall_comfort_config: OverallComfortConfig = OverallComfortConfig()):
        """
        Args:
            local_comfort:
                A dataframe of local comfort with timestamp.
            overall_comfort_config:
                Configuration of overall comfort calculation. Refer to class OverallComfortConfig.
        """
        self.local_comfort = local_comfort
        self.overall_comfort_config = overall_comfort_config
        self.overall_comfort = pd.Series(index=self.local_comfort.index)

    def run(self, num_cores: int = 2):
        """
        Start overall comfort calculation.
        Results are saved in attribute overall_comfort.

        Args:
            num_cores:
                Number of cpu cores used to run calculation. Default is 2.
        """
        pool = mp.Pool(num_cores)
        q = mp.Manager().Queue()
        tasks = []
        for i in self.local_comfort.index:
            task = pool.apply_async(self._sub_run, args=(i, q,))
            tasks.append(task)
        for task in tasks:
            task.get()
        pool.close()
        pool.join()

        for _ in self.local_comfort.index:
            i, overall_comfort_i = q.get()
            self.overall_comfort.loc[i] = overall_comfort_i

    def _sub_run(self, i, q):
        """
        Calculate overall comfort for each iteration.

        Args:
            i: Index of current iteration.
            q: Multiprocessing queue to collect results.
        Returns:
             Results are put into queue including:
             i: Index of current iteration.
             overall_comfort_i: Value of overall comfort of current iteration.
        """
        overall_comfort_model = OverallComfortCalculator(
            local_comfort=self.local_comfort.loc[i, :],
            overall_comfort_config=self.overall_comfort_config
        )
        overall_comfort_i = overall_comfort_model.overall_comfort
        q.put((i, overall_comfort_i))

