#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023/9/28 18:45
# @Author  : Eric
from typing import Optional, Literal
import multiprocessing as mp
import warnings

import pandas as pd

from tscm.local_sensation import LocalSensationCalculator
from tscm.overall_sensation import OverallSensationCalculator
from tscm.config import HumanConfig, LocalSensationConfig, OverallSensationConfig


class TSCMObject:
    """
    Object for thermal local_sensation_sorted model.

    Attributes:
        local_sensation:
            A dataframe of results of local local_sensation_sorted.
        overall_sensation:
            A series of results of overall local_sensation_sorted.
        model_num:
            A series of model numbers.
    """
    def __init__(self,
                 skin_temperature: Optional[pd.DataFrame] = None,
                 delta_skin_temperature: Optional[pd.DataFrame] = None,
                 delta_core_temperature: Optional[pd.Series] = None,
                 local_sensation: Optional[pd.DataFrame] = None,
                 human_config: HumanConfig = HumanConfig(),
                 local_sensation_config: LocalSensationConfig = LocalSensationConfig(),
                 overall_sensation_config: OverallSensationConfig = OverallSensationConfig(),
                 output: Literal['all', 'ls', 'os'] = 'all'):
        """
        Args:
            skin_temperature:
                A dataframe of measured or simulated skin temperatures with timestamp.
            delta_skin_temperature:
                A series of derivative of skin temperatures with timestamp.
                Only needed and used for dynamic local local_sensation_sorted.
            delta_core_temperature:
                A series of derivative of core temperatures with timestamp.
                Only needed and used for dynamic local local_sensation_sorted.
            human_config:
                Configuration of human object. Refer to class HumanConfig.
            local_sensation_config:
                Configuration of local sensation calculation. Refer to class LocalSensationConfig.
            overall_sensation_config:
                Configuration of overall sensation calculation. Refer to class OverallSensationConfig.
        """

        self.human_config = human_config
        self.local_sensation_config = local_sensation_config
        self.overall_sensation_config = overall_sensation_config

        self.skin_temperature = skin_temperature
        self.delta_skin_temperature = delta_skin_temperature
        self.delta_core_temperature = delta_core_temperature

        self.local_sensation = local_sensation
        self.overall_sensation = None
        self.model_num = None

        # output
        self.output = output
        self._input = None
        if self.skin_temperature is None and self.local_sensation is None:
            raise ValueError('No input of skin temperature data or local sensation data.')
        if self.skin_temperature is None and self.output == 'ls':
            raise ValueError('No input of skin temperature data.')

        if self.skin_temperature is None and self.local_sensation is not None:
            self.index = self.local_sensation.index
            self.input = 'ls'
        elif self.skin_temperature is not None and self.local_sensation is None:
            self.index = self.skin_temperature.index
            self.local_sensation = pd.DataFrame().reindex_like(self.skin_temperature)
            self.input = 'tsk'
        elif self.skin_temperature is not None and self.local_sensation is not None:
            warnings.warn('Both skin temperature and local sensation are input.'
                          'Follow-up calculation will be based on local sensation data.', RuntimeWarning)
            self.index = self.local_sensation.index
            self.input = 'ls'

        self.overall_sensation = pd.Series(index=self.index)
        self.model_num = pd.Series(index=self.index)

        # dynamic
        if self.local_sensation_config.dynamic:
            if self.delta_core_temperature is None or self.delta_skin_temperature is None:
                self.local_sensation_config.dynamic = False
                warnings.warn('No input of delta core/skin temperature.'
                              'Dynamic influence on local sensations will not be calculated', RuntimeWarning)


    def run(self, num_cores: int = 2):
        """
        Start local and overall sensation calculation.
        Results are saved in attributes local_sensation_sorted and baseline_sensation.

        Args:
            num_cores:
                Number of cpu cores used to run calculation. Default is 2.
        """
        pool = mp.Pool(num_cores)
        q = mp.Manager().Queue()
        tasks = []
        for i in self.index:
            task = pool.apply_async(self.sub_run, args=(i, q,))
            tasks.append(task)
        for task in tasks:
            task.get()  # raise error caused by subprocess
        pool.close()
        pool.join()
        for _ in self.index:
            i, local_sensation_i, overall_sensation_i, model_num_i = q.get()
            self.local_sensation.loc[i, :] = local_sensation_i
            self.overall_sensation.loc[i] = overall_sensation_i
            self.model_num.loc[i] = model_num_i

    def sub_run(self, i, q):
        """
        Calculate local and overall sensation for each iteration.

        Args:
            i: Index of current iteration.
            q: Multiprocessing queue to collect results.
        Returns:
             Results are put into queue including:
             i: Index of current iteration.
             local_sensation_: A series of local sensations of current iteration.
             overall_sensation_: Value of overall sensation of current iteration.
             model_num_i: Model number of current iteration.
        """
        if self.local_sensation_config.dynamic:
            delta_skin_temperature_ = self.delta_skin_temperature.loc[i, :]
            delta_core_temperature_ = self.delta_core_temperature.loc[i]
        else:  # static local_sensation_sorted
            delta_skin_temperature_, delta_core_temperature_ = [None] * 2

        if self.output == 'ls':
            local_sensation_model = LocalSensationCalculator(skin_temperature=self.skin_temperature.loc[i, :],
                                                             delta_skin_temperature=delta_skin_temperature_,
                                                             delta_core_temperature=delta_core_temperature_,
                                                             human_config=self.human_config,
                                                             local_sensation_config=self.local_sensation_config)
            local_sensation_i = local_sensation_model.local_sensation
            model_num_i = None
            overall_sensation_i = None

        elif self.output == 'os':
            if self.input == 'tsk':
                local_sensation_model = LocalSensationCalculator(skin_temperature=self.skin_temperature.loc[i, :],
                                                                 delta_skin_temperature=delta_skin_temperature_,
                                                                 delta_core_temperature=delta_core_temperature_,
                                                                 human_config=self.human_config,
                                                                 local_sensation_config=self.local_sensation_config)
                local_sensation_i = local_sensation_model.local_sensation
            else:
                local_sensation_i = self.local_sensation.loc[i, :]

            overall_sensation_model = OverallSensationCalculator(local_sensation=local_sensation_i,
                                                             overall_sensation_config=self.overall_sensation_config)
            model_num_i = overall_sensation_model.model_num
            overall_sensation_i = overall_sensation_model.overall_sensation

        else:  # elif self.output == 'all':
            if self.input == 'tsk':
                local_sensation_model = LocalSensationCalculator(skin_temperature=self.skin_temperature.loc[i, :],
                                                                 delta_skin_temperature=delta_skin_temperature_,
                                                                 delta_core_temperature=delta_core_temperature_,
                                                                 human_config=self.human_config,
                                                                 local_sensation_config=self.local_sensation_config)
                local_sensation_i = local_sensation_model.local_sensation
            else:
                local_sensation_i = self.local_sensation.loc[i, :]

            overall_sensation_model = OverallSensationCalculator(local_sensation=local_sensation_i,
                                                             overall_sensation_config=self.overall_sensation_config)
            model_num_i = overall_sensation_model.model_num
            overall_sensation_i = overall_sensation_model.overall_sensation

        q.put((i, local_sensation_i, overall_sensation_i, model_num_i))
