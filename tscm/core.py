#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023/9/28 18:45
# @Author  : Eric
from typing import Optional
import multiprocessing as mp
import warnings

import pandas as pd

from tscm.local_sensation import SkinTemperatureProcessor
from tscm.whole_sensation import LocalSensationProcessor
from tscm.config import HumanConfig, LocalSensationConfig, WholeSensationConfig


class TSCMObject:
    """
    Object for thermal sensation model.

    Attributes:
        local_sensation:
            A dataframe of results of local sensation.
        whole_sensation:
            A series of results of whole body sensation.
        model_num:
            A series of model numbers.
    """
    def __init__(self, skin_temperature: pd.DataFrame,
                 delta_skin_temperature: Optional[pd.DataFrame] = None,
                 delta_core_temperature: Optional[pd.Series] = None,
                 human_config: HumanConfig = HumanConfig(),
                 local_sensation_config: LocalSensationConfig = LocalSensationConfig(),
                 whole_sensation_config: WholeSensationConfig = WholeSensationConfig()):
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
            whole_sensation_config:
                Configuration of whole sensation calculation. Refer to class WholeSensationConfig.
        """

        self.human_config = human_config
        self.local_sensation_config = local_sensation_config
        self.whole_sensation_config = whole_sensation_config

        self.skin_temperature = skin_temperature
        self.delta_skin_temperature = delta_skin_temperature
        self.delta_core_temperature = delta_core_temperature

        self.local_sensation = None
        self.whole_sensation = None
        self.model_num = None

    def run(self, num_cores: int = 2):
        """
        Start local and whole-body sensation calculation.
        Results are saved in attributes local_sensation and whole_sensation.

        Args:
            num_cores:
                Number of cpu cores used to run calculation. Default is 2.
        """
        if self.local_sensation_config.dynamic:
            if self.delta_core_temperature is None or self.delta_skin_temperature is None:
                self.local_sensation_config.dynamic = False
                warnings.warn('No input of delta core/skin temperature.'
                              'Dynamic sensation will not be calculated', RuntimeWarning)

        self.local_sensation = pd.DataFrame().reindex_like(self.skin_temperature)
        self.whole_sensation = pd.Series(index=self.skin_temperature.index)
        self.model_num = pd.Series(index=self.skin_temperature.index)

        pool = mp.Pool(num_cores)
        q = mp.Manager().Queue()
        tasks = []
        for i in self.skin_temperature.index:
            t = pool.apply_async(self.sub_run, args=(i, q,))
            tasks.append(t)
        for t in tasks:
            t.get()  # raise error caused by subprocess
        pool.close()
        pool.join()
        for _ in self.skin_temperature.index:
            i, local_sensation_i, whole_sensation_i, model_num_i = q.get()
            self.local_sensation.loc[i, :] = local_sensation_i
            self.whole_sensation.loc[i] = whole_sensation_i
            self.model_num.loc[i] = model_num_i

    def sub_run(self, i, q):
        """
        Calculate local and whole-body sensation for each iteration.

        Args:
            i: Index of current iteration.
            q: Multiprocessing queue to collect results.
        Returns:
             Results are put into queue including:
             i: Index of current iteration.
             local_sensation_: A series of local sensations of current iteration.
             whole_sensation_: Value of whole-body sensation of current iteration.
             model_num_i: Model number of current iteration.
        """
        if self.local_sensation_config.dynamic:
            delta_skin_temperature_ = self.delta_skin_temperature.loc[i, :]
            delta_core_temperature_ = self.delta_core_temperature.loc[i]
        else:  # static sensation
            delta_skin_temperature_, delta_core_temperature_ = [None] * 2

        local_sensation_ = SkinTemperatureProcessor(skin_temperature=self.skin_temperature.loc[i, :],
                                                    delta_skin_temperature=delta_skin_temperature_,
                                                    delta_core_temperature=delta_core_temperature_,
                                                    human_config=self.human_config,
                                                    local_sensation_config=self.local_sensation_config
                                                    ).get_local_sensation()

        whole_sensation_model = LocalSensationProcessor(local_sensation=local_sensation_,
                                                        whole_sensation_config=self.whole_sensation_config)
        model_num_i = whole_sensation_model.model_num
        whole_sensation_ = whole_sensation_model.get_whole_sensation()

        q.put((i, local_sensation_, whole_sensation_, model_num_i))
