#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023/11/1 18:13
# @Author  : Eric
from typing import Literal

import pandas as pd
import numpy as np

from tscm.const import COEFFICIENT
from tscm.config import WholeSensationConfig


class SensationModel:
    """
    A base class of whole-body local_sensation_sorted calculation model.

    Attributes:
        bigger_group:
            A string that is either 'warm' or 'cool'. Represent whether more local sensations are in warm side or in
            cool side.
        is_cool_dominated:
            A boolean represents whether whole-body local_sensation_sorted is dominated by cool or cold feeling in dominant body
            parts ('Chest', 'Back', and 'Pelvis').  It will be True when bigger group is cool and all opposite
            sensations are smaller than or equal to 1.
        is_no_opposite:
            A boolean represents whether the input set of local sensations should use no opposite calculation model.
    """
    dominant_parts = None  # value will be rewritten by child class LocalSensationProcessor

    def __init__(self, local_sensation: pd.Series):
        """
        Args:
            local_sensation: A series of local local_sensation_sorted.
        """
        self.local_sensation = local_sensation
        self.body_parts = self.local_sensation.index
        self._body_part_num = len(self.body_parts)

        self._local_sensation_sorted = None
        self._body_parts_sorted = None

        self._max = None
        self._min = None
        self._second_max = None
        self._second_min = None
        self._third_max = None
        self._third_min = None

        self._dominant_parts = SensationModel.dominant_parts
        self._min_dominant_part = None
        self.is_cool_dominated = None

        self.bigger_group = None
        self.is_no_opposite = None

        if self._body_part_num != 0:
            self._initialize_sorted_sensations()
            self._initialize_extreme_values()
            self._initialize_dominated_state()
            self._initialize_opposite_state()

    def _initialize_sorted_sensations(self):
        _local_sensation_sorted = sorted(zip(self.local_sensation, self.body_parts), reverse=True)
        self._local_sensation_sorted = [sensation for sensation, _ in _local_sensation_sorted]
        self._body_parts_sorted = [body_part for _, body_part in _local_sensation_sorted]

    def _initialize_extreme_values(self):
        self._max = self._local_sensation_sorted[0]
        self._min = self._local_sensation_sorted[-1]
        self._second_max = self._local_sensation_sorted[1] if self._body_part_num > 1 else None
        self._second_min = self._local_sensation_sorted[-2] if self._body_part_num > 1 else None
        self._third_max = self._local_sensation_sorted[2] if self._body_part_num > 2 else None
        self._third_min = self._local_sensation_sorted[-3] if self._body_part_num > 2 else None

    def _initialize_dominated_state(self):
        self._dominant_parts = [body_part for body_part in self.body_parts if body_part in self._dominant_parts]
        self._min_dominant_part = min(self.local_sensation[self._dominant_parts]) if self._dominant_parts else 0
        self.is_cool_dominated = True if self._min_dominant_part <= -1 else False

    def _initialize_opposite_state(self):
        if sum(self.local_sensation.gt(0)) >= sum(self.local_sensation.lt(0)):
            self.bigger_group = 'warm'
            self.is_no_opposite = True if self._min >= -1 and not self.is_cool_dominated else False
        else:
            self.bigger_group = 'cool'
            self.is_no_opposite = True if self._max <= 1 else False

    def _is_hands_feet_extreme(self, side: Literal['warm', 'cool']) -> bool:
        """
        Justify whether the most and second most extreme local sensations are in two hands or two feet.

        Args:
            side:
                A string indicates which extreme side used in the justification.
                The value should be either 'warm' or 'cool'.

        Returns:
            Return True if sensations in two hands or two feet are the most and second most extreme.
            Return False if not.
        """
        body_parts_sorted = self._body_parts_sorted if side == 'warm' else self._body_parts_sorted[::-1]

        is_hands_extreme = body_parts_sorted[0].endswith('Hand') and body_parts_sorted[1].endswith('Hand')
        is_feet_extreme = body_parts_sorted[0].endswith('Foot') and body_parts_sorted[1].endswith('Foot')
        return is_hands_extreme or is_feet_extreme

    def get_model_num(self) -> int:
        """
        Returns the number of which calculation model should be used for the input set of local sensations.

        The calculation models and corresponding numbers are as follows:

        *1: No-opposite high level warm (complaint warm)
        *2: No-opposite high level cool (complaint cool)
        *3: No-opposite low level warm (gradual warm)
        *4: No-opposite low level cool (gradual cool)
        *5: Opposite dominated cold
        *6: Opposite warm
        *7: Opposite cool
        """
        if not self.is_no_opposite and self.bigger_group == 'cool':
            return 7
        if not self.is_no_opposite and self.is_cool_dominated:
            return 5
        if not self.is_no_opposite and self.bigger_group == 'warm':
            return 6

        if self.is_no_opposite and self.bigger_group == 'warm':
            criterion = self._second_max if not self._is_hands_feet_extreme('warm') else self._third_max
            return 1 if criterion >= 2 else 3

        if self.is_no_opposite and self.bigger_group == 'cool':
            criterion = self._second_min if not self._is_hands_feet_extreme('cool') else self._third_min
            return 2 if criterion <= -2 else 4


def sig(x: float, a: int, t: float) -> float: return 1 / (1 + np.exp(-a * (x - t)))


def _opposite_modifier(local_sensation_smaller, overall_sensation) -> float:
        """Returns a float of combined force as modifier for opposite warm and cool models."""
        coeff_df = coefficient_ucb.loc['a_dS_-2':'c_dS_2', :]

        individual_force = pd.Series(index=local_sensation_smaller.index)
        for body_part in local_sensation_smaller.index:
            delta_sensation = local_sensation_smaller[body_part] - overall_sensation
            if delta_sensation <= -2:
                a, b, c = coeff_df.loc[['a_dS_-2', 'b_dS_-2', 'c_dS_-2'], body_part].tolist()
            elif -2 < delta_sensation < 2:
                a, b, c = coeff_df.loc[['a_dS_-2_2', 'b_dS_-2_2', 'c_dS_-2_2'], body_part].tolist()
            else:  # delta_sensation >= 2
                a, b, c = coeff_df.loc[['a_dS_2', 'b_dS_2', 'c_dS_2'], body_part].tolist()

            individual_force[body_part] = a * (delta_sensation - c) + b

        individual_force_sorted = sorted(individual_force, key=abs, reverse=True)
        if len(individual_force_sorted) == 0:
            return 0
        if len(individual_force_sorted) == 1:
            combined_force = float(individual_force_sorted[0])
        else:
            combined_force = float(individual_force_sorted[0]) + 0.1 * float(individual_force_sorted[1])

        extreme_sensation_abs = abs(sorted(local_sensation_smaller, key=abs, reverse=True)[0])
        if extreme_sensation_abs < 1:
            return 0
        if 1 <= extreme_sensation_abs < 2:
            # return combined_force
            return (extreme_sensation_abs-1) * combined_force
            # return sig(extreme_sensation_abs - 1, 10, 0.5) * combined_force
        if extreme_sensation_abs >= 2:
            return combined_force


class NoOppositeModel(SensationModel):
    def __init__(self, local_sensation):
        super().__init__(local_sensation)

    def complaint_model_warm(self) -> float:
        """Returns the whole-body local_sensation_sorted calculated by no-opposite high level warm (complaint warm) model."""
        if self._is_hands_feet_extreme('warm'):
            local = 0.5 * self._max + 0.5 * self._third_max
        else:
            local = 0.5 * self._max + 0.5 * self._second_max
        modifier = _opposite_modifier(self.local_sensation[self.local_sensation >= local], local)
        modifier += _opposite_modifier(self.local_sensation[self.local_sensation <= 0], np.mean(self.local_sensation)+modifier)
        if self._min_dominant_part <= 0:
            m = _opposite_modifier(self.local_sensation[self.local_sensation >= 0], self._min_dominant_part)
            # return local + min([abs(self._min_dominant_part), 1]) * (self._min_dominant_part - local + m)
            return local + modifier
        else:
            return local + modifier

    def complaint_model_cool(self) -> float:
        """Returns the whole-body local_sensation_sorted calculated by no-opposite high level cool (complaint cool) model."""
        if self._min_dominant_part == self._min:
            local = self._min_dominant_part
        elif self._is_hands_feet_extreme('cool'):
            local = 0.38 * self._min + 0.62 * self._third_min
        else:
            local = 0.38 * self._min + 0.62 * self._second_min

        modifier = _opposite_modifier(self.local_sensation[self.local_sensation >= 0], np.mean(self.local_sensation))
        modifier += _opposite_modifier(self.local_sensation[self.local_sensation <= local+modifier], local+modifier)

        return local + modifier

    def _body_part_num_interval(self) -> int:
        """
        Calculate adjusted number of body parts for interval calculation.

        Returns: An int of number of adjusted body part.
        """
        hands_num = len([body_part for body_part in self.body_parts if body_part.endswith('Hand')])
        feet_num = len([body_part for body_part in self.body_parts if body_part.endswith('Foot')])

        if hands_num == 2 and feet_num == 2:
            return self._body_part_num - 2
        elif hands_num == 2 or feet_num == 2:
            return self._body_part_num - 1
        else:
            return self._body_part_num

    def gradual_model_warm(self, sensation_type='all') -> float:
        """Returns the whole-body local_sensation_sorted calculated by no-opposite low level warm (gradual warm) model."""
        interval = 2 / self._body_part_num_interval()
        local_sensation_sorted = self._local_sensation_sorted

        if self._is_hands_feet_extreme('warm'):
            del local_sensation_sorted[1]

        local_sensation_selected = local_sensation_sorted[:2]  # first and second as initial
        for i in range(2, len(local_sensation_sorted)):
            local_sensation_selected.append(local_sensation_sorted[i])
            if local_sensation_sorted[i] > 2 - interval * (i - 1):
                break  # stop when local_sensation_sorted which meets the condition

        overall = np.mean(local_sensation_selected)
        # overall = np.mean([i if i>=-1 else -1 for i in local_sensation_selected])

        # modifier = _opposite_modifier(self.local_sensation_sorted[self.local_sensation_sorted >= overall], overall)
        # modifier += _opposite_modifier(self.local_sensation_sorted[self.local_sensation_sorted <= 0], np.mean(local_sensation_sorted)+modifier)

        modifier = _opposite_modifier(self.local_sensation[self.local_sensation <= 0], np.mean(local_sensation_sorted))
        modifier += _opposite_modifier(self.local_sensation[self.local_sensation >= overall+modifier], overall+modifier)

        # modifier = _opposite_modifier(self.local_sensation_sorted[self.local_sensation_sorted <= 0], overall)
        # modifier += _opposite_modifier(self.local_sensation_sorted[self.local_sensation_sorted >= overall], overall)



        '''if self._min_dominant_part <= 0:
            m = _opposite_modifier(self.local_sensation_sorted[self.local_sensation_sorted >= 0], self._min_dominant_part)
            modifier += sig(abs(self._min_dominant_part), 30, 0.5) * (self._min_dominant_part - np.mean(local_sensation_selected) - modifier + m)'''
        '''if local_sensation_sorted[1] >= 1:
            modifier += min([local_sensation_sorted[1]-1, 1]) * (np.mean(local_sensation_sorted[:2]) - np.mean(local_sensation_selected) - modifier)'''

        return overall + modifier if sensation_type != 'no' else np.mean(local_sensation_selected)

    def gradual_model_cool(self, sensation_type='all') -> float:
        """Returns the whole-body local_sensation_sorted calculated by no-opposite low level cool (gradual cool) model."""
        interval = 2 / self._body_part_num_interval()
        local_sensation_sorted = self._local_sensation_sorted[::-1]

        if self._is_hands_feet_extreme('cool'):
            del local_sensation_sorted[1]

        local_sensation_selected = local_sensation_sorted[:2]  # first and second as initial
        for i in range(2, len(local_sensation_sorted)):
            local_sensation_selected.append(local_sensation_sorted[i])
            if local_sensation_sorted[i] < -2 + interval * (i - 1):
                break  # stop when local_sensation_sorted which meets the condition

        overall = np.mean([i if i <= 1 else 1 for i in local_sensation_selected])
        if self._min_dominant_part <= -1:
            overall = min([np.mean(local_sensation_selected), self._min_dominant_part])
        '''elif -1 < self._min_dominant_part <= 0:
            if np.mean(local_sensation_selected) > self._min_dominant_part:
                overall = (np.mean(local_sensation_selected)
                           + sig(abs(self._min_dominant_part), 10, 0.5) *
                           (self._min_dominant_part - np.mean(local_sensation_selected)))
            else:
                overall = np.mean(local_sensation_selected)'''
        '''else:
            overall = np.mean(local_sensation_selected)'''

        modifier = _opposite_modifier(self.local_sensation[self.local_sensation >= 0], np.mean(local_sensation_sorted))
        modifier += _opposite_modifier(self.local_sensation[self.local_sensation <= overall+modifier], overall+modifier)

        # modifier = _opposite_modifier(self.local_sensation_sorted[self.local_sensation_sorted <= overall], overall)
        # modifier += _opposite_modifier(self.local_sensation_sorted[self.local_sensation_sorted >= 0], overall)


        '''if local_sensation_sorted[1] <= -1:
            modifier += (min([abs(local_sensation_sorted[1])-1, 1]) *
                         (0.38 * local_sensation_sorted[0] + 0.62 * local_sensation_sorted[1] - overall - modifier))'''

        return overall + modifier if sensation_type != 'no' else overall

    def no_opposite_model(self, bigger_group=None, sensation_type='all') -> float:
        """Returns the whole-body local_sensation_sorted calculated by no-opposite model."""
        if len(self.local_sensation) == 0:
            return 0
        if len(self.local_sensation) == 1:
            return self.local_sensation.values[0]
        if len(self.local_sensation) == 2 and self._is_hands_feet_extreme('warm'):
            return self.local_sensation.mean()

        if not bigger_group:
            bigger_group = self.bigger_group

        if bigger_group == 'warm':
            criterion = self._second_max if not self._is_hands_feet_extreme('warm') else self._third_max
            return self.complaint_model_warm() if criterion >= 2 else self.gradual_model_warm(sensation_type)

        if bigger_group == 'cool':
            criterion = self._second_min if not self._is_hands_feet_extreme('cool') else self._third_min
            return self.complaint_model_cool() if criterion <= -2 else self.gradual_model_cool(sensation_type)


class OppositeModel(SensationModel):
    def __init__(self, local_sensation):
        super().__init__(local_sensation)
        self._local_sensation_warm = self.local_sensation[self.local_sensation > 0]
        self._local_sensation_cool = self.local_sensation[self.local_sensation < 0]
        # self._local_sensation_warm_neutral = self.local_sensation_sorted[self.local_sensation_sorted >= 0]
        self._local_sensation_warm_neutral = self.local_sensation.where(self.local_sensation >= 0, 0)
        # self._local_sensation_cool_neutral = self.local_sensation_sorted[self.local_sensation_sorted <= 0]
        self._local_sensation_cool_neutral = self.local_sensation.where(self.local_sensation <= 0, 0)
        # self._local_sensation_cool_neutral_ex = self.local_sensation_sorted[self.local_sensation_sorted <= 1]
        self._local_sensation_cool_neutral_ex = self.local_sensation.where(self.local_sensation <= 1, 1)
        # self._local_sensation_warm_neutral_ex = self.local_sensation_sorted[self.local_sensation_sorted >= -1]
        self._local_sensation_warm_neutral_ex = self.local_sensation.where(self.local_sensation >= -1, -1)

        self._local_sensation_neutral = self.local_sensation.where(self.local_sensation >= -1, -1)
        self._local_sensation_neutral = self._local_sensation_neutral.where(self.local_sensation <= 1, 1)
        self._local_sensation_neutral = np.mean(self._local_sensation_neutral)

    def opposite_dominated_model(self) -> float:
        """
        Returns the whole-body local_sensation_sorted calculated by opposite dominated cold model.
        When bigger group is in cool side, this model will not be used.
        """
        lower_dominant = self.local_sensation[self.local_sensation <= 0]
        lower_sens = NoOppositeModel(self.local_sensation).no_opposite_model('cool', 'all')
        modifier = _opposite_modifier(self._local_sensation_cool, np.mean(self.local_sensation))
        # modifier += _opposite_modifier(lower_dominant, lower_sens)
        return lower_sens

    def opposite_warm(self) -> float:
        """Returns the whole-body local_sensation_sorted calculated by opposite warm model."""
        overall_sensation_bigger_ex = NoOppositeModel(self.local_sensation).no_opposite_model('warm', 'all')
        # overall_sensation_bigger = NoOppositeModel(self._local_sensation_warm_neutral).no_opposite_model('warm', 'all')
        # overall_sensation_bigger = np.mean(self.local_sensation_sorted)
        # modifier = _opposite_modifier(self._local_sensation_cool, overall_sensation_bigger)
        return overall_sensation_bigger_ex

    def opposite_cool(self) -> float:
        """Returns the whole-body local_sensation_sorted calculated by opposite cool model."""
        overall_sensation_bigger_ex = NoOppositeModel(self.local_sensation).no_opposite_model('cool', 'all')
        overall_sensation_bigger = NoOppositeModel(self._local_sensation_cool_neutral).no_opposite_model('cool', 'all')
        # overall_sensation_bigger = np.mean(self.local_sensation_sorted)
        modifier = _opposite_modifier(self._local_sensation_warm, overall_sensation_bigger)
        return overall_sensation_bigger_ex

    def opposite_model(self) -> float:
        """Returns the whole-body local_sensation_sorted calculated by opposite model."""
        if self.bigger_group == 'cool':
            return self.opposite_cool()
        if self.is_cool_dominated:
            return self.opposite_dominated_model()
        if self.bigger_group == 'warm':
            return self.opposite_warm()


def sig2(diff, compare, sorted_sensation):
    if compare == 'larger':
        sorted_sensation = [i for i in sorted_sensation if i <= 0]
        return np.prod([sig(i, 15, 0) for i in sorted_sensation[:diff]])
    if compare == 'smaller':
        sorted_sensation = [i for i in sorted_sensation if i >= 0]
        return np.prod([sig(-i, 30, 0) for i in sorted_sensation[-diff-1:]])


class LocalSensationProcessor(SensationModel):
    """
    Calculate whole-body local_sensation_sorted for a set of local sensations.

    Attributes:
        model_num: The number of whole-body local_sensation_sorted calculation model. Refer to function get_model_num.
    """
    def __init__(self, local_sensation, whole_sensation_config: WholeSensationConfig):
        """
        Args:
            whole_sensation_config:
                Configurations for whole-body local_sensation_sorted calculation. Refer to class WholeSensationConfig.
        """
        self.whole_sensation_config = whole_sensation_config
        SensationModel.dominant_parts = self.whole_sensation_config.dominant_parts

        super().__init__(local_sensation)
        self.model_num = self.get_model_num()

    def smoothed_whole_sensation(self) -> float:
        """Return smoothed whole-body local_sensation_sorted."""

        x1 = self._min_dominant_part
        x2 = self._max
        x3 = self._min
        x4 = self._third_max if self._is_hands_feet_extreme('warm') else self._second_max
        x5 = self._third_min if self._is_hands_feet_extreme('cool') else self._second_min
        x6 = np.median(self._local_sensation_sorted)
        x7 = len(self.local_sensation[self.local_sensation > 0])-len(self.local_sensation[self.local_sensation < 0])
        d = min([len(self.local_sensation[self.local_sensation > 0]), len(self.local_sensation[self.local_sensation < 0])])
        lo = self._local_sensation_sorted[d:len(self._local_sensation_sorted)-d]
        # x6 = np.mean(lo) if lo else np.median(self._local_sensation_sorted)

        no_opposite_model = NoOppositeModel(self.local_sensation)
        opposite_model = OppositeModel(self.local_sensation)
        y1 = no_opposite_model.complaint_model_warm()
        y2 = no_opposite_model.complaint_model_cool()
        y3 = no_opposite_model.gradual_model_warm()
        y4 = no_opposite_model.gradual_model_cool()
        y5 = opposite_model.opposite_dominated_model()
        y6 = opposite_model.opposite_warm()
        y7 = opposite_model.opposite_cool()

        alpha = self.whole_sensation_config.smooth_alpha
        smooth_adjusted = self.whole_sensation_config.smooth_adjusted
        y_i, y_k, w_ik = [None] * 3  # Initialize variables

        if self.model_num == 1:
            y_i = y1
            y_k = [y3, y5, y6, y7]
            w_ik = [sig(-x4, alpha, -2),
                    sig(-x1, alpha, 1),
                    sig(x1, alpha, -1) * sig(-x3, alpha, 1),
                    sig(-x6, alpha, 0)]

        elif self.model_num == 2:
            y_i = y2
            y_k = [y4, y5, y6, y7]
            w_ik = [sig(x5, alpha, -2),
                    sig(-x1, alpha, 1) * sig(x6, alpha, 0),
                    sig(x6, alpha, 0) * sig(x1, alpha, -1),
                    sig(x2, alpha, 1)]

        elif self.model_num == 3:
            y_i = y3
            y_k = [y1, y4, y5, y6, y7]
            w_ik = [sig(x4, alpha, 2),
                    sig(-x6, alpha, 0) * sig(-x2, alpha, -1),
                    sig(-x1, alpha, 1),
                    sig(x1, alpha, -1) * sig(-x3, alpha, 1),
                    sig(-x6, alpha, 0) * sig(x2, alpha, 1)]

        elif self.model_num == 4:
            y_i = y4
            y_k = [y2, y3, y5, y6, y7]
            w_ik = [sig(-x5, alpha, 2),
                    sig(x6, alpha, 0) * sig(x3, alpha, -1),
                    sig(-x1, alpha, 1) * sig(x6, alpha, 0),
                    sig(x6, alpha, 0) * sig(x1, alpha, -1) * sig(-x3, alpha, -1),
                    sig(x2, alpha, 1)]

        elif self.model_num == 5:
            y_i = y5
            y_k = [y1, y2, y3, y4, y6, y7]
            w_ik = [sig(x6, alpha, 0) * sig(x4, alpha, 2) * sig(x3, alpha, -1),
                    sig(-x6, alpha, 0) * sig(-x2, alpha, -1) * sig(-x5, alpha, 2),
                    sig(x6, alpha, 0) * sig(-x4, alpha, -2) * sig(x3, alpha, -1),
                    sig(-x6, alpha, 0) * sig(-x2, alpha, -1) * sig(x5, alpha, -2),
                    sig(x6, alpha, 0) * sig(x1, alpha, -1) * sig(-x3, alpha, 1),
                    sig2(x7, 'smaller', self._local_sensation_sorted) * sig(x2, alpha, 1)]

        elif self.model_num == 6:
            y_i = y6
            y_k = [y1, y2, y3, y4, y5, y7]
            w_ik = [sig(x4, alpha, 2) * sig(x3, alpha, -1),
                    sig(-x6, alpha, 0) * sig(-x2, alpha, -1) * sig(-x5, alpha, 2),
                    sig(-x4, alpha, -2) * sig(x3, alpha, -1),
                    sig(-x6, alpha, 0) * sig(-x2, alpha, -1) * sig(x5, alpha, -2),
                    sig(-x1, alpha, 1),
                    sig2(x7, 'smaller', self._local_sensation_sorted) * sig(x2, alpha, 1)]
            print('sig6')
            print(sig2(x7, 'smaller', self._local_sensation_sorted))

        elif self.model_num == 7:
            y_i = y7
            y_k = [y1, y2, y3, y4, y5, y6]
            w_ik = [sig(x6, alpha, 0) * sig(x4, alpha, 2) * sig(x3, alpha, -1) * sig(x1, alpha, -1),
                    sig(-x2, alpha, -1) * sig(-x5, alpha, 2),
                    sig(x6, alpha, 0) * sig(-x4, alpha, -2) * sig(x3, alpha, -1) * sig(x1, alpha, -1),
                    sig(-x2, alpha, -1) * sig(x5, alpha, -2),
                    sig(-x1, alpha, 1) * sig(x6, alpha, 0),
                    sig(x6, alpha, 0) * sig(-x3, alpha, 1) * sig(x1, alpha, -1)]

        y_ik = [y - y_i for y in y_k]
        w_y_ik = [float(w * y) for w, y in zip(w_ik, y_ik)]

        if not smooth_adjusted:
            return y_i + np.sum(w_y_ik)

        if smooth_adjusted:
            w_y_ik_max_index = np.argmax(np.abs(w_y_ik))
            y_i_modified = y_i + w_y_ik[w_y_ik_max_index]
            y_ik_modified = [y_k[i] - y_i if i == w_y_ik_max_index else y_k[i] - y_i_modified for i in range(len(y_k))]
            w_y_ik_modified = [float(w * y) for w, y in zip(w_ik, y_ik_modified)]
            print(w_y_ik_modified)
            print(w_ik)
            print(y_i + np.sum(w_y_ik_modified))
            print(np.sum(w_y_ik_modified))
            print(y_i)
            print()
            # return y_i_modified
            return y_i + np.sum(w_y_ik_modified)

    def whole_sensation(self) -> float:
        """Return whole-body local_sensation_sorted."""
        if self.is_no_opposite:
            return NoOppositeModel(self.local_sensation).no_opposite_model()
        if not self.is_no_opposite:
            return OppositeModel(self.local_sensation).opposite_model()

    def get_whole_sensation(self) -> float:
        return self.smoothed_whole_sensation() if self.whole_sensation_config.smooth else self.whole_sensation()
