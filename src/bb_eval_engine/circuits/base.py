from typing import Optional, Dict, Any, Union, Sequence

import abc
import numpy as np
import random

from ..base import EvaluationEngineBase
from ..util.design import Design
from ..util.importlib import import_cls

from .util.id import IDEncoder


SpecType = Union[float, int]
SpecSeqType = Union[Sequence[SpecType], SpecType]


class FlowManager(abc.ABC):

    @abc.abstractmethod
    def interpret(self, design: Design, *args, **kwargs) -> Dict[str, Any]:
        """
        Implement this method for the interpretation of each design parameter.
        This method can be used in batch_evaluate.
        Parameters
        ----------
        design: Design
            design object under consideration
        *args:
            optional positional arguments
        **kwargs:
            optional keyword arguments
        Returns
        -------
        values: Dict[str, Any]
            a dictionary representing the values of design parameters
        """
        raise NotImplementedError

    @abc.abstractmethod
    def batch_evaluate(self, batch_of_designs: Sequence[Design], *args, **kwargs) -> Sequence[Any]:
        raise NotImplementedError


class CircuitsEngineBase(EvaluationEngineBase, abc.ABC):

    def __init__(self, yaml_fname: Optional[str] = None,
                 specs: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        EvaluationEngineBase.__init__(self, yaml_fname, specs, **kwargs)

        self.spec_range = specs['spec_range']
        self.params_vec = {}
        self.params = self.specs['params']
        self.search_space_size = 1
        for key, value in self.specs['params'].items():
            listed_value = np.arange(value[0], value[1], value[2]).tolist()
            self.params_vec[key] = listed_value
            self.search_space_size = self.search_space_size * len(list(listed_value))

        self.id_encoder = IDEncoder(self.params_vec)

        # flow manager takes in a parameter dictionary and has functions to run simulation
        # does not take care of parameter interface with simulator
        _eval_cls_str = self.specs['flow_manager_cls']
        _eval_params = self.specs['flow_manager_params']
        _eval_cls = import_cls(_eval_cls_str)
        self.flow_manager: FlowManager = _eval_cls(**_eval_params, **kwargs)

    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)

    def get_rand_sample(self):
        """
        override this method to change the meaning of each design value
        """
        design_list = []
        for key, vec in self.params_vec.items():
            rand_idx = random.randrange(len(list(vec)))
            design_list.append(rand_idx)
        attrs = self.spec_range.keys()
        design = Design(design_list, attrs)
        design['id'] = design.id(self.id_encoder)
        # update design value_dict
        self._interpret_design(design)
        return design

    # noinspection PyUnresolvedReferences
    def _interpret_design(self, design: Design) -> None:
        """This function takes in a design object, finds what values mean and put them the
        value_dict attribure of the design"""
        param_values = {}
        for param_idx, key in zip(design['value'], self.params_vec.keys()):
            param_values[key] = self.params_vec[key][param_idx]
        design['value_dict'] = param_values

    def generate_rand_designs(self, n: int = 1, evaluate: bool = False, seed: Optional[int] = None,
                              **kwargs) -> Sequence[Design]:
        if seed:
            self.set_seed(seed)
        tried_designs, valid_designs = [], []
        remaining = n
        while remaining != 0:
            trying_designs = []
            useless_iter_count = 0
            while len(trying_designs) < remaining:
                rand_design = self.get_rand_sample()
                if rand_design in tried_designs or rand_design in trying_designs:
                    useless_iter_count += 1
                    if useless_iter_count > n * 10:
                        raise ValueError(f'large amount randomly selected samples failed {n}')
                    continue
                trying_designs.append(rand_design)

            if evaluate:
                self.evaluate(trying_designs)
                n_valid = 0
                for design in trying_designs:
                    tried_designs.append(design)
                    if design['valid']:
                        n_valid += 1
                        valid_designs.append(design)
                remaining = remaining - n_valid
            else:
                remaining = remaining - len(trying_designs)

        efficiency = len(valid_designs) / len(tried_designs)
        print(f'Efficiency: {efficiency}')
        return valid_designs

    def evaluate(self, designs: Sequence[Design], *args, **kwargs) -> Any:
        results = self.flow_manager.batch_evaluate(designs, sync=True)
        self.update_designs_with_results(designs, results)
        return designs

    def compute_penalty(self, spec_nums: SpecSeqType, spec_kwrd: str) -> SpecSeqType:
        """
        implement this method to compute the penalty(s) of a given spec key word based on the
        what the provided numbers for that specification.
        Parameters
        ----------
        spec_nums: SpecSeqType
            Either a single number or a sequence of numbers for a given specification.
        spec_kwrd: str
            The keyword of the specification of interest.

        Returns
        -------
            Either a single number or a sequence of numbers representing the penalty number for
            that specification
        """
        if not hasattr(spec_nums, '__iter__'):
            list_spec_nums = [spec_nums]
        else:
            list_spec_nums = spec_nums

        penalties = []
        for spec_num in list_spec_nums:
            penalty = 0
            ret = self.spec_range[spec_kwrd]
            if len(ret) == 3:
                spec_min, spec_max, w = ret
            else:
                spec_min, spec_max = ret
                w = 1
            if spec_max is not None:
                if spec_num > spec_max:
                    # if (spec_num + spec_max) != 0:
                    #     penalty += w*abs((spec_num - spec_max) / (spec_num + spec_max))
                    # else:
                    #     penalty += 1000
                    penalty += w * abs(spec_num - spec_max) / abs(spec_num)
                    # penalty += w * abs(spec_num - spec_max) / self.avg_specs[spec_kwrd]
            if spec_min is not None:
                if spec_num < spec_min:
                    # if (spec_num + spec_min) != 0:
                    #     penalty += w*abs((spec_num - spec_min) / (spec_num + spec_min))
                    # else:
                    #     penalty += 1000
                    penalty += w * abs(spec_num - spec_min) / abs(spec_min)
                    # penalty += w * abs(spec_num - spec_min) / self.avg_specs[spec_kwrd]
            penalties.append(penalty)
        return penalties

    def update_designs_with_results(self, designs: Sequence[Design],
                                    results: Sequence[Dict[str, Any]]) -> None:
        """
        Override this method to change the behavior of appending the results to the Design objects.
        This method updates the designs in-place.
        Parameters
        ----------
        designs: Sequence[Design]
            the sequence of designs
        results: Sequence[Dict[str, Any]]
            the sequence of dictionaries each representing the result of simulating designs in
            the order that was given

        Returns
        -------
            None
        """
        if len(designs) != len(results):
            raise ValueError('lengths do not match between the designs and the results')
        for design, result in zip(designs, results):
            try:
                for k, v in result.items():
                    design[k] = v
                design['valid'] = True
                self.post_process_design(design)
            except AttributeError:
                design['valid'] = False

    # noinspection PyMethodMayBeStatic
    def post_process_design(self, design: Design) -> None:
        """
        override this method to do post-processing of the design object. Use this function to
        compute cost function.
        Parameters
        ----------
        design: Design
            the Design object under consideration.

        Returns
        -------
        None
            This function should manipulate design object in-place.
        """
        cost = 0
        for spec_kwrd in self.spec_range:
            cost += self.compute_penalty(design[spec_kwrd], spec_kwrd)[0]
        design['cost'] = cost
