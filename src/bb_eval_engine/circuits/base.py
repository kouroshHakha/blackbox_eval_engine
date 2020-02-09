from typing import Optional, Dict, Any, Union, Sequence

import abc
import random

from ..base import EvaluationEngineBase
from bb_eval_engine.data.design import Design
from ..util.importlib import import_cls


SpecType = Union[float, int]
SpecSeqType = Union[Sequence[SpecType], SpecType]


class FlowManager(abc.ABC):

    def __init__(self, *args, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def batch_evaluate(self, batch_of_designs: Sequence[Design], *args, **kwargs) -> Sequence[Any]:
        raise NotImplementedError


class CircuitsEngineBase(EvaluationEngineBase, abc.ABC):

    def __init__(self, yaml_fname: Optional[str] = None,
                 specs: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        EvaluationEngineBase.__init__(self, yaml_fname, specs, **kwargs)

        # flow manager takes in a parameter dictionary and has functions to run simulation
        # does not take care of parameter interface with simulator
        _eval_cls_str = self.specs['flow_manager_cls']
        _eval_params = self.specs['flow_manager_params']
        _eval_cls = import_cls(_eval_cls_str)
        self.flow_manager: FlowManager = _eval_cls(**_eval_params, **kwargs)

    def get_rand_sample(self):
        """
        override this method to change the meaning of each design value
        """
        design_list = []
        for key, vec in self.params_vec.items():
            rand_idx = random.randrange(len(list(vec)))
            design_list.append(rand_idx)
        design = Design(design_list, key_specs=self.spec_range.keys())

        return design

    def _get_evaluated_designs(self, designs: Sequence[Design], *args,
                               **kwargs) -> Sequence[Design]:
        """Side effect: design objects will have more attributes"""
        results = self.flow_manager.batch_evaluate(designs, sync=True)
        self.update_designs_with_results(designs, results)
        return designs

    def compute_penalty(self, vals: SpecSeqType, key: str, *args, **kwargs) -> SpecSeqType:

        try:
            spec_num_iter = iter(vals)
        except TypeError:
            spec_num_iter = iter([vals])

        penalties = []
        for spec_num in spec_num_iter:
            ret = self.spec_range[key]
            penalty = 0
            if len(ret) == 3:
                spec_min, spec_max, w = ret
            else:
                spec_min, spec_max = ret
                w = 1
            if spec_max is not None:
                if spec_num > spec_max:
                    # if (spec_num + spec_max) != 0:
                    #     penalty += w * abs((spec_num - spec_max) / (spec_num + spec_max))
                    # else:
                    #     penalty += 1000
                    penalty += w * abs(spec_num - spec_max) / abs(spec_num)
                    # penalty += w * abs(spec_num - spec_max) / self.avg_specs[spec_kwrd]
            elif spec_min is not None:
                if spec_num < spec_min:
                    # if (spec_num + spec_min) != 0:
                    #     penalty += w * abs((spec_num - spec_min) / (spec_num + spec_min))
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
