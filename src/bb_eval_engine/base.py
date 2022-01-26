from typing import Optional, Dict, Any, Sequence, List, Union, cast

import abc
import yaml
import random
import numpy as np
import itertools
import time
from dataclasses import dataclass, field

from .data.design import Design
from .util.encoder import IntIDEncoder
from .sampler import RandomSampler


SpecType = Union[float, int]
SpecSeqType = Union[Sequence[SpecType], SpecType]

@dataclass(frozen=True)
class Spec:
    name: str
    lb: float = None
    ub: float = None
    weight: float = 1

    def meet(self, val):
        ans = True
        if self.ub:
            ans = ans and (val < self.ub)
        if self.lb:
            ans = ans and (val > self.lb)
        return ans

@dataclass
class DesignVar:
    name: str
    min: float
    max: float
    step: float
    vec: np.ndarray = field(init=False, repr=False)
    index_vec: np.ndarray = field(init=False, repr=False)
    min_index: int = 0
    max_index: int = field(init=False, repr=False)
    num: int = field(init=False, repr=False)

    def __post_init__(self):
        self.vec = np.arange(self.min , self.max, self.step)
        self.index_vec = np.arange(len(self.vec))
        self.max_index = self.index_vec[-1]
        self.num = self.max_index + 1


class EvaluationEngineBase(abc.ABC):

    env = None

    def __init__(self, yaml_fname: Optional[str] = None,
                 specs: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        if yaml_fname:
            with open(yaml_fname, 'r') as f:
                specs = yaml.load(f, Loader=yaml.FullLoader)

        self.specs = specs
        self.spec_range = {}
        for key, val in specs['spec_range'].items():
            self.spec_range[key] = Spec(key, *val)

        self.design_vars = {}
        self.params_vec = cast(Dict[str, np.ndarray], {})
        self.params = self.specs['params']
        self.search_space_size = 1
        for key, value in self.specs['params'].items():
            listed_value = np.arange(value[0], value[1], value[2])
            self.params_vec[key] = listed_value
            self.search_space_size = self.search_space_size * len(listed_value)
            self.design_vars[key] = DesignVar(key, *value)

        self.params_min = [0]*len(self.params_vec)
        self.params_max = []
        for val in self.params_vec.values():
            self.params_max.append(len(val)-1)
        self.id_encoder = IntIDEncoder(self.params_vec)

        self.input_bounds = ([x.min for x in self.design_vars.values()],
                             [x.max for x in self.design_vars.values()])

    def meets_specs(self, design: Design, objective_kw=''):
        for spec_key, spec in self.spec_range.items():
            if not spec.meet(design[spec_key]) and spec_key != objective_kw:
                return False
        return True

    @staticmethod
    def set_seed(seed):
        random.seed(seed)

    @property
    def ndim(self):
        return len(self.params_vec)

    def design_iter(self):
        dns_values_list = [range(len(v)) for v in self.params_vec.values()]
        for dsn in itertools.product(*dns_values_list):
            yield Design(dsn, key_specs=self.spec_range.keys())

    def generate_rand_designs(self, n: int = 1, evaluate: bool = False,
                              seed: Optional[int] = None, sampler: str='uniform') -> Sequence[Design]:
        """
        Generates a random sequence of Design samples.

        Parameters
        ----------
        n: int
            number of individuals in the data base.
        evaluate: bool
            True to evaluate the value of each design and populate its attributes.
        seed: Optional[int]
            Initial seed for random number generator.
        sampler: str [`uniform`, `lhs`]
            Determines the type of sampler used

        Returns
        -------
        database: Sequence[Design]
            a sequence of design objects
        """

        if seed:
            self.set_seed(seed)
        # Changes: assume that invalid designs are just penalized with a very large cost value

        if sampler == 'uniform':
            get_samples = RandomSampler.get_uniform_samples
        elif sampler == 'lhs':
            get_samples = RandomSampler.get_lh_sample
        else:
            raise ValueError(f'sampler {sampler} is not valid.')

        raw_samples = get_samples(np.array(self.params_min), np.array(self.params_max), n)
        dsns = [Design(sample.astype('int')) for sample in raw_samples]
        if not evaluate:
            return dsns
        return self.evaluate(dsns)

        # valid_designs = []
        # tried_designs = set()
        # remaining = n
        # while remaining != 0:
        #     trying_designs = set()
        #     useless_iter_count = 0
        #     s = time.time()
        #     while len(trying_designs) < remaining:
        #         rand_design = self.get_rand_sample()
        #         if rand_design in tried_designs or rand_design in trying_designs:
        #             useless_iter_count += 1
        #             if useless_iter_count > n * 10:
        #                 raise ValueError(f'large amount randomly selected samples failed {n}')
        #             continue
        #         trying_designs.add(rand_design)
        #     print(f'Finished generating designs in {time.time() - s} seconds.')
        #     if evaluate:
        #         self.evaluate(list(trying_designs))
        #         n_valid = 0
        #         for design in trying_designs:
        #             tried_designs.add(design)
        #             if design['valid']:
        #                 n_valid += 1
        #                 valid_designs.append(design)
        #         remaining = remaining - n_valid
        #     else:
        #         tried_designs.update(trying_designs)
        #         valid_designs += list(trying_designs)
        #         remaining = remaining - len(valid_designs)
        #
        # efficiency = len(valid_designs) / len(tried_designs)
        # print(f'Efficiency: {efficiency}')
        # return valid_designs

    def find_worst(self, vals: SpecSeqType, key: str, *args, ret_penalty: bool = True, **kwargs):
        """
        Parameters
        ----------
        vals: List[float]
            Values of consideration.
        key: str
            Spec name.
        ret_penalty: bool
            True to also return the worst penalty

        Returns
        -------
            the worst spec value according to the criteria
            Optional penalty
        """
        if not hasattr(vals, '__iter__'):
            vals = [vals]

        penalties = self.compute_penalty(vals, key)
        worst_penalty = max(penalties)
        worst_idx = penalties.index(worst_penalty)
        if ret_penalty:
            return vals[worst_idx], worst_penalty
        else:
            return vals[worst_idx]

    @abc.abstractmethod
    def get_rand_sample(self) -> Design:
        """
        implement this method to implement the random generation and meaning of each design
        Returns
        -------
        design: Design
            design object.
        """
        raise NotImplementedError

    def evaluate(self, designs: Sequence[Design], *args, **kwargs) -> Sequence[Design]:
        """Evaluates (runs simulations) a sequence of design objects, while resetting the state of
        designs.
        kwargs:
            do_interpret: True to run self._interpret_design wich modifies the design objects,
            by making it false it will be assumed that design objects are ready for
            get_evaluated_designs.
        """
        for dsn in designs:
            do_interpret = kwargs.get('do_interpret', True)
            if do_interpret:
                # update design key params
                self._interpret_design(dsn)
            dsn['id'] = dsn.id(self.id_encoder)
            dsn.clear_specs()
        return self._get_evaluated_designs(designs, *args, **kwargs)

    # noinspection PyUnresolvedReferences
    def _interpret_design(self, design: Design) -> None:
        """This function takes in a design object, finds what values mean and put them the
        spec of the design.
        Default behaviour: lookup the params_vec and replace the index with the actual value.
        """
        for param_idx, key in zip(design, self.params_vec.keys()):
            design.value_dict[key] = self.params_vec[key][param_idx].item()

    @abc.abstractmethod
    def _get_evaluated_designs(self, designs: Sequence[Design], *args, **kwargs) -> Any:
        """
        Evaluates (runs simulations) a sequence of design objects.
        Parameters
        ----------
        designs: Sequence[Design]
            input designs to be evaluated
        Returns
        -------
            Anything
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compute_penalty(self, vals: SpecSeqType, key: str, *arg, **kwargs) -> SpecSeqType:
        """
        Implement this method to compute the penalty(s) of a given spec key word based on the
        what the provided numbers for that specification.
        Parameters
        ----------
        vals: SpecSeqType
            Either a single number or a sequence of numbers for a given specification.
        key: str
            The keyword of the specification of interest.

        Returns
        -------
            A single number or a sequence of numbers representing the penalty number for
            that specification
        """
        raise NotImplementedError
