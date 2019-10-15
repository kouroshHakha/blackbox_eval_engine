from typing import Union, Tuple, Sequence, Mapping, Dict

from multiprocessing.dummy import Pool as ThreadPool
import os
from pathlib import Path
import abc
import jinja2

PathLike = Union[str, Path]
StateValue = Union[float, int, str]

debug = False


class NgSpiceWrapper(abc.ABC):

    BASE_TMP_DIR = os.environ.get('NGSPICE_TMP_DIR', '/tmp/ckt_da')

    def __init__(self, num_process: int, design_netlist: PathLike,
                 root_dir: PathLike = None) -> None:

        if root_dir is None:
            self.root_dir = NgSpiceWrapper.BASE_TMP_DIR
        else:
            self.root_dir = root_dir

        self.num_process = num_process
        self.base_design_name = Path(design_netlist).stem
        self.gen_dir = Path(self.root_dir) / f'designs_{self.base_design_name}'

        self.gen_dir.mkdir(parents=True, exist_ok=True)

        with open(design_netlist, 'r') as raw_file:
            self.content = raw_file.read()

    def get_design_name(self, dsn_id) -> str:
        return f'{self.base_design_name}_{dsn_id}'

    def get_design_folder(self, dsn_id) -> Path:
        return self.gen_dir / self.get_design_name(dsn_id)

    def _create_design(self, state: Mapping[str, StateValue], dsn_id) -> str:
        design_folder = self.get_design_folder(dsn_id)
        design_folder.mkdir(parents=True, exist_ok=True)

        fpath = design_folder / f'{self.base_design_name}.cir'

        temp = jinja2.Template(self.content)
        new_content = temp.render(**state)

        with open(str(fpath), 'w') as f:
            f.write(new_content)

        return str(fpath.resolve())

    @staticmethod
    def _simulate(fpath: str) -> int:
        info = 0  # this means no error occurred
        command = f'ngspice -b {fpath} >/dev/null 2>&1'
        exit_code = os.system(command)
        if debug:
            print(command)
            print(fpath)

        if exit_code % 256:
            # raise RuntimeError('program {} failed!'.format(command))
            info = 1  # this means an error has occurred
        return info

    def _create_design_and_simulate(self,
                                    state: Dict[str, StateValue],
                                    verbose: bool = False) -> Tuple[Mapping[str, StateValue],
                                                                    Mapping[str, StateValue], int]:

        if debug:
            print('state', state)
            print('verbose', verbose)

        fpath = self._create_design(state, dsn_id=state['id'])
        info = self._simulate(fpath)
        specs = self.translate_result(state)
        return state, specs, info

    def run(self, states: Sequence[Mapping[str, StateValue]],
            verbose: bool = False) -> Sequence[Tuple[Mapping[str, StateValue],
                                                     Mapping[str, StateValue], int]]:
        """
        This method runs simulations for a batch of input states in parallel.
        """

        pool = ThreadPool(processes=self.num_process)
        arg_list = [(state, verbose) for state in states]
        specs = pool.starmap(self._create_design_and_simulate, arg_list)
        pool.close()
        return specs

    @abc.abstractmethod
    def translate_result(self, state: Mapping[str, StateValue]) -> Mapping[str, StateValue]:
        """
        This method needs to be overwritten according to circuit needs,
        parsing output, playing with the results to get a cost function, etc.
        The designer should look at his/her netlist and accordingly write this function.
        state should include keywords which refer to output path
        """
        raise NotImplemented
