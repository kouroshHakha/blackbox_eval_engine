from typing import Union, Tuple, Sequence, Mapping, Dict, Any

import abc
from datetime import datetime
from dataclasses import dataclass
from multiprocessing.dummy import Pool as ThreadPool
import os
import numpy as np
from pathlib import Path
import jinja2
import atexit
import h5py

from utils.file import read_yaml, write_yaml

PathLike = Union[str, Path]
StateValue = Union[float, int, str]

debug = False


@dataclass
class Netlist:
    fpath: str
    content: str

    def __hash__(self):
        return hash(self.content)


class NgSpiceWrapper(abc.ABC):

    try:
        BASE_TMP_DIR = os.environ['NGSPICE_TMP_DIR']
    except KeyError:
        raise ValueError("Environment variable NGSPICE_TMP_DIR is not set.")

    def __init__(self, num_process: int, design_netlist: PathLike,
                 root_dir: PathLike = None) -> None:

        if root_dir is None:
            self.root_dir: Path = Path(NgSpiceWrapper.BASE_TMP_DIR).resolve()
        else:
            self.root_dir: Path = Path(root_dir).resolve()

        self.num_process: int = num_process
        self.base_design_name: str = Path(design_netlist).stem
        self.gen_dir: Path = self.root_dir / f'designs_{self.base_design_name}'

        self.gen_dir.mkdir(parents=True, exist_ok=True)

        with open(design_netlist, 'r') as raw_file:
            self.content = raw_file.read()

        # get/create cache file
        self.cache_path = self.gen_dir / 'cache.yaml'
        if self.cache_path.exists():
            self.cache = read_yaml(self.cache_path)
        else:
            self.cache = {}
        atexit.register(self._write_cache)

    def _write_cache(self):
        print(f'Saving cache for {self.base_design_name} ....')
        write_yaml(self.cache_path, self.cache)

    def get_design_name(self, dsn_id) -> str:
        return f'{self.base_design_name}_{dsn_id}'

    def get_design_folder(self, dsn_id) -> Path:
        return self.gen_dir / self.get_design_name(dsn_id)

    @classmethod
    def _save_as_hdf5_rec(cls, obj: Dict[str, Union[Dict, np.ndarray]], root: h5py.File):
        for k, v in obj.items():
            if isinstance(v, np.ndarray):
                root.create_dataset(name=k, data=v)
            elif isinstance(v, dict):
                grp = root.create_group(name=k)
                cls._save_as_hdf5_rec(v, grp)
            else:
                raise ValueError(f'Does not support type {type(obj)}')

    @classmethod
    def save_as_hdf5(cls, data_dict: Dict[str, Any], fpath: PathLike) -> None:
        with h5py.File(fpath, 'w') as root:
            cls._save_as_hdf5_rec(data_dict, root)

    @classmethod
    def _load_hdf5_rec(cls, root: h5py.Group) -> Dict[str, Any]:
        init_dict = {}
        for k, v in root.items():
            if isinstance(v, h5py.Dataset):
                init_dict[k] = np.array(v)
            elif isinstance(v, h5py.Group):
                init_dict[k] = cls._load_hdf5_rec(v)
            else:
                raise ValueError(f'Does not support type {type(v)}')

        return init_dict

    @classmethod
    def load_hdf5(cls, fpath: PathLike) -> Dict[str, Any]:
        with h5py.File(fpath, 'r') as f:
            return cls._load_hdf5_rec(f)

    def _create_design(self, state: Mapping[str, StateValue], dsn_id: str) -> Netlist:
        """
        Parameters
        ----------
        state: Mapping[str, StateValue]
            State dictionary from jinja variable to the value
        dsn_id: str
            Design object id

        Returns
        -------
        ret: Union[str, bool]
            False if netlist has been loaded, the fpath value if netlist has been created.
        """
        design_folder = self.get_design_folder(dsn_id)
        design_folder.mkdir(parents=True, exist_ok=True)

        fpath = design_folder / f'{self.base_design_name}.cir'

        temp = jinja2.Template(self.content)
        new_content = temp.render(**state)

        if new_content not in self.cache:
            with open(str(fpath), 'w') as f:
                f.write(new_content)

        return Netlist(fpath=str(fpath.resolve()), content=new_content)

    @staticmethod
    def _simulate(netlist: Netlist) -> int:
        info = 0  # this means no error occurred
        command = f'ngspice -b {netlist.fpath} >/dev/null 2>&1'
        exit_code = os.system(command)
        if debug:
            print(command)
            print(netlist.fpath)

        if exit_code % 256:
            info = 1  # this means an error has occurred
        return info

    def _update_cache(self, netlist: Netlist):
        self.cache[hash(netlist)] = datetime.utcnow()

    def _create_design_and_simulate(self,
                                    state: Dict[str, StateValue],
                                    verbose: bool = False) -> Tuple[Mapping[str, StateValue],
                                                                    Mapping[str, StateValue], int]:

        if debug:
            print('state', state)
            print('verbose', verbose)

        netlist = self._create_design(state, dsn_id=state['id'])
        loaded = False
        if hash(netlist) not in self.cache:
            print(f'Simulating design {state["id"]} ...')
            info = self._simulate(netlist)
            if not info:
                # simulation succeeded
                self._update_cache(netlist)
        else:
            print(f'Skipped simulation. Loaded results from {netlist.fpath}.')
            info = 0
            loaded = True

        if info != 0:
            raise ValueError(f'Ngspice simulation failed. Check log: {str(netlist.fpath)}.')
        else:
            try:
                specs = self.translate_result(state)
            except FileNotFoundError as e:
                if loaded:
                    print('Loaded results had some issues. Redoing the simulation ...')
                    del self.cache[hash(netlist)]
                    return self._create_design_and_simulate(state, verbose)
                else:
                    raise e

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
