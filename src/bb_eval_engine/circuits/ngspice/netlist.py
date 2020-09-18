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
from numbers import Number
import subprocess

from utils.file import read_yaml, write_yaml

PathLike = Union[str, Path]
StateValue = Union[float, int, str]


@dataclass(eq=True)
class Netlist:
    fpath: Path
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
        self.cache: Dict[str, Tuple[int, str]]
        self.updated_cache = False
        if self.cache_path.exists():
            self.cache = read_yaml(self.cache_path) or {}
            stat = os.stat(str(self.cache_path))
            self.last_cache_mtime = stat[-1]
        else:
            self.cache = {}
            self.last_cache_mtime = 0
        # atexit takes care of saving the current cache content in case of an error
        atexit.register(self._write_cache)

    def _write_cache(self):
        if self.updated_cache:
            # read the yaml if cache file already exists and has been modified since last time visited
            if self.cache_path.exists():
                stat = os.stat(str(self.cache_path))
                if self.last_cache_mtime < stat[-1]:
                    current_cache = read_yaml(self.cache_path)
                    if current_cache is None:
                        current_cache = {}
                else:
                    current_cache = {}
            else:
                current_cache = {}
            current_cache.update(self.cache)
            # print(f'Saving cache for {self.base_design_name} ....')
            write_yaml(self.cache_path, current_cache)
            # update last mtime stamp after updating cache file
            self.last_cache_mtime = os.stat(str(self.cache_path))[-1]
            self.updated_cache = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._write_cache()

    def get_design_name(self, dsn_id) -> str:
        return f'{self.base_design_name}_{dsn_id}'

    def get_design_folder(self, dsn_id) -> Path:
        return self.gen_dir / self.get_design_name(dsn_id)

    @classmethod
    def _save_as_hdf5_rec(cls, obj: Mapping[str, Union[Mapping, np.ndarray]], root: h5py.File):
        for k, v in obj.items():
            if isinstance(v, np.ndarray):
                root.create_dataset(name=k, data=v)
            elif isinstance(v, dict):
                grp = root.create_group(name=k)
                cls._save_as_hdf5_rec(v, grp)
            elif isinstance(v, Number):
                root.create_dataset(name=k, data=v)
            else:
                raise ValueError(f'Does not support type {type(v)}')

    @classmethod
    def save_as_hdf5(cls, data_dict: Mapping[str, Any], fpath: PathLike) -> None:
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

    @classmethod
    def create_new_name(cls, dsn_folder: PathLike) -> Path:
        counter = 1
        new_name = dsn_folder / 'sim.hdf5'
        while new_name.exists():
            new_name = dsn_folder / f'sim_{counter}.hdf5'
            counter += 1
        return new_name

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

        return Netlist(fpath=fpath.resolve(), content=new_content)

    @staticmethod
    def _simulate(netlist: Netlist, debug: bool = False) -> int:
        info = 0  # this means no error occurred
        command = ['ngspice', '-b', f'{netlist.fpath}']
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if 'Error' in result.stdout.decode('utf-8') + result.stderr.decode('utf-8'):
            info = 1

        if debug:
            print(command)
            print(netlist.fpath)
        return info

    def _update_cache(self, netlist: Netlist, hdf5_file: PathLike):
        self.cache[netlist] = (datetime.utcnow(), str(hdf5_file))
        self.updated_cache = True

    def _create_design_and_simulate(self,
                                    state: Mapping[str, StateValue],
                                    verbose: bool = False,
                                    debug: bool = False) -> Tuple[Mapping[str, StateValue],
                                                                  Mapping[str, StateValue], int]:

        if debug and verbose:
            print('state', state)

        netlist = self._create_design(state, dsn_id=state['id'])
        loaded = False
        if netlist not in self.cache:
            if verbose:
                print(f'Simulating design {state["id"]} ...')
            info = self._simulate(netlist, debug=debug and verbose)
            if not info:
                # simulation succeeded and not it cache
                results = self.parse_output(state)
                dsn_folder = self.get_design_folder(state['id'])
                hdf5_file: Path = self.create_new_name(dsn_folder)
                self.save_as_hdf5(results, hdf5_file)
                self._update_cache(netlist, hdf5_file)
        else:
            if verbose:
                print(f'Skipped simulation. Loaded results from {netlist.fpath.parent}.')
            info = 0
            loaded = True

        if info != 0:
            raise ValueError(f'Ngspice simulation failed. Check log: {str(netlist.fpath)}.')
        else:
            try:
                hdf5_path = self.cache[netlist][1]
                results = self.load_hdf5(hdf5_path)
                specs = self.translate_result(state, results)
            except OSError as e:
                if loaded:
                    print('Loaded results had some issues. Redoing the simulation ...')
                    del self.cache[netlist]
                    return self._create_design_and_simulate(state, verbose)
                else:
                    raise e

        return state, specs, info

    def run(self, states: Sequence[Mapping[str, StateValue]],
            verbose: bool = False, debug: bool = False) -> Sequence[Tuple[Mapping[str, StateValue],
                                                                          Mapping[str, StateValue], int]]:
        """
        This method runs simulations for a batch of input states in parallel.
        """
        if debug:
            specs = [self._create_design_and_simulate(state, verbose, debug) for state in states]
        else:
            pool = ThreadPool(processes=self.num_process)
            arg_list = [(state, verbose) for state in states]
            specs = pool.starmap(self._create_design_and_simulate, arg_list)
            pool.close()

        return specs

    @classmethod
    @abc.abstractmethod
    def parse_output(cls, state: Mapping[str, StateValue]) -> Mapping[str, np.ndarray]:
        """
        This method needs to be overwritten according to circuit needs, parsing output.
        The designer should look at his/her netlist and accordingly write this function.
        state should include keywords which refer to output path
        """
        raise NotImplementedError

    @abc.abstractmethod
    def translate_result(self, state: Mapping[str, StateValue],
                         results: Mapping[str, np.ndarray]) -> Mapping[str, Any]:
        raise NotImplemented
