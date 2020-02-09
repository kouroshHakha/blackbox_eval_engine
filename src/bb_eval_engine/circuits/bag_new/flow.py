from typing import Dict, Any, List, Union

import abc
import yaml
import os
from pathlib import Path
from jinja2 import Template

from bag_mp.core import BagMP
from bag_mp.client_wrapper import synchronize, FutureWrapper
from bag_mp.file import read_file

from ..base import FlowManager
from bb_eval_engine.data.design import Design


class EvalTemplate:
    def __init__(self, temp_path: os.PathLike):
        self._path: Path = Path(temp_path).resolve()
        self.content: str = read_file(self._path)
        self.jinja_temp = Template(self.content)

    def render_plain(self, params: Dict[str, Any]) -> str:
        return self.jinja_temp.render(**params)

    def render_yaml(self, params: Dict[str, Any]) -> Dict[str, Any]:
        yaml_content = self.render_plain(params)
        specs = yaml.load(yaml_content, Loader=yaml.Loader)
        return specs


class BagFlowManager(FlowManager, abc.ABC):
    def __init__(self, temp_fname, base_name,  *args, **kwargs):
        FlowManager.__init__(self, *args, **kwargs)
        self.template = self._get_template(temp_fname)
        self.base_name = base_name
        # the project object used for running minimum executable tasks
        interactive = kwargs.pop('interactive', False)
        verbose = kwargs.pop('verbose', False)
        processes = kwargs.pop('processes', False)
        self.prj = BagMP(interactive=interactive, verbose=verbose, processes=processes)

    @staticmethod
    def get_results(results: List[FutureWrapper]) -> Any:
        synchronize(results)
        cleared_results = []
        for job_res in results:
            try:
                res = job_res.result()
                cleared_results.append(res)
            except SystemError:
                cleared_results.append(SystemError)
        return cleared_results

    @staticmethod
    def sync(results: Union[List[FutureWrapper], FutureWrapper]) -> Any:
        return synchronize(results)

    def render(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self.template.render_yaml(params)

    @staticmethod
    def _get_template(fname: os.PathLike) -> EvalTemplate:
        return EvalTemplate(fname)

    def update_impl_lib_cell_name_with_design_id(self, design: Design) -> Dict[str, str]:
        param_update = {}
        dsn_id = design['id']
        base_name = self.base_name
        param_update['impl_lib'] = f'{base_name}_{dsn_id}'
        param_update['impl_cell'] = f'{base_name}'

        return param_update
