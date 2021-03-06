import argparse
from pathlib import Path
import yaml
import pickle
import time
import IPython

from utils.pdb import register_pdb_hook
register_pdb_hook()

from bb_eval_engine.util.importlib import import_cls
from bb_eval_engine.base import EvaluationEngineBase

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('specs_fname', help='specs yaml file')
    parser.add_argument('--db_path', '-db', type=str, default='',
                        help='[Optional] location to store the db file')
    parser.add_argument('-n', '--number', dest='number', type=int, default=1,
                        help='number of individuals in the database')
    parser.add_argument('-s', '--seed', default=None, type=int,
                        help='the seed used for generating the data base')
    parser.add_argument('-v', '--verbose', default=False, action='store_true',
                        help='True to make the underlying processes verbose')
    parser.add_argument('-proc', '--processes', default=False, action='store_true',
                        help='True to make multi-processes enable, default is multi-thread')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='number of workers used for multi-process or multi-thread')
    parser.add_argument('-p', '--pause', default=False, action='store_true',
                        help='True to stop after running')
    args = parser.parse_args()
    return args


def run_main(args: argparse.Namespace):

    with open(args.specs_fname, 'r') as f:
        specs = yaml.load(f, Loader=yaml.Loader)

    kwargs = dict(
        verbose=args.verbose,
        processes=args.processes,
        num_workers=args.num_workers,
    )

    eval_engine_str = specs['bb_engine']
    eval_engine_params = specs['bb_engine_params']
    eval_engine_cls = import_cls(eval_engine_str)
    eval_engine: EvaluationEngineBase = eval_engine_cls(specs=eval_engine_params, **kwargs)
    start = time.time()
    designs = eval_engine.generate_rand_designs(n=args.number, evaluate=True, seed=args.seed)

    if args.db_path:
        db_path = Path(args.db_path)
        db_path.parent.mkdir(exist_ok=True, parents=True)
        with open(db_path, 'wb') as f:
            pickle.dump(designs, f, pickle.HIGHEST_PROTOCOL)

        print(f'data base stored in {str(db_path)}')
    print(f'random generation of {args.number} samples took {time.time() - start:.6} seconds')

    if args.pause:
        IPython.embed()


if __name__ == '__main__':
    args = parse_args()
    run_main(args)