from typing import Sequence, Any, Mapping

import abc

class FlowManager(abc.ABC):

    @abc.abstractmethod
    def batch_evaluate(self, batch_of_designs: Sequence[Mapping[str, Any]], *args, **kwargs) \
            -> Sequence[Any]:
        raise NotImplementedError