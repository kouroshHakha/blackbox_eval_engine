from __future__ import annotations
from typing import List, Any, Dict, Iterable
from copy import copy

from utils.immutable import to_immutable


class Design:
    def __init__(self, value: Iterable[Any], key_specs: Iterable[str] = ()):

        self._value: List[Any] = list(value)
        self._key_specs = list(key_specs)
        # value is the list representation of design.
        # value_dict is the dictionary representation of value. Should be populated from outside.
        # specs includes both key_specs (i.e. optimization specs) and debugging specs (e.g. sim_dir)
        self._attrs: Dict[str, Any] = {'value': value, 'id': None, 'value_dict': {}, 'specs': {}}
        self.reset_key_specs()

    @property
    def value(self):
        return self._value

    @property
    def value_dict(self):
        return self._attrs['value_dict']

    @property
    def specs(self):
        return self._attrs['specs']

    @property
    def key_specs(self):
        kspecs = {}
        for spec in self._key_specs:
            kspecs[spec] = self._attrs['specs'][spec]
        return kspecs

    def clear_specs(self):
        """Deletes all debugging specs and nullifies all the key_specs"""
        keys = list(self._attrs['specs'].keys())
        for k in keys:
            if k not in self._key_specs:
                del self._attrs['specs'][k]
            else:
                self._attrs['specs'][k] = None

    def reset_key_specs(self):
        """Nullifies only the key specs"""
        for k in self._key_specs:
            self._attrs['specs'][k] = None

    def id(self, id_encoder):
        id_str = id_encoder.convert_list_2_id(self._value)
        self._attrs['id'] = id_str
        return id_str

    def copy(self):
        new_dsn = Design(copy(self._value), copy(self._key_specs))
        new_dsn._attrs['value_dict'] = copy(self._attrs['value_dict'])
        new_dsn._attrs['id'] = self._attrs['id']
        for k in self.specs:
            new_dsn._attrs['specs'][k] = copy(self._attrs['specs'][k])
        return new_dsn

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._value[item]
        elif isinstance(item, str):
            if item is 'id':
                return self._attrs['id']
            return self.specs[item]
        else:
            raise ValueError(f'Cannot get item of type: {type(item)}')

    def __setitem__(self, item: Any, value: Any = None):
        if isinstance(item, int):
            self._value[item] = value
        elif isinstance(item, str):
            if item is 'id':
                self._attrs['id'] = value
            else:
                self.specs[item] = value
        else:
            raise ValueError(f'Cannot set item of type: {type(item)}')

    def __delitem__(self, key):
        if isinstance(key, str):
            if key is 'id':
                del self._attrs['id']
            else:
                del self.specs[key]
        else:
            raise ValueError(f'Cannot delete item of type: {type(key)}')

    def __contains__(self, item):
        if isinstance(item, int):
            return item in self._value
        elif isinstance(item, str):
            return item in self.specs
        else:
            raise ValueError(f'Item of type: {type(item)} is not supported')

    def __str__(self):
        return f'{self.value_dict}, {self.key_specs}'

    def __repr__(self):
        return self._value.__repr__()

    def __hash__(self):
        # if self._value is None:
        #     raise ValueError('attribute value hashable is not set')

        return hash(to_immutable(self._value))

    def __eq__(self, other):
        if self.value is None or other.value is None:
            raise ValueError('attribute value hashable is not set')
        if isinstance(other, Design):
            return self.value == other.value
        elif isinstance(other, list):
            return self.value == other
        else:
            raise ValueError(f'Cannot compare type Design with other type {type(other)}')

    def __lt__(self, other: Design):
        return self._value.__lt__(other._value)

    def __len__(self):
        return len(self._value)

    def __iter__(self):
        return iter(self.value)

    # handle pickling, because of the __getattr__ implementation it will look for __getstate__ in
    # self._attrs and raises KeyError
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)
