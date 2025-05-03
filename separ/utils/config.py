from __future__ import annotations

from configparser import ConfigParser
from typing import Union, Dict 

def parse(raw: Dict[str, str]):
    """Parse a configuration, transforming strings to numerical values when possible"""
    parsed = dict()
    for name, value in raw.items():
        if value.lower() in ['true', 'false']:
            value = value.capitalize()
        try:
            parsed[name] = eval(value)
        except:
            parsed[name] = value 
    return parsed

class Config:
    def __init__(self, **params):
        for name, value in params.items():
            self.__setattr__(name, value)
            
    def __len__(self) -> int:
        return len(self.__dict__)
    
    def __getitem__(self, name: str):
        return self.__dict__[name]
    
    def __iter__(self):
        return iter(self.__dict__)
    
    def items(self):
        return self.__dict__.items()

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()
    
    def to_dict(self) -> dict:
        return self.__dict__
    
    def __or__(self, other: Config):
        if not isinstance(other, Config):
            return NotImplementedError
        return Config(**self.__dict__, **other.__dict__)
    
    def __contains__(self, name):
        return name in self.keys()
    
    def __add__(self, other: Config) -> Config:
        if isinstance(other, Config):
            return Config(**self, **other)
        else:
            raise NotImplementedError
        
    def __radd__(self, other: Config) -> Config:
        if isinstance(other, Config):
            return self + other 
        elif other == 0:
            return self 
        else:
            raise NotImplementedError
        
    def update(self, other: Union[Config, dict]):
        for param, value in other.items():
            if value is not None:
                self.__setattr__(param, value)
    
    # def __getattr__(self, name):
    #     if name not in self.__dict__.keys():
    #         return None
    #     else:
    #         return self.__dict__[name]
    
    @classmethod
    def from_ini(cls, path: str) -> Dict[str, Config]:
        cparser = ConfigParser()
        cparser.optionxform = str
        cparser.read(path)
        confs = dict()
        for section in cparser.sections():
            confs[section] = Config(**parse(cparser[section]))
        return confs