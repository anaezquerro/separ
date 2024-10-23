from __future__ import annotations
from configparser import SectionProxy
import pickle 

class Config(object):
    EXTENSION = 'cnf'
    REMOVE = ['kwargs', 'self']

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            self.__setattr__(key, value)

    def __call__(self) -> dict:
        return self.__dict__
    
    def __getattr__(self, attr):
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:
            return None
    
    def __contains__(self, item) -> bool:
        return item in self.__dict__.keys()
    
    def __or__(self, other: Config) -> Config:
        if isinstance(other, Config):
            return Config(**other(), **self())
        else:
            raise NotImplementedError
        
    def __ior__(self, other: Config) -> Config:
        if isinstance(other, Config):
            self.update(**other())
        else:
            raise NotImplementedError
        return self 
    
    def __getstate__(self): 
        return self.__dict__
    
    def __setstate__(self, d): 
        self.__dict__.update(d)
    
    def join(self, other: Config) -> Config:
        """Inplace update with other configuration

        Args:
            other (Config): Other configuration.

        Returns:
            Config: Updated configuration.
        """
        if isinstance(other, Config):
            self.update(**other())
            return self 
        else:
            raise NotImplementedError

    def update(self, **kwargs) -> Config:
        for key, value in kwargs.items():
            self.__setattr__(key, value)
        return self 
            
    def pop(self, name):
        value = getattr(self, name)
        self.__delattr__(name)
        return value 
    
    def save(self, path: str):
        if not path.endswith(f'.{self.EXTENSION}'):
            path += f'.{self.EXTENSION}'
        with open(path, 'wb') as writer:
            pickle.dump(self(), writer)
            
    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as reader:
            params = pickle.load(reader)
        return cls(**params)

    @classmethod
    def from_class(cls, data) -> Config:
        data = dict(filter(lambda x: not (x[0].startswith('_') or x[0] in Config.REMOVE), data.items()))
        return Config(**data)

    @classmethod 
    def from_ini(cls, section: SectionProxy) -> Config:
        data = dict()
        for param, value in section.items():
            try:
                data[param] = eval(value)
            except NameError:
                data[param] = value 
        return Config(**data)
