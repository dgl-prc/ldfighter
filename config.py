import yaml
import os

def read_config():
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../config.yaml")
    with open(cfg_path, 'r') as file:
        confg = yaml.load(file, Loader=yaml.FullLoader)
    return confg


class DictToClass(object):
    '''
    ;;将字典准换为 class 类型
    '''
    @classmethod
    def _to_class(cls, _obj):
        _obj_ = type('new', (object,), _obj)
        [setattr(_obj_, key, cls._to_class(value)) if isinstance(value, dict) else setattr(_obj_, key, value) for
         key, value in _obj.items()]
        return _obj_

class ReadConfigFiles(object):
    def __init__(self):
        '''
        ;;获取当前工作路径
        '''
        self.cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../config.yaml")
 
    @classmethod
    def open_file(cls):
        return yaml.load(
            open(cls().cfg_path, 'r', encoding='utf-8').read(), Loader=yaml.FullLoader
        )
 
    @classmethod
    def cfg(cls, item=None):
        return DictToClass._to_class(cls.open_file().get(item) if item else cls.open_file())

