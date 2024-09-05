import importlib.util
import sys 

def get_config(path):
    spec = importlib.util.spec_from_file_location("config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module_config = module.module_config
    data_config = module.data_config
    return module_config, data_config