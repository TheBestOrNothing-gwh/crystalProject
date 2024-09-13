import importlib.util  


def get_config(config_path=""):
    # 使用spec来加载模块  
    spec = importlib.util.spec_from_file_location("module", config_path)  
    module = importlib.util.module_from_spec(spec)  
    spec.loader.exec_module(module)
    return module.module_config, module.data_config, module.trainer_config