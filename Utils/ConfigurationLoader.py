from collections import namedtuple
import yaml

# Stores model configuration details; namedtuple is used so attributes can be referenced easily
ModelConfig = namedtuple('ModelConfig',
                         ['model_name', 'n_epochs', 'batch_size', 'lr', 'optimizer', 'optimizer_kws', 'lr_scheduler',
                          'lr_scheduler_kws'])
# Stores environment configuration details; namedtuple is used so attributes can be referenced easily
Environment = namedtuple('Environment', ['model', 'project_id', 'local_file_path_root', 'cloud_file_path_root'])


def load_environment(path_to_env='../Config/env.yaml') -> Environment:
    """
    Parses environment yaml file
    @param path_to_env:   location of environment file

    @rtype: Environment tuple
    """
    with open(path_to_env) as stream:
        return Environment(**yaml.safe_load(stream))


def load_model_config(env=None, path_to_model_configs='../Config/models.yaml'):
    """
    Retrieves the proper model configuration as specified by the environment config
    The environment variable "model" is used to specify which model configuration to use

    @param env: environment to use, loads default if None
    @param path_to_model_configs: location of model configs file
    @return: ModelConfig tuple
    """
    if env is None:
        env = load_environment()

    with open(path_to_model_configs) as stream:
        data = yaml.safe_load(stream)
        if env.model in data.keys():
            return ModelConfig(**data[env.model])
        else:
            raise KeyError(f"The model specified in env.yaml '{env.model}' does not exist.")
