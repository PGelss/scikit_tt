import toml
from pathlib import Path

def get_env_var():

    implementation = ""

    config_path = Path(__file__).joinpath("config/config.toml")

    config = toml.load(config_path)

    implementation = config["scikit-tt"]["env_vars"]["IMPL"]

    if implementation == "python":

        print("python implementation")

        return implementation

    elif implementation == "julia":

        print("julia implementation")

        return implementation

    else:

        raise ValueError(f'{implementation} is not a valid value for IMPL')

    
