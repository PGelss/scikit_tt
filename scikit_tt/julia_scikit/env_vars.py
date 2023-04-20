import toml
from pathlib import Path

def default_env_var():

    implementation = ""

    # Go three levels up
    config_path = Path(__file__).parents[2].joinpath("config.toml")

    # Load configuration file
    config = toml.load(config_path)

    # Load get default environment variable
    implementation = config["scikit-tt"]["env_vars"]["DEF_IMPL"]

    if implementation == "python":

        print("python implementation")

        return implementation

    elif implementation == "julia":

        print("julia implementation")

        return implementation

    else:

        raise ValueError(f'{implementation} is not a valid value for IMPL')

    
