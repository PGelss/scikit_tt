import toml
from scikit_tt.utils import enable_julia


def get_env_var():

    implementation = ""

    config = toml.load("./config.toml")

    implementation = config["scikit-tt"]["env_vars"]["IMPL"]

    if implementation == "python":

        print("python implementation")

        return implementation

    elif implementation == "julia":

        enable_julia()

        print("julia implementation")

        return implementation

    else:

        raise ValueError(f'{implementation} is not a valid value for IMPL')

    
