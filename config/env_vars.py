import os
from scikit_tt.utils import enable_julia

def get_env_var():

    implementation = ""

    if os.environ.get("IMPL") == "python":

        implementation = "python"

        print("python implementation")

        return implementation

    elif os.environ.get("IMPL") == "julia":

        enable_julia()

        implementation = "julia"

        print("julia implementation")

        return implementation

    else:

        implementation = os.environ.get("IMPL")

        raise ValueError(f'{implementation} is not a valid value for IMPL')

    
