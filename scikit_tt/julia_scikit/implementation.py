from pathlib import Path
import os

def enable_julia():

    print("Importing Julia")
    
    from julia.api import Julia

    jl = Julia(compiled_modules=False)

    from julia import Pkg

    parent_path = Path(__file__).parent

    julia_path  = str(parent_path.joinpath("ScikitTT"))
    
    # Make package available by tracking it by path
    print("Adding ScikitTT to Julia packages")
    Pkg.develop(path=julia_path)

    # Set environment variable to julia
    os.environ["IMPL"] = "julia"

    return

def disable_julia():

    os.environ["IMPL"] = "python"

    return


def get_julia_scikit():

    from julia import ScikitTT as julia_scikit

    return julia_scikit



