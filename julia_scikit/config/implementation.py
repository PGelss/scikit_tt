from pathlib import Path

def enable_julia():

    print("Importing Julia")
    
    from julia.api import Julia

    jl = Julia(compiled_modules=False)

    from julia import Pkg

    parent_path = Path(__file__).parent

    julia_path  = parent_path.joinpath("ScikitTT")
    
    # Make package available by tracking it by path
    Pkg.develop(path=julia_path)


def get_julia_scikit():

    from julia import ScikitTT as julia_scikit

    return julia_scikit



