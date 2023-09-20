from .model import build_ovvg


def build_model(args):
    # we use register to maintain models from catdet6 on.
    from .registry import MODULE_BUILD_FUNCS

    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model = build_func(args)
    return model
