from ext_argparse import ParameterEnum, Parameter


# Info for a saved alignment
# - In train.py, this info is only used if use_pretrained_model=True
# - In generate.py, evaluate.py or example_viz.py, it is used regardless of the value of use_pretrained_model
class ModelParameters(ParameterEnum):
    # TODO: switch to an Enum parameter
    model_module_to_load = \
        Parameter(default="full_model", arg_type=str,
                  arg_help="Must be set to one of ['only_flow_net', 'full_model']. Dictates whether the model will be"
                           "loaded in full or only the flow_net part will be loaded.")
    model_name = \
        Parameter(default="model_A", arg_type=str,
                  arg_help="Name of the pre-trained model to use.")
    model_iteration = \
        Parameter(default=0, arg_type=int,
                  arg_help="Iteration number of the model to load.")
