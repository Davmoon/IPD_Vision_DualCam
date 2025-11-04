import yaml
import degirum as dg
import degirum_tools


def load_config(config_file):
    """Load parameters from the YAML configuration file if provided."""
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def benchmark(
    config_file=None,
    inference_host_address="@cloud",
    iterations=100,
    token=None,
    **kwargs,
):
    """Run AI inference on multiple models using a config file or command-line options."""

    # Default values
    default_config = {
        "model_zoo_url": "degirum/public",
        "model_names": [
            "mobilenet_v1_imagenet--224x224_quant_n2x_orca1_1",
            "mobilenet_v2_imagenet--224x224_quant_n2x_orca1_1",
            "resnet50_imagenet--224x224_pruned_quant_n2x_orca1_1",
            "efficientnet_es_imagenet--224x224_quant_n2x_orca1_1",
            "efficientdet_lite1_coco--384x384_quant_n2x_orca1_1",
            "mobiledet_coco--320x320_quant_n2x_orca1_1",
            "yolov8n_relu6_coco--640x640_quant_n2x_orca1_1",
            "yolov8n_relu6_face--640x640_quant_n2x_orca1_1",
            "deeplab_seg--513x513_quant_n2x_orca1_1",
        ],
    }

    # Load the config file if provided, otherwise use default values
    if config_file:
        config = load_config(config_file)
    else:
        config = default_config

    # Extract model-specific parameters from the config
    model_zoo_url = config.get("model_zoo_url", default_config["model_zoo_url"])
    model_names = config.get("model_names", default_config["model_names"])

    # Use token from environment or provided argument
    token = token or degirum_tools.get_token()  # Token from CLI argument or environment

    if not token:
        print(
            "Error: Please provide a cloud platform token via the CLI or ensure it is retrievable from the environment."
        )
        return

    # Print the number of models and iterations
    print(f"Models    : {len(model_names)}")
    print(f"Iterations: {iterations}\n")

    # Print the header
    CW = (62, 19, 16, 16)  # column widths
    header = f"{'Model name':{CW[0]}}| {'Postprocess Type':{CW[1]}} | {'Observed FPS':{CW[2]}} | {'Max Possible FPS':{CW[3]}} |"
    print(f"{'-'*len(header)}")
    print(header)
    print(f"{'-'*len(header)}")

    # Loop through each model name, measure FPS, and print results
    for model_name in model_names:
        # Prepare arguments for loading the model
        model_args = {
            "model_name": model_name,
            "inference_host_address": inference_host_address,
            "zoo_url": model_zoo_url,
            "token": token,
        }

        # Add extra keyword arguments (kwargs) to the model arguments
        model_args.update(kwargs)

        # Load the model
        model = dg.load_model(**model_args)

        # Use the model_time_profile utility to measure the FPS
        result = degirum_tools.model_time_profile(model, iterations)

        # Print the result for the current model
        print(
            f"{model_name:{CW[0]}}|"
            + f" {result.parameters.OutputPostprocessType:{CW[1]}} |"
            + f" {result.observed_fps:{CW[2]}.1f} |"
            + f" {result.max_possible_fps:{CW[3]}.1f} |"
        )
