# degirum_cli/predict_image.py

import degirum as dg
import degirum_tools
import os


def run_inference(
    inference_host_address,
    model_zoo_url,
    model_name,
    image_source,
    token=None,
    **kwargs,
):
    """Run AI inference on an image."""
    print("Running inference with the following parameters:")
    print(f"  Inference Host Address: {inference_host_address}")
    print(f"  Model Zoo URL: {model_zoo_url}")
    print(f"  Model Name: {model_name}")
    print(f"  Image Source: {image_source}")
    print(f"  Token: {'Provided' if token else 'Loaded from environment'}")
    print(f"  Extra Args: {kwargs}")

    token = token or degirum_tools.get_token()

    # Prepare load_model arguments
    model_args = {
        "model_name": model_name,
        "inference_host_address": inference_host_address,
        "zoo_url": model_zoo_url,
        "token": token,
    }
    model_args.update(kwargs)

    # Perform inference
    model = dg.load_model(**model_args)
    inference_result = model(image_source)
    print(f"Inference result: {inference_result}")
    # Check if the script is running in a display environment
    if (
        os.environ.get("DISPLAY") or os.name == "nt"
    ):  # DISPLAY is usually set in graphical environments
        # If DISPLAY exists (Linux/macOS) or on Windows, show the graphical result
        try:
            with degirum_tools.Display("AI Camera") as display:
                display.show_image(inference_result)  # Graphical results
        except Exception as e:
            print(f"Error displaying results: {e}")
    else:
        print("No display found. Skipping graphical output.")
