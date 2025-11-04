# degirum_cli/predict_video.py

import degirum as dg
import degirum_tools


def run_inference(
    inference_host_address,
    model_zoo_url,
    model_name,
    video_source,
    token=None,
    **kwargs,
):
    """Run AI inference on a video stream."""
    print("Running inference with the following parameters:")
    print(f"  Inference Host Address: {inference_host_address}")
    print(f"  Model Zoo URL: {model_zoo_url}")
    print(f"  Model Name: {model_name}")
    print(f"  Video Source: {video_source}")
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
    if video_source.isdigit():
        video_source = int(video_source)
    else:
        video_source = video_source

    # Perform inference
    model = dg.load_model(**model_args)

    # Set the display title to show model_name, video_source, and inference_host_address
    display_title = f"{model_name} running on {video_source} {inference_host_address}"

    # Run AI inference on video stream
    inference_results = degirum_tools.predict_stream(model, video_source)

    # Display the results with a live video stream
    # Press 'x' or 'q' to stop
    with degirum_tools.Display(display_title) as display:
        for inference_result in inference_results:
            display.show(inference_result)
