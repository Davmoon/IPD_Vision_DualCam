import click
import ast  # Import ast to safely evaluate string representations of Python objects
from .predict_image import run_inference as predict_image_main
from .predict_video import run_inference as predict_video_main
from .benchmark import benchmark
from .composition_run import composition_run


@click.group()
def cli():
    """Degirum AI CLI for image and video inference."""
    pass


# Use hyphens in options; Click will convert to underscores internally
@click.command()
@click.option(
    "--inference-host-address",
    default="@cloud",
    show_default=True,
    help="Hardware location for inference (e.g., @cloud, @local, IP).",
)
@click.option(
    "--model-zoo-url",
    default="degirum/public",
    show_default=True,
    help="URL or path to the model zoo.",
)
@click.option(
    "--model-name",
    default="yolov8n_relu6_coco--640x640_quant_n2x_orca1_1",
    show_default=True,
    help="Name of the model to use for inference.",
)
@click.option(
    "--image-source",
    default="https://raw.githubusercontent.com/DeGirum/PySDKExamples/main/images/ThreePersons.jpg",
    show_default=True,
    help="Path or URL to the image for inference.",
)
@click.option(
    "--token",
    help="Cloud platform token to use for inference. Attempts to load from environment if not provided.",
)
@click.argument("extra_args", nargs=-1)  # Capture additional keyword arguments
def predict_image(
    inference_host_address,
    model_zoo_url,
    model_name,
    image_source,
    token,
    extra_args,
):
    """Run AI inference on an image with extra options."""

    # Convert the additional arguments into a dictionary
    kwargs = {}
    for arg in extra_args:
        if "=" in arg:
            key, value = arg.split("=", 1)
            try:
                # Safely evaluate the value to handle lists, tuples, numbers, booleans, etc.
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                # If the value cannot be evaluated, treat it as a string
                pass
            kwargs[key] = value
        else:
            # Handle boolean flags (e.g., measure_time)
            kwargs[arg] = True

    # Call the main function and pass the kwargs
    predict_image_main(
        inference_host_address,
        model_zoo_url,
        model_name,
        image_source,
        token,
        **kwargs,
    )


# Use hyphens in options; Click will convert to underscores internally
@click.command()
@click.option(
    "--inference-host-address",
    default="@cloud",
    show_default=True,
    help="Hardware location for inference (e.g., @cloud, @local, IP).",
)
@click.option(
    "--model-zoo-url",
    default="degirum/public",
    show_default=True,
    help="URL or path to the model zoo.",
)
@click.option(
    "--model-name",
    default="yolov8n_relu6_coco--640x640_quant_n2x_orca1_1",
    show_default=True,
    help="Name of the model to use for inference.",
)
@click.option(
    "--video-source",
    default="https://raw.githubusercontent.com/DeGirum/PySDKExamples/main/images/example_video.mp4",
    show_default=True,
    help="Video source (camera index, URL, or file path).",
)
@click.option(
    "--token",
    help="Cloud platform token to use for inference. Attempts to load from environment if not provided.",
)
@click.argument("extra_args", nargs=-1)  # Capture additional keyword arguments
def predict_video(
    inference_host_address, model_zoo_url, model_name, video_source, token, extra_args
):
    """Run AI inference on a video stream."""

    # Convert the additional arguments into a dictionary
    kwargs = {}
    for arg in extra_args:
        if "=" in arg:
            key, value = arg.split("=", 1)
            try:
                # Safely evaluate the value to handle lists, tuples, numbers, booleans, etc.
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                # If the value cannot be evaluated, treat it as a string
                pass
            kwargs[key] = value
        else:
            # Handle boolean flags (e.g., measure_time)
            kwargs[arg] = True

    predict_video_main(
        inference_host_address,
        model_zoo_url,
        model_name,
        video_source,
        token,
        **kwargs,
    )


@click.command(name="benchmark")
@click.option(
    "--config-file", type=str, help="Optional path to YAML configuration file."
)
@click.option(
    "--inference-host-address",
    default="@cloud",
    show_default=True,
    help="Override the inference host address.",
)
@click.option(
    "--iterations",
    default=100,
    show_default=True,
    help="Number of iterations for the benchmark.",
)
@click.option(
    "--token",
    help="Cloud platform token to use for inference. Attempts to load from environment if not provided.",
)
@click.argument("extra_args", nargs=-1)  # Capture additional keyword arguments
def run_benchmark(config_file, inference_host_address, iterations, token, extra_args):
    """Benchmark AI models."""

    # Convert the additional arguments into a dictionary
    kwargs = {}
    for arg in extra_args:
        if "=" in arg:
            key, value = arg.split("=", 1)
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass
            kwargs[key] = value
        else:
            kwargs[arg] = True

    # Run the benchmark
    benchmark(
        config_file=config_file,
        inference_host_address=inference_host_address,
        iterations=iterations,
        token=token,
        **kwargs,
    )


@click.command()
@click.option(
    "-c",
    "--config-file",
    type=str,
    help="Path to gizmo composition YAML configuration file.",
)
@click.option(
    "--allow-stop",
    is_flag=True,
    help="Allow stopping the composition run from console.",
)
def run_composition(config_file, allow_stop):
    """Run gizmo composition defined in a YAML configuration file."""
    composition_run(config_file, allow_stop)


cli.add_command(predict_image)
cli.add_command(predict_video)
cli.add_command(run_benchmark)
cli.add_command(run_composition)

if __name__ == "__main__":
    cli()
