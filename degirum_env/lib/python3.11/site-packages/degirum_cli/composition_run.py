# degirum_cli/composition_run.py

from degirum_tools import streams


def composition_run(config_file: str, allow_stop: bool):
    """Run gizmo composition defined in a YAML configuration file."""

    print(f"Running gizmo composition defined in {config_file}")

    composition = streams.load_composition(config_file)
    composition.start(wait=False)
    if allow_stop:
        input("\nPress <Enter> to stop the composition...\n")
        composition.stop()
        print("Composition stopped.")
    else:
        composition.wait()
        print("Composition finished.")
