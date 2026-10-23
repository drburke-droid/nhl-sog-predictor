"""Where the pipeline reads inputs and writes renders. Override with BLUR_WORK=/some/dir."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("BLUR_WORK", os.path.join(HERE, "work"))
