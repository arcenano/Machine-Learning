# from loaders.ICDARdataloader import ICDARDataSet
import torch
from easydict import EasyDict  # Handle config file as dictionary
import json  # Load config file
import argparse  # Interface
from FOTS.model.model import FOTSModel
from visualization.plotFilters import plotFilters
from visualization.plotAlexFilters import plot_weights

def main(config):
    resume = config.resume
    input_dir = config.input_dir

    # Check whether GPU is available and can be used
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FOTSModel.load_from_checkpoint(
        checkpoint_path=config.trainer.save_dir, map_location="cpu", config=config
    )
    model = model.to("cuda:0")
    model.eval()
    for image_fn in input_dir.glob("*.jpg"):
        try:
            with torch.no_grad():
                ploy, im = Toolbox.predict(
                    image_fn, model, with_image, output_dir, with_gpu=True
                )
                print(len(ploy))
        except Exception as e:
            traceback.print_exc()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PyTorch Template")
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args = parser.parse_args()

    config = json.load(open(args.config))

    config = EasyDict(config)
    print(config)

    main(config)
