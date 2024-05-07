import os, sys
import torch, json
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from util import MODEL_DIR, DATA_DIR
from util import ROOT_DIR
from util.slconfig import SLConfig
from models.registry import MODULE_BUILD_FUNCS
import datasets.transforms as T

from PIL import Image
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    type=str,
    default="main_model",
    help="model name",
)
parser.add_argument(
    "--epoch", type=str, default="", help='epoch number, "" for latest, 0000 for first'
)
parser.add_argument(
    "--threshold",
    type=str,
    default="0.01",
    help="threshold for predictions, string",
)
parser.add_argument(
    "--data_folder_name",
    type=str,
    default="eida_dataset",
    help="root directory of the data",
)

id2name = {0: "line", 1: "circle", 2: "arc"}


def scale_positions(lines, heatmap_scale=(128, 128), im_shape=None):
    if len(lines) == 0:
        return []
    fx, fy = heatmap_scale[0] / im_shape[1], heatmap_scale[1] / im_shape[0]

    lines[:, :, 0] = np.clip(lines[:, :, 0] * fx, 0, heatmap_scale[0] - 1e-4)
    lines[:, :, 1] = np.clip(lines[:, :, 1] * fy, 0, heatmap_scale[1] - 1e-4)

    return lines


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors


def get_outputs_per_class(pred_dict):
    mask = pred_dict["labels"] == 0
    lines, line_scores = pred_dict["parameters"][mask][:, :4], pred_dict["scores"][mask]
    mask = pred_dict["labels"] == 1
    circles, circle_scores = (
        pred_dict["parameters"][mask][:, 4:8],
        pred_dict["scores"][mask],
    )
    mask = pred_dict["labels"] == 2
    arcs, arc_scores = pred_dict["parameters"][mask][:, 8:14], pred_dict["scores"][mask]
    lines, line_scores = lines.cpu().numpy(), line_scores.cpu().numpy()
    circles, circle_scores = circles.cpu().numpy(), circle_scores.cpu().numpy()
    arcs, arc_scores = arcs.cpu().numpy(), arc_scores.cpu().numpy()
    return lines, line_scores, circles, circle_scores, arcs, arc_scores


def main(
    root_dir=ROOT_DIR,
    model_name="main_model",
    data_folder_name="eida_dataset",
    threshold=0.3,
    epoch="",
):
    # load model
    model_folder = MODEL_DIR / f"{model_name}"
    encoder_only = "encoder-only" in model_name
    print("encoder_only", encoder_only)
    model_config_path = model_folder / "config_cfg.py"

    model_checkpoint_path = model_folder / f"checkpoint{epoch}.pth"
    npz_dir = model_folder / f"npz_preds{epoch}"
    os.makedirs(npz_dir, exist_ok=True)

    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda"
    model, _, postprocessors = build_model_main(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    _ = model.eval()

    # load all image paths
    images_folder_path = DATA_DIR / data_folder_name / "images"
    file_extension = ".jpg"
    image_paths = sorted(glob.glob(f"{images_folder_path}/*{file_extension}"))
    print(f"{images_folder_path}: Found {len(image_paths)} images")

    batch = "valid"
    anno_file = DATA_DIR / f"{data_folder_name}/{batch}.json"
    with open(anno_file, "r") as f:
        dataset = json.load(f)

    # heatmap_scale = (512, 512)
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # iterate over images
    for image_path in image_paths:
        im_name = os.path.basename(image_path).split(".")[0]
        print(im_name)
        for data in dataset:
            if data["filename"] == f"{im_name}{file_extension}":
                break
        else:
            print(f"##### Image {im_name} not found")
            # raise ValueError(f"Could not find image {im_name} in dataset")
            continue

        heatmap_scale = (128, 128)
        # Each image can have a different aspect ratio, we rescale the image to a fixed size (512, 512)
        # and then resize the heatmap to (128, 128) comparison.
        image = Image.open(image_path).convert("RGB")  # load image
        size = image.size
        im_shape = size
        image, _ = transform(image, None)
        # size = torch.Tensor([image.shape[1], image.shape[2]])

        torch.cuda.empty_cache()
        with torch.no_grad():
            output = model.cuda()(image[None].cuda())
        if encoder_only:
            output = output["interm_outputs"]
        output = postprocessors["param"](
            output, torch.Tensor([[size[0], size[1]]]).cuda(), to_xyxy=True
        )[0]
        scores = output["scores"]
        labels = output["labels"]
        boxes = output["parameters"]
        select_mask = scores > threshold
        box_label = labels[select_mask]
        boxes = boxes[select_mask]
        scores = scores[select_mask]
        pred_dict = {
            "parameters": boxes,
            "size": size,
            "labels": box_label,
            "scores": scores,
        }
        (
            lines,
            line_scores,
            circles,
            circle_scores,
            arcs,
            arc_scores,
        ) = get_outputs_per_class(pred_dict)
        good_circles = circles.reshape(-1, 2, 2)
        good_lines = lines.reshape(-1, 2, 2)
        good_arcs = arcs.reshape(-1, 3, 2)

        pos_l = scale_positions(good_lines.copy(), heatmap_scale, im_shape)
        pos_c = scale_positions(good_circles.copy(), heatmap_scale, im_shape)
        pos_a = scale_positions(good_arcs.copy(), heatmap_scale, im_shape)

        np.savez(
            npz_dir / f"{im_name}.npz",
            lines=pos_l,
            line_scores=line_scores,
            circles=pos_c,
            circle_scores=circle_scores,
            arcs=pos_a,
            arc_scores=arc_scores,
        )
        print(f"Saved {im_name}.npz")

        # save preds is npz format


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(
        ROOT_DIR,
        model_name=args.model_name,
        data_folder_name=args.data_folder_name,
        threshold=float(args.threshold),
        epoch=args.epoch,
    )
