import argparse
import os
import torch

from main import build_model_main
from util import MODEL_DIR, DEFAULT_CONF, DATA_DIR
from util.slconfig import SLConfig
from util.visualizer import COCOVisualizer
from PIL import Image
import datasets.transforms as T

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    type=str,
    default="main_model",
    help="Name of the model to use for inference (name of the folder containing checkpoints inside logs/)",
)
parser.add_argument(
    "--epoch",
    type=str,
    default="",
    help='epoch number, "" for latest, 0000 for first'
)
parser.add_argument(
    "--data_folder_name",
    type=str,
    default="eida_dataset",
    help="Name of the dataset on which to run inference (name of the folder containing images inside data/)",
)
parser.add_argument(
    "--threshold",
    type=str,
    default="0.01",
    help="threshold for predictions, format: 0.01",
)

k2prim = {0: 'line', 1: 'circle', 2: 'arc'}

def generate_prediction(img_path, model, threshold=0.3):
    print(f"Processing {os.path.basename(img_path)} ⚙️")

    image = Image.open(img_path).convert("RGB")

    with torch.no_grad():
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image, _ = transform(image, None)
        size = torch.Tensor([image.shape[1], image.shape[2]])

        output = model.cuda()(image[None].cuda())
        output = postprocessors['param'](output, torch.Tensor([[1.0, 1.0]]).cuda(), to_xyxy=False)[0]

        scores = output['scores']
        labels = output['labels']
        boxes = output['parameters']

        pred_dict = {
            'parameters': boxes[(scores > threshold)],
            'size': size,
            'labels': [k2prim[int(item)] for item in labels[(scores > threshold)]],
            'scores': scores,
        }

    return image, pred_dict


def save_pred_as_img(img_path, img, preds: dict, pred_dir, w_box=False, w_text=False, w_img=True, dpi=150, l_width=1):
    vslzr = COCOVisualizer()
    vslzr.visualize(
        img,
        preds,
        primitives_to_show=['line', 'circle', 'arc'],
        show_boxes=w_box,
        show_text=w_text,
        show_image=w_img,
        savedir=pred_dir,
        img_name=os.path.basename(img_path),
        dpi=dpi,
        linewidth=l_width
    )


if __name__ == "__main__":
    args = parser.parse_args()

    model_folder = MODEL_DIR / args.model_name
    dataset_folder = DATA_DIR / args.data_folder_name
    epoch = args.epoch

    img_folder = dataset_folder / "images"
    pred_folder = dataset_folder / f"preds_{args.model_name}{epoch}"

    config_path = f"{model_folder}/config_cfg.py" if os.path.isfile(f"{model_folder}/config_cfg.py") else DEFAULT_CONF
    model_checkpoint_path = f"{model_folder}/checkpoint{epoch}.pth"

    config = SLConfig.fromfile(config_path)
    config.device = 'cuda'

    model, criterion, postprocessors = build_model_main(config)
    checkpoint = torch.load(model_checkpoint_path, map_location='cuda')
    model.load_state_dict(checkpoint['model'])
    _ = model.eval()

    img_paths = list(img_folder.glob('*.jpg'))
    for path in img_paths:
        img, preds = generate_prediction(path, model)
        save_pred_as_img(path, img, preds, pred_folder)
