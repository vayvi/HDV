import argparse
import os
import time
from pathlib import Path

from PIL import Image
import numpy as np
import torch
import svgwrite

from main import build_model_main
from util.logger import SLogger
from util import MODEL_DIR, DEFAULT_CONF, DATA_DIR
from util.slconfig import SLConfig, DictAction
from util.visualizer import COCOVisualizer
from util.primitives import PRIM_INFO, get_arc_param, write_svg_dwg, line_to_xy, circle_to_xy, arc_to_xy, remove_duplicate_lines, \
    remove_small_lines, remove_duplicate_circles, remove_duplicate_arcs, remove_arcs_on_top_of_circles, \
    remove_arcs_on_top_of_lines
import datasets.transforms as T

import xml.etree.ElementTree as ET
import re
import shutil

from svg.path import parse_path
from svg.path.path import Line, Arc


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
parser.add_argument(
    "--export_formats",
    default="img+npy+svg",
    help="Format to export the predictions in, default: img, svg, npz",
)

prim_list = list(PRIM_INFO.keys())

def preprocess_img(img_path):
    image = Image.open(img_path).convert("RGB")
    with torch.no_grad():
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        tr_image, _ = transform(image, None)
    return image, tr_image

def scale_positions(prims, heatmap_scale=(128, 128), im_shape=None):
    if len(prims) == 0:
        return []
    fx, fy = heatmap_scale[0] / im_shape[1], heatmap_scale[1] / im_shape[0]

    prims[:, :, 0] = np.clip(prims[:, :, 0] * fx, 0, heatmap_scale[0] - 1e-4)
    prims[:, :, 1] = np.clip(prims[:, :, 1] * fy, 0, heatmap_scale[1] - 1e-4)

    return prims


def prim_to_xy(prim_type, p_preds):
    p_info = PRIM_INFO[prim_type]
    p_preds = p_preds[:, p_info["indices"]].cpu().numpy()

    if prim_type == "line":
        p_preds = np.array([line_to_xy(p) for p in p_preds])
    elif prim_type == "circle":
        p_preds = np.array([circle_to_xy(p) for p in p_preds])
    elif prim_type == "arc":
        p_preds = np.array([arc_to_xy(p) for p in p_preds])

    return p_preds.reshape(p_info["param_shape"])


def pred_to_dict(preds):
    prim_dict = {}
    for prim_k, prim_type in enumerate(prim_list):
        mask = preds["labels"] == prim_k
        prim_dict.update({
            f"{prim_type}s": prim_to_xy(prim_type, preds["parameters"][mask]),
            f"{prim_type}_scores": preds["scores"][mask].cpu().numpy()
        })
    return prim_dict

def process_preds(preds, prim_type, img_size):
    p_pred, p_score = preds[f"{prim_type}s"], preds[f"{prim_type}_scores"]

    if prim_type == "line":
        p_pred, p_score = remove_duplicate_lines(p_pred, img_size, p_score)
        p_pred, p_score = remove_small_lines(p_pred, img_size, p_score)
    elif prim_type == "circle":
        p_pred, p_score = remove_duplicate_circles(p_pred, img_size, p_score)
    elif prim_type == "arc":
        p_pred, p_score = remove_duplicate_arcs(p_pred, img_size, p_score)
        p_pred, p_score = remove_arcs_on_top_of_circles(p_pred, preds["circles"], img_size, p_score)
        p_pred, p_score = remove_arcs_on_top_of_lines(p_pred, preds["lines"], img_size, p_score)

    preds[f"{prim_type}s"] = p_pred
    preds[f"{prim_type}_scores"] = p_score
    return preds

def generate_prediction(img, processed_img, model, threshold=0.3):
    with torch.no_grad():
        out_size = torch.Tensor([[img.size[1], img.size[0]]]) # torch.Tensor([[1.0, 1.0]])

        output = model.cuda()(processed_img[None].cuda())
        output = postprocessors['param'](output, out_size.cuda(), to_xyxy=False)[0]

        scores = output['scores']

        # l_filter = (scores > 0.3) & (labels == 0)
        # c_filter = (scores > 0.3) & (labels == 1)
        # a_filter = (scores > 0.3) & (labels == 2)
        p_filter = scores > threshold # (l_filter | c_filter | a_filter)

        out_pred = {
            'parameters': output['parameters'][p_filter],
            'size': torch.Tensor([processed_img.shape[1], processed_img.shape[2]]),
            'labels': output['labels'][p_filter],
            'scores': scores[p_filter],
        }

    return out_pred


def postprocess_preds(model_preds, img_size):
    pred_dict = pred_to_dict(model_preds)

    for prim_type in prim_list:
        pred_dict = process_preds(pred_dict, prim_type, img_size)

    return pred_dict

def save_pred_as_img(img_name, img, preds: dict, pred_dir, w_box=False, w_text=False, w_img=True, dpi=150, l_width=1):
    vslzr = COCOVisualizer()
    vslzr.visualize(
        img,
        preds,
        primitives_to_show=prim_list,
        show_boxes=w_box,
        show_text=w_text,
        show_image=w_img,
        savedir=pred_dir,
        img_name=f"{img_name}.jpg",
        dpi=dpi,
        linewidth=l_width
    )


def save_pred_as_npz(img_name, pred_dict, pred_dir):
    np.savez(
        pred_dir / f"{img_name}.npz",
        **pred_dict
    )


def predict(image_path, output_dir, model, logger):
    im_name = Path(image_path).stem
    image = Image.open(image_path).convert("RGB")  # load image
    orig_img_size = image.size
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tr_img, _ = transform(image, None)
    size = torch.Tensor([tr_img.shape[1], tr_img.shape[2]])
    out_size = torch.Tensor([[orig_img_size[1], orig_img_size[0]]])

    # output
    output = model.cuda()(tr_img[None].cuda())
    output = postprocessors['param'](output, out_size.cuda(), to_xyxy=False)[0]

    threshold, arc_threshold = 0.3, 0.3
    scores = output['scores']
    labels = output['labels']
    boxes = output['parameters']
    select_mask = ((scores > threshold) & (labels != 2)) | ((scores > arc_threshold) & (labels == 2))
    labels = labels[select_mask]
    boxes = boxes[select_mask]
    scores = scores[select_mask]
    pred_dict = {'parameters': boxes, 'labels': labels, 'scores': scores}
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
    lines = np.array([line_to_xy(x) for x in lines])
    circles = np.array([circle_to_xy(x) for x in circles])
    arcs = np.array([arc_to_xy(x) for x in arcs])

    # some duplicate postprocessing
    lines, line_scores = remove_duplicate_lines(lines, orig_img_size, line_scores)
    lines, line_scores = remove_small_lines(lines, orig_img_size, line_scores)
    circles, circle_scores = remove_duplicate_circles(circles, orig_img_size, circle_scores)
    arcs, arc_scores = remove_duplicate_arcs(arcs, orig_img_size, arc_scores)
    arcs, arc_scores = remove_arcs_on_top_of_circles(arcs, circles, orig_img_size, arc_scores)
    arcs, arc_scores = remove_arcs_on_top_of_lines(arcs, lines, orig_img_size, arc_scores)

    lines = lines.reshape(-1, 2, 2)
    arcs = arcs.reshape(-1, 3, 2)

    size = [tr_img.shape[1], tr_img.shape[2]]

    dwg = svgwrite.Drawing(str(output_dir / f"{im_name}.svg"), profile="tiny", size=size)
    dwg.add(dwg.image(href=f"{im_name}.jpg", insert=(0, 0), size=size))
    dwg = write_svg_dwg(dwg, lines, circles, arcs, show_image=False, image=None)
    dwg.save(pretty=True)
    # break

    ET.register_namespace('', "http://www.w3.org/2000/svg")
    ET.register_namespace('xlink', "http://www.w3.org/1999/xlink")
    ET.register_namespace('sodipodi', "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd")
    ET.register_namespace('inkscape', "http://www.inkscape.org/namespaces/inkscape")

    input_folder = output_dir
    file_name = output_dir / f"{im_name}.svg"
    tree = ET.parse(file_name)
    root = tree.getroot()

    root.set('xmlns:inkscape', 'http://www.inkscape.org/namespaces/inkscape')
    root.set('xmlns:sodipodi', 'http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd')
    root.set('inkscape:version', '1.3 (0e150ed, 2023-07-21)')

    # Regular expression to match the 'a' or 'A' command in the 'd' attribute
    arc_regex = re.compile(r'[aA]')

    # Iterate over all 'path' elements
    for path in root.findall('{http://www.w3.org/2000/svg}path'):
        # Get the 'd' attribute
        d = path.get('d', '')

        # If the 'd' attribute contains an arc
        if arc_regex.search(d):
            # Add the 'sodipodi:type' and 'sodipodi:arc-type' attributes
            path.set('sodipodi:type', 'arc')
            path.set('sodipodi:arc-type', 'arc')
            path_parsed = parse_path(d)

            for e in path_parsed:
                if isinstance(e, Line):
                    continue
                elif isinstance(e, Arc):
                    center, radius, start_angle, end_angle, p0, p1 = get_arc_param([e])
                    path.set('sodipodi:cx', f'{center[0]}')
                    path.set('sodipodi:cy', f'{center[1]}')
                    path.set('sodipodi:rx', f'{radius}')
                    path.set('sodipodi:ry', f'{radius}')
                    path.set('sodipodi:start', f'{start_angle}')
                    path.set('sodipodi:end', f'{end_angle}')

    # Write the changes back to the file
    tree.write(file_name, xml_declaration=True)
    return {
        "lines": lines,
        "line_scores": line_scores,
        "circles": circles,
        "circle_scores": circle_scores,
        "arcs": arcs,
        "arc_scores": arc_scores,
    }


def save_pred_as_svg(img_path, img_name, img_size, pred_dict, pred_dir):
    lines, circles, arcs = pred_dict["lines"], pred_dict["circles"], pred_dict["arcs"]
    svg_file = pred_dir / f"{img_name}.svg"

    dwg = svgwrite.Drawing(svg_file, profile="tiny", size=img_size)
    dwg.add(dwg.image(href=f"{img_name}.jpg", insert=(0, 0), size=img_size))
    dwg = write_svg_dwg(dwg, lines, circles, arcs, show_image=False, image=None)
    dwg.save(pretty=True)

    ET.register_namespace('', "http://www.w3.org/2000/svg")
    ET.register_namespace('xlink', "http://www.w3.org/1999/xlink")
    ET.register_namespace('sodipodi', "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd")
    ET.register_namespace('inkscape', "http://www.inkscape.org/namespaces/inkscape")

    tree = ET.parse(svg_file)
    root = tree.getroot()

    root.set('xmlns:inkscape', 'http://www.inkscape.org/namespaces/inkscape')
    root.set('xmlns:sodipodi', 'http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd')
    root.set('inkscape:version', '1.3 (0e150ed, 2023-07-21)')

    # Regular expression to match the 'a' or 'A' command in the 'd' attribute
    arc_regex = re.compile(r'[aA]')
    for d_path in root.findall('{http://www.w3.org/2000/svg}path'):
        # Get the 'd' attribute
        d = d_path.get('d', '')

        # If the 'd' attribute contains an arc
        if arc_regex.search(d):
            # Add the 'sodipodi:type' and 'sodipodi:arc-type' attributes
            d_path.set('sodipodi:type', 'arc')
            d_path.set('sodipodi:arc-type', 'arc')
            path_parsed = parse_path(d)

            for e in path_parsed:
                if isinstance(e, Line):
                    continue
                elif isinstance(e, Arc):
                    center, radius, start_angle, end_angle, p0, p1 = get_arc_param([e])
                    d_path.set('sodipodi:cx', f'{center[0]}')
                    d_path.set('sodipodi:cy', f'{center[1]}')
                    d_path.set('sodipodi:rx', f'{radius}')
                    d_path.set('sodipodi:ry', f'{radius}')
                    d_path.set('sodipodi:start', f'{start_angle}')
                    d_path.set('sodipodi:end', f'{end_angle}')

        # Write the changes back to the file
    tree.write(svg_file, xml_declaration=True)


def set_config(conf_path):
    conf = SLConfig.fromfile(conf_path)
    conf.device = 'cuda'
    conf.dataset_file = 'synthetic'
    conf.mode = "primitives"
    conf.relative = False
    conf.common_queries = True
    conf.eval = True
    conf.coco_path = f"{DATA_DIR}/synthetic_processed"  # the path of coco
    conf.fix_size = False
    conf.batch_size = 1
    conf.boxes_only = False
    return conf


if __name__ == "__main__":
    args = parser.parse_args()

    formats = args.export_formats.split("+")

    model_folder = MODEL_DIR / args.model_name
    dataset_folder = DATA_DIR / args.data_folder_name
    epoch = args.epoch

    img_folder = dataset_folder / "images"

    config_path = f"{model_folder}/config_cfg.py" if os.path.isfile(f"{model_folder}/config_cfg.py") else DEFAULT_CONF
    model_checkpoint_path = f"{model_folder}/checkpoint{epoch}.pth"

    config = set_config(config_path)

    model, criterion, postprocessors = build_model_main(config)
    checkpoint = torch.load(model_checkpoint_path, map_location='cuda')
    model.load_state_dict(checkpoint['model'])
    _ = model.eval()

    for out in formats:
        # create or overwrite existing folders
        os.makedirs(dataset_folder / f"{out}_preds_{args.model_name}{epoch}", exist_ok=True)

    logger = SLogger(
        name="inference",
        log_file=dataset_folder / f"logs_{args.model_name}{epoch}.txt",
    )

    t0 = time.time()
    img_paths = list(img_folder.glob('*.jpg'))
    for path in img_paths:
        filename = Path(path).stem
        t1 = time.time()

        logger.info(f"\n⚙️  Processing {filename} as {' '.join(formats)}...")
        # preds = predict(path, dataset_folder / f"svg_preds_{args.model_name}{epoch}", model, logger)

        orig_img, tr_img = preprocess_img(path)
        preds = generate_prediction(orig_img, tr_img, model)

        if "img" in formats:
            # DO NOT WORK: correct by copying logic of inference notebook
            # for prim in prim_list:
            #     info = prim_info[prim]
            #     pred = preds['parameters'][:, info["indices"]]
            #     pos = scale_positions(pred.reshape(info["param_shape"]).copy(), (128, 128), orig_img.size)
            save_pred_as_img(filename, tr_img, preds, pred_dir=dataset_folder / f"img_preds_{args.model_name}{epoch}")

        preds = postprocess_preds(preds, orig_img.size)

        # logger.info(preds, color="cyan")

        if "npz" in formats:
            save_pred_as_npz(filename, preds, pred_dir=dataset_folder / f"npz_preds_{args.model_name}{epoch}")

        if "svg" in formats:
            save_pred_as_svg(path, filename, orig_img.size, preds, pred_dir=dataset_folder / f"svg_preds_{args.model_name}{epoch}")

        logger.info(f"\n✅  Done processing {filename} in {time.time() - t1:.2f}s")

    logger.info(f"\n🕒  Total time taken: {time.time() - t0:.2f}s")
