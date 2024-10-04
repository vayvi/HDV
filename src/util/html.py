import os
import argparse

# from src.util import DATA_DIR
from pathlib import Path
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

HTML_PATH = DATA_DIR / "html"


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_set",
    type=str,
    help="Data set name to visualize inferences",
)
parser.add_argument(
    "--filename",
    type=str,
    help="Name of the output HTML file",
)


def check_output_file(path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.isdir(path):
        path = f"{path}/output.html"
    return path


class HTML:
    def __init__(self, path=HTML_PATH, title=""):
        self.path = check_output_file(path)
        self.title = title
        try:
            with open(path, 'w') as f:
                f.write('<!DOCTYPE html>')
            self.f = open(path, 'a')
            self.f.write(f"<html data-theme='light'><head><title>{title}</title>" +
            """<style>
                .flex-container {
                    display: flex;
                    flex-wrap: nowrap;
                    overflow-x: auto;
                    flex-direction: column;
                }
                tr>th:first-child, tr>td:first-child {
                    position: sticky; 
                    left: 0;
                }
                td {
                    background-color: grey;
                }
                img {
                    max-width: 150px !important;
                    max-height: 150px !important;
                    width: auto; 
                    height: auto;
                }
                #table-container {
                  max-height: 90vh;
                  position: relative;
                  overflow: auto;
                }
                .table-header {
                  position: sticky;
                  top: 0;
                  z-index: 99;
                  background-color: #eaebec !important;
                  border-bottom: 1px solid #dbdbdb;
                }
                .is-center {
                    margin: auto;
                    text-align: center;
                }
            </style>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bulma@1.0.0/css/bulma.min.css">
            <link rel="icon" type="image/png" href="https://em-content.zobj.net/source/joypixels/257/test-tube_1f9ea.png"/>
            </head>
            <body>
            <div class="container is-fluid">
            """)
        except FileNotFoundError:
            print("Provided file doesn't exist.")

    def fname(self):
        return self.path

    def add_image(self, img_path, **kwargs):
        if 'style' in kwargs.keys():
            style = kwargs['style']
        else:
            style = ''
        self.f.write(f"<img src='{img_path}' style='{style}'>")

    def add_text(self, text):
        self.f.write(f'<p>{text}</p>\n')

    def start_table(self):
        self.f.write('<div id="table-container">\n<table class="table is-striped is-center">\n')

    def end_table(self):
        self.f.write('</table>\n</div>')

    def start_row(self):
        self.f.write('<tr>')

    def end_row(self):
        self.f.write('</tr>\n')

    def add_raw(self, text=''):
        self.f.write(text)

    def add_heading(self, heading, tag='h1'):
        self.f.write(f"<{tag} class='title pt-2'>{heading}</{tag}>")

    def add_cell(self, context='text', **kwargs):
        self.f.write('<td>')
        if context == 'image':
            self.add_image(**kwargs)
        else:  # Text
            self.f.write(kwargs['text'])
        self.f.write('</td>')

    def close_file(self):
        self.f.write("</div></body></html>")
        self.f.close()


def get_jpgs(folder_path):
    return [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]


def gen_html(filename, data_set):
    dataset_path = DATA_DIR / data_set
    orig_imgs = dataset_path / "images"

    inf_folders = [f for f in [fold for fold in os.listdir(dataset_path)] if f.startswith("img_preds")]

    html = HTML(dataset_path / f"{filename}.html", "Vectorisation finetuning")
    html.add_heading(f"Vectorisation finetuning experiments", tag="h1")

    # html.add_raw('<div class="flex-container">\n')
    html.start_table()

    html.add_raw(f"""<thead class='table-header'><tr>
                        <th class='first-col'>Original</th>
                        {''.join([f"<th>{folder.replace('img_preds_', '')}</th>" for folder in inf_folders])}
                     </tr></thead>
                 """)
    html.add_raw("<tbody>")

    for img in get_jpgs(orig_imgs):
        html.start_row()
        html.add_raw(f"""<th class='first-col'><div class="flex-container">
                            <img src='images/{img}'><br>
                            <span style="display:block">{img.split("/")[-1]}</span>
                        </div></th>
                     """)

        for folder in inf_folders:
            # model_name = folder.replace("img_preds_", "")
            # <b>{model_name}</b><br>
            html.add_raw(f"""<td><img src='{folder}/{img}'></td>""")
        html.end_row()

    html.add_raw("</tbody>")
    html.end_table()
    # html.add_raw("</div>\n")
    html.close_file()

if __name__ == "__main__":
    args = parser.parse_args()
    html_file = args.filename if args.filename else args.data_set
    gen_html(html_file, args.data_set)