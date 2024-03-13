import os
import shutil
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, default="data", help="data folder")
parser.add_argument(
    "--input_folder", type=str, default="clean_diagrams_raw", help="input folder"
)
parser.add_argument(
    "--image_file_extension", type=str, default="jpg", help="image file extension"
)

parser.add_argument("--exist_ok", action="store_true")

# Define the folder path
if __name__ == "__main__":
    args = parser.parse_args()
    root_dir = Path(args.data_folder)
    parent_folder_path = root_dir / args.input_folder
    folder_path = parent_folder_path / "images_and_svgs"

    os.makedirs(parent_folder_path / "svgs", exist_ok=args.exist_ok)
    os.makedirs(parent_folder_path / "images", exist_ok=args.exist_ok)
    number_of_files_in_folder = len(os.listdir(folder_path))
    counter = 0

    for filename in os.listdir(folder_path):
        # Check if the file is a ground truth corrected annotation of the format filename-corr.svg
        if filename.endswith("corr.svg"):
            counter += 1
            # Construct the paths to the SVG and PNG files
            svg_path = os.path.join(folder_path, filename)
            png_path = os.path.join(
                folder_path, filename[:-9] + f".{args.image_file_extension}"
            )

            # Check if the PNG file exists
            if os.path.exists(png_path):
                # Copy the SVG and PNG files to a new folder
                shutil.copy2(svg_path, parent_folder_path / "svgs")
                shutil.copy2(png_path, parent_folder_path / "images")
            else:
                print(
                    f"{args.image_file_extension} file does not exist for ",
                    os.path.basename(filename),
                )
    print("Number of files in folder: ", number_of_files_in_folder)
    print("Number of files copied: ", counter)
