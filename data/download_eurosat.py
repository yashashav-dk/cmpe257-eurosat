"""
Download and extract EuroSAT RGB.

Source: https://zenodo.org/record/7711810/files/EuroSAT_RGB.zip
Usage: python data/download_eurosat.py
"""
import os
import zipfile
import urllib.request
import shutil

URL = "https://zenodo.org/record/7711810/files/EuroSAT_RGB.zip"
DEST = os.path.join(os.path.dirname(__file__), "EuroSAT_RGB")
ZIP_PATH = os.path.join(os.path.dirname(__file__), "EuroSAT_RGB.zip")


def main():
    if os.path.isdir(DEST):
        n = len([d for d in os.listdir(DEST) if os.path.isdir(os.path.join(DEST, d))])
        if n == 10:
            print(f"Dataset already at {DEST} (10 classes)")
            return

    print(f"Downloading {URL}")
    urllib.request.urlretrieve(URL, ZIP_PATH)
    print(f"  size: {os.path.getsize(ZIP_PATH)/1e6:.1f} MB")

    print("Extracting ...")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(os.path.dirname(__file__))
    os.remove(ZIP_PATH)

    # Some Zenodo archives extract to EuroSAT_RGB/2750/<class>; flatten
    nested = os.path.join(DEST, "2750")
    if os.path.isdir(nested):
        for entry in os.listdir(nested):
            shutil.move(os.path.join(nested, entry), os.path.join(DEST, entry))
        os.rmdir(nested)

    classes = sorted([d for d in os.listdir(DEST) if os.path.isdir(os.path.join(DEST, d))])
    total = sum(len(os.listdir(os.path.join(DEST, c))) for c in classes)
    print(f"Done: {len(classes)} classes, {total} images")
    assert len(classes) == 10 and total >= 26000


if __name__ == "__main__":
    main()
