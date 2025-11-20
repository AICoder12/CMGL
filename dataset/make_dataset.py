import sys
import shutil
import os
from datasets.VisA import Visa_dataset
from datasets.MVTec import Mvtec_dataset
from datasets.BTAD import BTAD_dataset
from datasets.KSDD2 import KSDD2_dataset
from datasets.RSDD import RSDD_dataset
from datasets.DAGM import DAGM_dataset
from datasets.DTD import DTD_dataset




def move(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def process_dataset(dataset_cls, src_root, des_root, id_start=0, binary=True, to_255=True):

    move(des_root)
    dataset = dataset_cls(src_root)
    return dataset.make_VAND(binary=binary, to_255=to_255, des_path_root=des_root, id=id_start)


if __name__ == "__main__":
    id_counter = 0


    datasets_config = [
        {   
            "name": "visa",
            "class": Visa_dataset,
            "src": "Path to your root/visa",
            "des": "./dataset/mvisa/data/visa"
        },
        { 
            "name": "mvtec",
            "class": Mvtec_dataset,
            "src": "Path to your root/MVTec",
            "des": "./dataset/mvisa/data/mvtec"
        },
        {
            "name": "BTAD",
            "class": BTAD_dataset,
            "src": "/home/data/fty/ZAS/Dataset/BTech_Dataset_transformed",
            "des": "./dataset/mvisa/data/BTAD"
        },
        {
            "name": "KSDD2",
            "class": KSDD2_dataset,
            "src": "/home/data/fty/ZAS/Dataset/KSDD2",
            "des": "./dataset/mvisa/data/KSDD2"
        },
        {
            "name": "RSDD",
            "class": RSDD_dataset,
            "src": "/home/data/fty/ZAS/Dataset/RSDD",
            "des": "./dataset/mvisa/data/RSDD"
        },
        {
            "name": "DAGM",
            "class": DAGM_dataset,
            "src": "/home/data/fty/ZAS/Dataset/DAGM2007",
            "des": "./dataset/mvisa/data/DAGM"
        },
        {
            "name": "DTD",
            "class": DTD_dataset,
            "src": "/home/data/fty/ZAS/Dataset/DTDSynthetic",  
            "des": "./dataset/mvisa/data/DTD"
        },

    ]

    for config in datasets_config:
        print(f"Processing {config['name']}...")
        id_counter = process_dataset(
            dataset_cls=config["class"],
            src_root=config["src"],
            des_root=config["des"],
            id_start= 0
        )
        print(f"Finished {config['name']}, next ID: {id_counter}")

    