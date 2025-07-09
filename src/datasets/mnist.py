import numpy as np
import torch
import torchvision
import safetensors
import safetensors.torch
import shutil
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class MNISTDataset(BaseDataset):
    def __init__(
        self, name="train", *args, **kwargs
    ):
        """
        Args:
            input_length (int): length of the random vector.
            n_classes (int): number of classes.
            dataset_length (int): the total number of elements in
                this random dataset.
            name (str): partition name
        """
        index_path = ROOT_PATH / "data" / "mnist" / name / "index.json"

        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(name)

        super().__init__(index, *args, **kwargs)

    def _create_index(self, name):
        """
        Create index for the dataset. The function processes dataset metadata
        and utilizes it to get information dict for each element of
        the dataset.

        Args:
            input_length (int): length of the random vector.
            n_classes (int): number of classes.
            dataset_length (int): the total number of elements in
                this random dataset.
            name (str): partition name
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        index = []
        data_path = ROOT_PATH / "data" / "mnist" / name
        data_path.mkdir(exist_ok=True, parents=True)

        data = torchvision.datasets.MNIST(
            str(data_path), train=(name == "train"), download=True, transform=torchvision.transforms.ToTensor()
        )

        print("Creating Example Dataset")
        for i in tqdm(range(len(data))):
            # create dataset
            img, label = data[i]

            element_path = data_path / f"{i:06}.safetensors"
            element = {"tensor": img}
            safetensors.torch.save_file(element, element_path)

            index.append({"path": str(element_path), "label": label})

        shutil.rmtree(data_path / "MNIST")
        write_json(index, str(data_path / "index.json"))

        return index
