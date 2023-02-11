import hashlib
from collections import defaultdict
from itertools import combinations, product

import numpy as np
import tqdm


class Imdl:
    """
    Is My Data Leaking?
    A simple class for checking if there are duplicate samples for very big datasets.
    Hashes are used for avoiding loading the entire dataset into memory
    """

    def __init__(self, *datasets, transform=None, progress_bar: bool = False):
        """
        Constructor and hash pre-computer
        :param datasets: multiple datasets of iterables
        :param transform: function for turning output of dataset to numpy array
        :param progress_bar: bool
        """
        self.datasets = datasets
        if transform is None:
            transform = lambda x: x
        self.transform = [transform] * len(datasets)
        self.hash_sets = []
        datasets = datasets if not progress_bar else tqdm.tqdm(datasets, desc='computing hashes from datasets')
        for d_ix, d in enumerate(datasets):
            hash_set = self._compute_hash_set(d, self.transform[d_ix])
            self.hash_sets.append(hash_set)

    def add_dataset(self, dataset, transform=None):
        """
        Add single dataset
        :param dataset: dataset
        :param transform: function for turning output of dataset to numpy array
        :return: None
        """
        self.datasets.append(dataset)
        if transform is None:
            transform = lambda x: x
        self.transform.append(transform)
        self.hash_sets.append(self._compute_hash_set(dataset, transform))

    def find_duplicates(self, progress_bar=False):
        """
        Find duplicate samples from datasets
        :param progress_bar: bool
        :return: np.array containing pairs of duplicates [N x 2(Dataset_Id) x 2(Image_Id)]
        """
        duplicates = []
        comb = combinations(enumerate(self.hash_sets), 2)
        comb = comb if not progress_bar else tqdm.tqdm(comb, desc='verifying duplicate hashes')
        for (dataset_id_1, h1), (dataset_id_2, h2) in comb:
            duplicate_candidates = h1.keys() & h2.keys()
            for d_cand in duplicate_candidates:
                id1 = h1[d_cand]
                id2 = h2[d_cand]
                for image_id_1, image_id_2 in product(id1, id2):
                    duplicate = self.transform[dataset_id_1](self.datasets[dataset_id_1][image_id_1]) == \
                                self.transform[dataset_id_2](self.datasets[dataset_id_2][image_id_2])
                    if np.all(duplicate):
                        duplicates.append(((dataset_id_1, image_id_1), (dataset_id_2, image_id_2)))
        return np.array(duplicates)

    @staticmethod
    def _get_hash_numpy(data: np.ndarray):
        return hashlib.sha1(np.ascontiguousarray(data.view(np.uint8))).hexdigest()

    @staticmethod
    def _compute_hash_set(data, transform):
        hash_set = defaultdict(lambda: [])
        for i in range(len(data)):
            numpy_image = transform(data[i])
            hsh = Imdl._get_hash_numpy(numpy_image)
            hash_set[hsh].append(i)
        return hash_set
