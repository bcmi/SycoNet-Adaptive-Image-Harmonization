import os
import cv2
import numpy as np
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from copy import deepcopy


class CUSTOMDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--is_train', type=bool, default=True, help='whether in the training phase')
        parser.set_defaults(max_dataset_size=float("inf"), new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        BaseDataset.__init__(self, opt)
        self.real_paths, self.mask_paths = '', ''
        self.isTrain = opt.isTrain
        self.data_mode = getattr(opt, 'data_mode', 'single')

        if opt.isTrain == False:
            print('loading test file: ')
            if self.data_mode == 'multiple':
                self.real_pairs = self._collect_real_mask_pairs(opt.real, opt.mask)
                self.real_paths = [p[0] for p in self.real_pairs]
                self.mask_paths = [p[1] for p in self.real_pairs]
            else:
                self.real_path = opt.real
                self.mask_path = opt.mask
                self.real_pairs = [(opt.real, opt.mask)]
        else:
            raise NotImplementedError('Sorry, the training code has not been released.')

        self.transform = get_transform(opt)
        self.input_transform = transforms.Compose([
                transforms.ToTensor(),
            ])

    def _collect_real_mask_pairs(self, real_dir, mask_dir):
        real_map = {}
        for root, _, files in os.walk(real_dir):
            for filename in files:
                name, ext = os.path.splitext(filename)
                if not ext.lower() or not name:
                    continue
                real_map[name] = os.path.join(root, filename)

        pairs = []
        for root, _, files in os.walk(mask_dir):
            for filename in files:
                name, _ = os.path.splitext(filename)
                real_path = real_map.get(name)
                if real_path is not None:
                    pairs.append((real_path, os.path.join(root, filename)))
                    del real_map[name]

        if not pairs:
            raise FileNotFoundError(f'No matching files found between real folder "{real_dir}" and mask folder "{mask_dir}".')

        pairs.sort(key=lambda x: x[0])
        return pairs

    def __getitem__(self, index):
        sample = self.get_sample(index)
        self.check_sample_types(sample)
        sample_raw = deepcopy(sample)
        sample = self.augment_sample(sample)
        real = self.input_transform(sample['real'])
        mask = sample['mask'].astype(np.float32)

        real_raw = self.input_transform(sample_raw['real'])
        mask_raw = sample_raw['mask'].astype(np.float32)

        output = {
            'mask': mask[np.newaxis, ...].astype(np.float32),
            'real': real,
            'mask_raw': mask_raw,
            'real_raw': real_raw,
            'img_path':sample['img_path']
        }

        return output


    def check_sample_types(self, sample):
        assert sample['real'].dtype == 'uint8'


    def augment_sample(self, sample):
        if self.transform is None:
            return sample

        additional_targets = {target_name: sample[target_name]
                              for target_name in self.transform.additional_targets.keys()}

        valid_augmentation = False
        while not valid_augmentation:
            aug_output = self.transform(image=sample['real'], **additional_targets)
            valid_augmentation = self.check_augmented_sample(aug_output)

        for target_name, transformed_target in aug_output.items():
            sample[target_name] = transformed_target

        return sample

    def check_augmented_sample(self, aug_output):
        return aug_output['mask'].sum() > 1.0


    def get_sample(self, index=0):
        if self.data_mode == 'multiple':
            real_path, mask_path = self.real_pairs[index]
        else:
            real_path, mask_path = self.real_path, self.mask_path

        real = cv2.imread(real_path)
        if real is None:
            raise FileNotFoundError(f'Failed to read real image: {real_path}')
        real = cv2.cvtColor(real, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path)
        if mask is None:
            raise FileNotFoundError(f'Failed to read mask image: {mask_path}')
        mask = mask[:, :, 0].astype(np.float32) / 255.
        mask = mask.astype(np.uint8)

        return {'mask': mask, 'real': real, 'img_path': real_path}

    def __len__(self):
        """Return the total number of images."""
        return len(self.real_pairs)
