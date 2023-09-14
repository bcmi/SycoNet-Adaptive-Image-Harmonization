import cv2
import random
import numpy as np
import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from copy import deepcopy

class iHarmony4Dataset(BaseDataset):
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
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        self.image_paths = []
        self.isTrain = opt.isTrain
        if opt.isTrain==False:
            print('loading test file: ')
            self.keep_background_prob = -1
            if opt.dataset_name in ['HAdobe5k', 'HCOCO', 'HFlickr', 'Hday2night']:
                self.trainfile = os.path.join(opt.dataset_root, opt.dataset_name, opt.dataset_name + '_test.txt')
                with open(self.trainfile,'r') as f:
                        for line in f.readlines():
                            self.image_paths.append(os.path.join(opt.dataset_root, opt.dataset_name, 'composite_images', line.rstrip()))
            else:
                raise NotImplementedError('%s not implemented.' % (opt.dataset_name))
        else:
            raise NotImplementedError('Sorry, the training code has not been released.')

        self.transform = get_transform(opt)
        self.input_transform = transforms.Compose([
                transforms.ToTensor(),
            ])

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
            valid_augmentation = self.check_augmented_sample(sample, aug_output)

        for target_name, transformed_target in aug_output.items():
            sample[target_name] = transformed_target

        return sample

    def check_augmented_sample(self, sample, aug_output):
        if self.keep_background_prob < 0.0 or random.random() < self.keep_background_prob:
            return True

        return aug_output['mask'].sum() > 1.0

    def get_sample(self, index):
        path = self.image_paths[index]
        mask_path = self.image_paths[index].replace('composite_images','masks')
        mask_path = '_'.join(mask_path.split('_')[:-1]) + '.png'
        real_path = self.image_paths[index].replace('composite_images','real_images')
        real_path = '_'.join(real_path.split('_')[:-2]) + '.jpg'

        real = cv2.imread(real_path)
        real = cv2.cvtColor(real, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path)
        mask = mask[:, :, 0].astype(np.float32) / 255.
        mask = mask.astype(np.uint8)

        return {'mask': mask, 'real': real, 'img_path': path}

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)
