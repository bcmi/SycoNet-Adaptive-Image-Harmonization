import os
from util import util
from options.test_options import TestOptions
from data import create_dataset
from models import create_model

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    if opt.data_mode == 'multiple':
        opt.batch_size = max(1, int(opt.batch_size))
    else:
        opt.batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.data_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    div_num = opt.augment_num
    
    print('total number of test images: %d' % len(dataset))
 
    save_dir = os.path.join(opt.results_dir)
    os.makedirs(save_dir, exist_ok=True)
    if opt.eval:
        model.eval()

    for i, data in enumerate(dataset):
        batch_size = data['real'].shape[0]
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break

        for j in range(div_num):
            model.set_input(data)  # unpack data from data loader
            model.test()           # run inference
            visuals = model.get_current_visuals()  # get image results
            transfer_img = visuals.get('transfer_img_c', None)

            if transfer_img is None:
                raise ValueError("'transfer_img_c' not found in visuals.")

            for b in range(batch_size):
                img_path = str(data['img_path'][b])
                _, image_name = os.path.split(img_path)
                image_name, _ = os.path.splitext(image_name)

                batch_out = transfer_img[b]
                save_path = os.path.join(save_dir, image_name + '_' + str(j) + '.jpg')
                output_c = util.tensor2im(batch_out)
                util.save_image(output_c, save_path, aspect_ratio=opt.aspect_ratio)

                print(f'[{i * opt.batch_size + b}], {image_name}, z num: {j}')

