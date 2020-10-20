"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import torchvision.transforms as transforms

import coremltools as ct
import torch
import cv2 as cv
import numpy as np
from PIL import Image


def original_example():
    print('Running the original example of result creation')
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name,
                           '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()  # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    webpage.save()  # save the HTML


def load_model_with_options():
    """
    Load the model details
    """
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    # dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    return opt, model

def create_normalized_tensor(image_name):
    """
    Create normalized tensor given image path
    :param image_name: image path
    :return: image tensor
    """
    input_img = Image.open(image_name)
    transforms_train = [transforms.Resize(256, 256), transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # Normalize the tensor
    input_tensor = input_img
    for t in transforms_train:
        input_tensor = t(input_tensor)
    input_tensor = torch.unsqueeze(input_tensor, 0)
    return input_tensor


def write_clmodel_res(out_name, res_img):
    """
    Save the result of model prediction
    :param out_name:
    :param res: result of the prediction
    :return:
    """
    img = np.zeros((res_img.shape[2], res_img.shape[3], 3))
    for k in range(3):
        img[:, :, k] = 255 * res_img[0, 2 - k, :, :]
    cv.imwrite(out_name, img)


def model_conversion():
    print('Running model conversion')
    opt, model = load_model_with_options()
    if opt.eval:
        model.eval()

    # Read image and create input tensor
    input_tensor = create_normalized_tensor(opt.input_img)

    # Model conversion
    model.netG.eval()
    # # Step 1 - create a traced model
    traced_model = torch.jit.trace(model.netG, input_tensor)

    if opt.core_input == 'image':
        ssmodel = ct.convert(
            traced_model,
            inputs=[ct.ImageType(name="input1", shape=input_tensor.shape, bias=[-1, -1, -1], scale=1 / 127.0)]
        )
        ssmodel.save(opt.model_path)
        # Test model
        input_img = Image.open(opt.input_img)
        res = ssmodel.predict({"input1": input_img})
    elif opt.core_input == 'tensor':
        ssmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="input1", shape=input_tensor.shape)]
        )
        ssmodel.save(opt.model_path)
        # Test model
        res = ssmodel.predict({"input1": input_tensor.numpy()})

    if opt.res_img != '':
        write_clmodel_res(opt.res_img, res['226'])


if __name__ == '__main__':
    print('Before running the script, make sure:')
    print('that you put the pretrained model in facades_pix2pix')
    print('http://cmp.felk.cvut.cz/~tylecr1/facade/')
    print('pip install -r requirements.txt.')
    print('--dataroot ./datasets/facades --gpu_ids -1 --direction BtoA --model pix2pix --name '
          'facades_pix2pix --res_img path_to_img.jpg --model_path path_to_model.mlmodel')
    # original_example()
    model_conversion()
