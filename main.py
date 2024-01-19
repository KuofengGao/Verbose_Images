import os
from pathlib import Path
import logging
import argparse
import time
import pynvml
from scipy import log
import numpy as np
from PIL import Image

import torch
from lavis.models import load_model_and_preprocess


# setup device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def measure_gpu(model, input_token, iter_num, device_id):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
    multi_cap_list = []
    t1 = time.time()
    power_list = []
    avg_length, verbose_length, verbose_caption = 0, 0, 0
    for _ in range(iter_num):
        caption = model.generate(input_token, use_nucleus_sampling=True, top_p=0.9, temperature=1, num_beams=1, max_length=512, repetition_penalty=1)[0]
        length = len(caption.split(' '))
        multi_cap_list.append(caption)
        avg_length += (length / iter_num)
        if length > verbose_length:
            verbose_length = length
            verbose_caption = caption
        power = pynvml.nvmlDeviceGetPowerUsage(handle)
        power_list.append(power)
    t2 = time.time()
    latency = (t2 - t1) / iter_num
    s_energy = sum(power_list) / len(power_list) * latency
    energy = s_energy / (10 ** 3) / iter_num               
    pynvml.nvmlShutdown()
    return latency, energy, verbose_caption, avg_length, multi_cap_list


class TestModel:
    def clamp(self, delta, clean_imgs):
        MEAN = torch.tensor([[[0.48145466]], [[0.4578275]], [[0.40821073]]]).to(clean_imgs.device)
        STD = torch.tensor([[[0.26862954]], [[0.26130258]], [[0.27577711]]]).to(clean_imgs.device)

        clamp_imgs = (((delta.data + clean_imgs.data) * STD + MEAN) * 255).clamp(0, 255)
        clamp_delta = (clamp_imgs/255 - MEAN) / STD - clean_imgs.data

        return clamp_delta


    def test_model(self, args, logger):
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device
        )
        model = model.eval()

        ITER, STEP_SIZE, EPSILON = args.iter, args.step_size, args.epsilon
        input_text = ""

        raw_image = Image.open("imgs/COCO_val2014_000000002006.jpg").convert("RGB")
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        
        image = image.to(device)  
        delta = torch.randn_like(image, requires_grad=True)
        
        verbose_len, verbose_energy, verbose_latency = 0, 0, 0
        ori_latency, ori_energy, ori_caption, ori_len, ori_multi_cap_list = measure_gpu(model, {"image": image, "args": args, "prompt": [input_text], "logger": logger}, 3, args.gpu)
        output_text = ori_caption
        
        verbose_multi_cap_list = []
        verbose_len_list, verbose_energy_list, verbose_latency_list, ori_latency_list, ori_energy_list, ori_len_list = [],[],[],[],[],[]
        ori_latency_list.append(ori_latency)
        ori_energy_list.append(ori_energy)
        ori_len_list.append(ori_len)

        for tdx in range(ITER):
            result = model.generate_verbose_images({"image": image + delta, "text_input": [output_text], "logger": logger})

            loss1, loss2, loss3 = result["loss1"], result["loss2"], result["loss3"]
            loss1_val, loss2_val, loss3_val = loss1.detach().clone(), loss2.detach().clone(), loss3.detach().clone()

            ratio1 = 10.0 * log(tdx + 1) - 20.0
            ratio2 = 0.5 * log(tdx + 1) + 1.0

            if tdx == 0:
                lambda1 = torch.abs(loss1_val / loss2_val / ratio1)
                lambda2 = torch.abs(loss1_val / loss3_val / ratio2)
            else:
                cur_lambda1 = torch.abs(loss1_val / loss2_val / ratio1)
                cur_lambda2 = torch.abs(loss1_val / loss3_val / ratio2)                     
                lambda1 = 0.9 * last_lambda1 + 0.1 * cur_lambda1
                lambda2 = 0.9 * last_lambda2 + 0.1 * cur_lambda2
            
            last_lambda1, last_lambda2 = lambda1, lambda2  
            
            loss = loss1 + lambda1 * loss2 + lambda2 * loss3

            model.zero_grad()
            loss.backward(retain_graph=False)
            delta.data = delta - STEP_SIZE * torch.sign(delta.grad.detach())
            delta.data = self.clamp(delta, image).clamp(-EPSILON, EPSILON)
            delta.grad.zero_()
                        
            output_latency, output_energy, output_text, output_len, output_multi_cap_list = measure_gpu(model, {"image": image + delta, "args": args, "prompt": [input_text], "logger": logger}, 3, args.gpu)

            if output_len > verbose_len:
                verbose_energy = output_energy
                verbose_latency = output_latency
                verbose_len = output_len
                verbose_multi_cap_list = output_multi_cap_list
                
        verbose_latency_list.append(verbose_latency)
        verbose_energy_list.append(verbose_energy)
        verbose_len_list.append(verbose_len)

        for len_idx in range(3):
            logger.info('Original sequences: %s', ori_multi_cap_list[len_idx])
            logger.info('Verbose sequences: %s', verbose_multi_cap_list[len_idx])
            logger.info('------------------------')

        logger.info('Original images, Length: %.2f, Energy: %.2f, Latency: %.2f', ori_len, ori_energy, ori_latency)
        logger.info('Verbose images, Length: %.2f, Energy: %.2f, Latency: %.2f', verbose_len, verbose_energy, verbose_latency)


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('generate verbose images')
    parser.add_argument('--epsilon', type=float, default=0.032, help='the perturbation magnitude')
    parser.add_argument('--step_size', type=float, default=0.0039, help='the step size')
    parser.add_argument('--iter', type=int, default=1000, help='the iteration')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--seed', type=int, default=256, help='random seed')
    return parser.parse_args()


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    def log_string(str):
        logger.info(str)
        print(str)

    '''CREATE DIR'''
    exp_dir = Path(os.path.join(args.root_path, 'log'))
    exp_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath(args.dataset)
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("OPT")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/log.txt' % log_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)
        
    test_blip2 = TestModel()
    test_blip2.test_model(args, logger)


if __name__ == '__main__':
    args = parse_args()
    main(args)

