import tqdm.auto as tqdm
import torch.nn.functional as F
from typing import Optional, Dict, Sequence
from typing import List, Optional, Tuple, Union
import transformers
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import numpy as np
from dataclasses import dataclass, field
from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from torchvision import transforms
from PIL import Image   
from torch.nn.parallel import DistributedDataParallel as DDP
import util.misc as misc
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('llama_adapterV2 pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--save_every_epochs', default=5, type=int,
                        help='save model every N epochs')


    # Model parameters
    parser.add_argument('--llama_type', default='7B', type=str,
                        help='Type of LLaMA model') #
    parser.add_argument('--llama_path', default='/path/to/llama', type=str,
                        help='path to LLaMA pretrained checkpoint')
    parser.add_argument('--max_words', default=96, type=int,
                        help='max number of input words')
    parser.add_argument('--clip_model', default='ViT-L/14', type=str,
                        help='CLIP model name')
    parser.add_argument('-train_clip_visual', action='store_true', help='train clip visual')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_config', default='configs/data/pretrain/EN.yaml', type=str,
                        help='dataset config path')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)


    parser.add_argument('--output_dir', default='./output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--split_epoch', type=int, default=50)

    return parser

def get_tokenizer(tokenizer_path, max_img_size = 100, image_num = 32):
    '''
    Initialize the image special tokens
    max_img_size denotes the max image put length and image_num denotes how many patch embeddings the image will be encoded to 
    '''
    if isinstance(tokenizer_path,str):
        image_padding_tokens = []
        text_tokenizer = LlamaTokenizer.from_pretrained(
            tokenizer_path,
        )
        special_token = {"additional_special_tokens": ["<image>","</image>"]}
        for i in range(max_img_size):
            image_padding_token = ""
            
            for j in range(image_num):
                image_token = "<image"+str(i*image_num+j)+">"
                image_padding_token = image_padding_token + image_token
                special_token["additional_special_tokens"].append("<image"+str(i*image_num+j)+">")
            image_padding_tokens.append(image_padding_token)
            text_tokenizer.add_special_tokens(
                special_token
            )
            ## make sure the bos eos pad tokens are correct for LLaMA-like models
            text_tokenizer.pad_token_id = 0
            text_tokenizer.bos_token_id = 1
            text_tokenizer.eos_token_id = 2    
    
    return  text_tokenizer,image_padding_tokens    

def combine_and_preprocess(question,image_list,image_padding_tokens):
    
    transform = transforms.Compose([                        
                transforms.RandomResizedCrop([512,512],scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ])
    images  = []
    new_qestions = [_ for _ in question]
    padding_index = 0
    for img in image_list:
        img_path = img['img_path']
        position = img['position']
        
        
        
        image = Image.open(img_path).convert('RGB')   
        image = transform(image)
        image = image.unsqueeze(0).unsqueeze(-1) # c,w,h,d
        
        ## pre-process the img first
        target_H = 512 
        target_W = 512 
        target_D = 4 
        # This can be different for 3D and 2D images. For demonstration we here set this as the default sizes for 2D images. 
        images.append(torch.nn.functional.interpolate(image, size = (target_H,target_W,target_D)))
        
        ## add img placeholder to text
        new_qestions[position] = "<image>"+ image_padding_tokens[padding_index] +"</image>" + new_qestions[position]
        padding_index +=1
    
    vision_x = torch.cat(images,dim = 1).unsqueeze(0) #cat tensors and expand the batch_size dim
    text = ''.join(new_qestions) 
    return text, vision_x, 
    
    
def main(args):
    
    # Paralellism init from openclip
    misc.init_distributed_mode(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # set up parallel shit
    #n_gpus = torch.cuda.device_count()
    #world_size = n_gpus
    #rank = dist.get_rank()
    #dist.init_process_group("nccl")
    #print(f"Start running DDP on rank {rank}.")
    #device_id = rank % world_size

    print("Setup tokenizer")
    text_tokenizer,image_padding_tokens = get_tokenizer('./Quick_demo/Language_files')
    print("Finish loading tokenizer")
    
    ### Initialize a simple case for demo ###
    print("Setup demo case")
    question = "Can you identify any visible signs of Cardiomegaly in the image?"
    image =[
            {
                'img_path': './Quick_demo/view1_frontal.jpg',
                'position': 0, #indicate where to put the images in the text string, range from [0,len(question)-1]
            }, # can add abitrary number of imgs
        ] 
        
    text,vision_x = combine_and_preprocess(question,image,image_padding_tokens)    
        
    print("Finish loading demo case")
    
    print("Setup Model")
    model = MultiLLaMAForCausalLM(
        lang_model_path='./Quick_demo/Language_files', ### Build up model based on LLaMa-13B config
    )
    model = DDP(model, device_ids=[args.gpu], find_unused_parameters=True)
    #fsdp_model = FullyShardedDataParallel(
    """
    ckpt = torch.load('./Quick_demo/pytorch_model.bin', map_location ='cpu') # Please dowloud our checkpoint from huggingface and Decompress the original zip file first
    model.load_state_dict(ckpt)
    print("Finish loading model")
    
    #model = model.to('cuda')
    
    model.eval() 
    with torch.no_grad():
        lang_x = text_tokenizer(
                text, max_length=2048, truncation=True, return_tensors="pt", legacy=False
        )['input_ids'].to('cuda')
        
        vision_x = vision_x.to('cuda')
        generation = model.generate(lang_x,vision_x)
        generated_texts = text_tokenizer.batch_decode(generation, skip_special_tokens=True) 
        print('---------------------------------------------------')
        print('Input: ', question)
        print('Output: ', generated_texts[0])
    """
    
if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
       
