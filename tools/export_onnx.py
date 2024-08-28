import _init_path
import argparse
import datetime
import glob
import os
from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_sched
from tensorboardX import SummaryWriter

from mtr.datasets import build_dataloader
from mtr.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from mtr.utils import common_utils
from mtr.models import model as model_utils

from train_utils.train_utils import train_model


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default="tools/cfgs/waymo/mtr+100_percent_data_test_onnx.yaml", help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=2, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=3, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=1, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='onnx5', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--without_sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=2, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=None, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=5, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--not_eval_with_train', action='store_true', default=False, help='')
    parser.add_argument('--logger_iter_interval', type=int, default=50, help='')
    parser.add_argument('--ckpt_save_time_interval', type=int, default=300, help='in terms of seconds')

    parser.add_argument('--add_worker_init_fn', action='store_true', default=False, help='')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

def main(onnx_name):
    args, cfg = parse_config()
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    # output_dir.mkdir(parents=True, exist_ok=True)
    # ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt = ckpt_dir / 'best_model.pth'

    model = model_utils.MotionTransformer(config=cfg.MODEL)

    checkpoint = torch.load(ckpt, map_location='cuda')
    model.load_state_dict(checkpoint['model_state'], strict=True)
    model.cuda()
    
    # def forward(self, track_index_to_predict,
    #             obj_trajs, obj_trajs_mask, 
    #             map_polylines, map_polylines_mask, 
    #             obj_trajs_last_pos, map_polylines_center,
    #             center_objects_type):    
    
    track_index_to_predict = torch.arange(8).cuda()
    obj_trajs = torch.randn(8,32,11,29,dtype=torch.float32).cuda()
    obj_trajs_masks = torch.ones(8,32,11, dtype=torch.bool).cuda()

    map_polylines = torch.randn(8,512,20,9,dtype = torch.float32).cuda()
    map_polylines_mask = torch.ones(8,512,20,dtype = torch.bool).cuda()
    
    obj_trajs_last_pos = torch.randn(8,32,3, dtype=torch.float32).cuda()
    map_polylines_center = torch.randn(8,512,3,dtype = torch.float32).cuda()
    center_objects_type = torch.ones(8, dtype=torch.int64).cuda()

    # 定义输入tensor
    # x = torch.randn(1, 3, args.input_size[0], args.input_size[1])
    input_names = ["track_index_to_predict", "obj_trajs", "obj_trajs_masks",
              "map_polylines", "map_polylines_mask", "obj_trajs_last_pos", "map_polylines_center", "center_objects_type"]
    inputs = (track_index_to_predict, obj_trajs, obj_trajs_masks,
              map_polylines, map_polylines_mask, obj_trajs_last_pos, map_polylines_center, center_objects_type)
    out_names = ["pred_scores","pred_trajs"]
    # print(msg)
    # 模型调整为eval模式
    model.eval()
    # 开始转onnx
    
    trace_backbone = torch.jit.trace(model, inputs, check_trace=False)
    # torch.onnx.export(model, batch, 'test.onnx', export_params=True, training=False, input_names=input_names, output_names=out_names)
    torch.onnx.export(trace_backbone, inputs, 
                      f'/home/nh/code/auto/MTR/{onnx_name}', 
                      export_params=True, 
                      input_names=input_names, output_names=out_names)
    print('please run: python -m onnxsim test.onnx test_sim.onnx\n')

if __name__ == '__main__':
    # args = get_args_parser()
    # args = args.parse_args()
    main("test5.onnx")