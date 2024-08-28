# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved

import _init_path
import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path

import numpy as np

from eval_utils import eval_utils
from mtr.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from mtr.datasets import build_dataloader
from mtr.models import model as model_utils
from mtr.utils import common_utils

import matplotlib.pyplot as plt

# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorrt
import tensorrt as trt
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple cuda-python
from cuda import cudart


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    # parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    # parser.add_argument('--cfg_file', type=str, default="tools/cfgs/waymo/mtr+100_percent_data.yaml", help='specify the config for training')
    parser.add_argument('--cfg_file', type=str, default="tools/cfgs/waymo/mtr+100_percent_data_test_onnx.yaml", help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    # parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--workers', type=int, default=1, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='onnx5', help='extra tag for this experiment')
    # parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--ckpt', type=str, default="output/cfgs/waymo/mtr+100_percent_data_test_onnx/onnx5/ckpt/best_model.pth", help='checkpoint to start from')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def eval_engine_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=False):
    # load checkpoint
    if args.ckpt is not None: 
        it, epoch = model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
    else:
        it, epoch = -1, -1
    model.cuda()

    logger.info(f'*************** LOAD MODEL (epoch={epoch}, iter={it}) for EVALUATION *****************')
    # start evaluation
    eval_utils.eval_one_epoch(
        cfg, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir, save_to_file=args.save_to_file
    )
    
class OutputAllocator(trt.IOutputAllocator):
    def __init__(self):
        trt.IOutputAllocator.__init__(self)
        self.buffers = {}
        self.shapes = {}

    def reallocate_output(self, tensor_name, memory, size, alignment):
        ptr = cudart.cudaMalloc(size)[1]
        self.buffers[tensor_name] = ptr
        return ptr

    def notify_shape(self, tensor_name, shape):
        self.shapes[tensor_name] = tuple(shape)
    
onnx_path = "/home/nh/code/auto/MTR/test5.onnx"
engine_path = "/home/nh/code/auto/MTR/test5.engine"

def onnx_engine():
    logger  = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser  = trt.OnnxParser(network, logger)
    config  = builder.create_builder_config()

    with open(onnx_path, "rb") as f:
        parser.parse(f.read())

    engineString = builder.build_serialized_network(network, config)

    with open(engine_path, "wb") as f:
        f.write(engineString)


def load_engine():
    with open(engine_path, "rb") as f:
        engineString = f.read()

    logger  = trt.Logger(trt.Logger.VERBOSE)
    runtime = trt.Runtime(logger)
    engine  = runtime.deserialize_cuda_engine(engineString)
    return engine

def infer_engine(engine, input):
    # with open(engine_path, "rb") as f:
    #     engineString = f.read()

    # logger  = trt.Logger(trt.Logger.VERBOSE)
    # runtime = trt.Runtime(logger)
    # engine  = runtime.deserialize_cuda_engine(engineString)
    context = engine.create_execution_context()


    ## input 是一个字典，包含模型的输入张量名称和对应的数据
    # input = {"track_index_to_predict": np.arange(8),
    #         "obj_trajs": np.random.randn(8,32,11,29),
    #         "obj_trajs_masks": np.ones([8,32,11]),
    #         "map_polylines": np.random.randn(8,512,20,9),
    #         "map_polylines_mask": np.ones([8,512,20]),
                
    #         "obj_trajs_last_pos": np.random.randn(8,32,3),
    #         "map_polylines_center": np.random.randn(8,512,3),
    #         "center_objects_type": np.ones(8)}


    ## 用于保存 cuda 内存指针的容器，用于后续释放内存
    input_buffers = {}
    ## num_io_tensors 表示模型输入输出张量的数量
    for i in range(engine.num_io_tensors):
        ## get_tensor_name 获取张量名称，就是在导出 ONNX 模型中定义的名称
        name = engine.get_tensor_name(i)
        ## get_tensor_mode 获取张量类别，包括输入和输出两种，这里只处理输入张量
        if engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
            continue

        ## 获取输入张量数据
        array = input[name]
        ## 模型要求的输入张量数据类型，这里我们把转换成 numpy 类型
        dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
        ## 保持输入张量类型与模型要求的数据类型一致
        array = array.astype(dtype)
        ## numpy 数组的内存布局有可能不是连续的，这里需要转换为连续的内存布局，以便使用指针拷贝
        array = np.ascontiguousarray(array)

        ## cudaMalloc 分配 GPU 内存，返回内存指针和错误码
        err, ptr = cudart.cudaMalloc(array.nbytes)
        if err > 0:
            raise Exception("cudaMalloc failed, error code: {}".format(err))

        ## 暂时保存内存指针，后续还需要释放    
        input_buffers[name] = ptr
        ## cudaMemcpy 将数据从 CPU 拷贝到 GPU，其中 array.ctypes.data 是 numpy 数组的内存指针
        cudart.cudaMemcpy(ptr, array.ctypes.data, array.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        ## set_input_shape 设置输入张量的实际形状，对于 dynamic shape 这一步是必要的，因为动态维度在 ONNX 转换过程中被设置成了 -1，这里不设置将会报错
        context.set_input_shape(name, array.shape)
        ## set_tensor_address 设置输入张量的内存地址
        context.set_tensor_address(name, ptr)
    
    output_allocator = OutputAllocator()
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
            continue

        context.set_output_allocator(name, output_allocator)
        
    context.execute_async_v3(0)
    
    output = {}
    for name in output_allocator.buffers.keys():
        ptr = output_allocator.buffers[name]
        shape = output_allocator.shapes[name]
        dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
        nbytes = np.prod(shape) * dtype.itemsize

        output_buffer = np.empty(shape, dtype = dtype)
        cudart.cudaMemcpy(output_buffer.ctypes.data, ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        output[name] = output_buffer

    for name in input_buffers.keys():
        ptr = input_buffers[name]
        cudart.cudaFree(ptr)

    for name in output_allocator.buffers.keys():
        ptr = output_allocator.buffers[name]
        cudart.cudaFree(ptr)
        
    return output

# def visualize_new(input_dict, pred_scores, pred_trajs):
def visualize_new(scenario_id, center_objects_id, 
                  pred_scores, pred_trajs, 
                  obj_trajs, obj_trajs_mask,
                  map_polylines, map_polylines_mask,
                  track_index_to_predict):
    root_dir = Path("/home/lotuspeak/code/auto/MTR/visualize/del_cuda_engine")

    root_dir.mkdir(parents=True, exist_ok=True)
    # pred_scores = pred_scores.cpu().numpy()
    # pred_trajs = pred_trajs.cpu().numpy()

    # scenario_id = input_dict['scenario_id']
    # obj_trajs = input_dict['obj_trajs'].cpu().numpy()
    # obj_trajs_mask = input_dict['obj_trajs_mask'].cpu().numpy()
    # obj_trajs_future_state = input_dict['obj_trajs_future_state'].cpu().numpy()
    # obj_trajs_future_mask = input_dict['obj_trajs_future_mask'].cpu().numpy()
    # obj_ids = input_dict['obj_ids']
    # map_polylines = input_dict['map_polylines'].cpu().numpy()
    # map_polylines_mask = input_dict['map_polylines_mask'].cpu().numpy()
    # track_index_to_predict = input_dict['track_index_to_predict'].cpu().numpy()

    batch_size = len(scenario_id)
    for idx in range(batch_size):
        file_name = root_dir / f"{scenario_id[idx]}_{center_objects_id[idx]}.png"
        cur_map_polylines = map_polylines[idx]
        cur_map_polylines_mask = map_polylines_mask[idx]
        cur_obj_trajs = obj_trajs[idx]
        # cur_obj_trajs_future = obj_trajs_future_state[idx]
        # cur_obj_trajs_future_mask = obj_trajs_future_mask[idx]
        cur_obj_trajs_mask = obj_trajs_mask[idx]
        center_obj_track_index = track_index_to_predict[idx]
        plt.figure(figsize=(20,15))
        # plt.scatter([-30,30],[-20,20], marker='+')
        plt.xlim(-30, 120)
        plt.ylim(-60, 60)
        for pl_idx, pl in enumerate(cur_map_polylines):
            if cur_map_polylines_mask[pl_idx].sum() <= 0:
                continue
            pl_mask = cur_map_polylines_mask[pl_idx]
            xs = pl[pl_mask][:,0]
            ys = pl[pl_mask][:,1]
            plt.plot(xs,ys, marker='.', color = 'gray')

        for traj_idx, traj in enumerate(cur_obj_trajs):
            if traj_idx == center_obj_track_index:
                continue
            traj_mask = cur_obj_trajs_mask[traj_idx]
            xs = traj[traj_mask][:,0]
            ys = traj[traj_mask][:,1]
            plt.plot(xs,ys, marker='.', color = 'yellow', linewidth=1.5)

        ## history tgt green
        center_traj = cur_obj_trajs[center_obj_track_index]
        center_traj_mask = cur_obj_trajs_mask[center_obj_track_index]
        xs = center_traj[center_traj_mask][:,0]
        ys = center_traj[center_traj_mask][:,1]
        plt.plot(xs,ys, marker='.', color = 'green', linewidth = 1.5) 

        ## blue gt
        # center_traj_future = cur_obj_trajs_future[center_obj_track_index]
        # center_traj_future_mask = cur_obj_trajs_future_mask[center_obj_track_index].astype(bool)
        # xs = center_traj_future[center_traj_future_mask][:,0]
        # ys = center_traj_future[center_traj_future_mask][:,1]
        # plt.plot(xs,ys, marker='.', color = 'blue') 

        ## red predictobj_trajs_n
        cur_pred_trajs = pred_trajs[idx]
        for pred_traj in cur_pred_trajs[:]:
            plt.plot(pred_traj[:,0],pred_traj[:,1], marker='.', color = 'red')
            # break

        plt.savefig(file_name)
        plt.close()


def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus
          
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval'

    if not args.eval_all:
        num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
        epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
        eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id)
    else:
        epoch_id = None
        eval_output_dir = eval_output_dir / 'eval_all_default'

    if args.eval_tag is not None:
        eval_output_dir = eval_output_dir / args.eval_tag

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    if args.fix_random_seed:
        common_utils.set_random_seed(666)

    # ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else output_dir / 'ckpt'

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        # batch_size=args.batch_size,
        batch_size=1,
        dist=dist_test, 
        # workers=args.workers,
        workers=1, 
        logger=logger, training=False
    )
    
    engine = load_engine()
    
    for i, batch_dict in enumerate(test_loader):
        input_dict = batch_dict['input_dict']
        if i > 10:
            break
        # if input_dict['scenario_id'][0] == '214dfc443104674a':s
        track_index_to_predict = input_dict['track_index_to_predict']
        obj_trajs = input_dict['obj_trajs']
        obj_trajs_mask = input_dict['obj_trajs_mask']
        map_polylines = input_dict['map_polylines']
        map_polylines_mask = input_dict['map_polylines_mask']
        obj_trajs_last_pos = input_dict['obj_trajs_last_pos']
        map_polylines_center = input_dict['map_polylines_center']
        center_objects_type = input_dict['center_objects_type']
        
        scenario_id = input_dict['scenario_id']
        center_objects_id = input_dict['center_objects_id']

        print(track_index_to_predict.shape)
        print(obj_trajs.shape)
        print(obj_trajs_mask.shape)
        print(map_polylines.shape)
        print(map_polylines_mask.shape)
        print(obj_trajs_last_pos.shape)
        print(map_polylines_center.shape)
        print(center_objects_type.shape)
        
        sample_mask = track_index_to_predict<30
        if sample_mask.shape[0] == 1:
            continue
        
        track_index_to_predict = track_index_to_predict[sample_mask].numpy()
        obj_trajs = obj_trajs[sample_mask].numpy()
        obj_trajs_mask = obj_trajs_mask[sample_mask].numpy()
        map_polylines = map_polylines[sample_mask].numpy()
        map_polylines_mask = map_polylines_mask[sample_mask].numpy()
        obj_trajs_last_pos = obj_trajs_last_pos[sample_mask].numpy()
        map_polylines_center = map_polylines_center[sample_mask].numpy()
        center_objects_type = center_objects_type[sample_mask].numpy()
        
        scenario_id = scenario_id[sample_mask]
        center_objects_id = center_objects_id[sample_mask]
        
        track_index_to_predict_n = np.zeros(8, dtype=np.int64)
        obj_trajs_n = np.zeros([8,32,11,29], dtype=np.float32)
        obj_trajs_mask_n = np.zeros([8,32,11],dtype=np.bool_)
        map_polylines_n = np.zeros([8,512,20,9], dtype=np.float32)
        map_polylines_mask_n = np.zeros([8,512,20], dtype=np.bool_)
        obj_trajs_last_pos_n = np.zeros([8,32,3],dtype=np.float32)
        map_polylines_center_n = np.zeros([8,512,3],dtype=np.float32)
        center_objects_type_n = np.zeros(8, dtype=np.int64)
        
        sample_len = obj_trajs.shape[0]
        agents_len = obj_trajs.shape[1]
        map_len = map_polylines.shape[1]

        track_index_to_predict_n[:min(8,sample_len)] = track_index_to_predict[:min(8,sample_len)]
        obj_trajs_n[:min(8,sample_len), :min(32,agents_len)] = obj_trajs[:min(8,sample_len), :min(32,agents_len)]
        obj_trajs_mask_n[:min(8,sample_len), :min(32,agents_len)] = obj_trajs_mask[:min(8,sample_len), :min(32,agents_len)]
        map_polylines_n[:min(8,sample_len), :min(512,map_len)] = map_polylines[:min(8,sample_len), :min(512,map_len)]
        map_polylines_mask_n[:min(8,sample_len), :min(512,map_len)] = map_polylines_mask[:min(8,sample_len), :min(512,map_len)]
        
        obj_trajs_last_pos_n[:min(8,sample_len), :min(32,agents_len)] = obj_trajs_last_pos_n[:min(8,sample_len), :min(32,agents_len)]
        map_polylines_center_n[:min(8,sample_len), :min(512,map_len)] = map_polylines_center[:min(8,sample_len), :min(512,map_len)]
        center_objects_type_n[:min(8,sample_len)] = center_objects_type[:min(8,sample_len)]
        
        scenario_id_n = scenario_id[:min(8,sample_len)]
        center_objects_id_n = center_objects_id[:min(8,sample_len)]
        
        input = {"track_index_to_predict": track_index_to_predict_n,
                "obj_trajs": obj_trajs_n,
                "obj_trajs_masks": obj_trajs_mask_n,
                "map_polylines": map_polylines_n,
                "map_polylines_mask": map_polylines_mask_n,
                    
                "obj_trajs_last_pos": obj_trajs_last_pos_n,
                "map_polylines_center": map_polylines_center_n,
                "center_objects_type": center_objects_type_n}
        
        output = infer_engine(engine, input)
        
        pred_scores = output['pred_scores']
        pred_trajs = output['pred_trajs']
        
        visualize_new(scenario_id_n, center_objects_id, pred_scores, pred_trajs, 
                        obj_trajs_n, obj_trajs_mask_n, map_polylines_n, map_polylines_mask_n, track_index_to_predict_n)
                
                
                
            # pred_scores, pred_trajs, pred_dense_future_trajs, intention_points = \
            # pred_scores, pred_trajs = \
            #     model(input_dict['track_index_to_predict'],
            #         input_dict['obj_trajs'], input_dict['obj_trajs_mask'],
            #         input_dict['map_polylines'], input_dict['map_polylines_mask'],
            #         input_dict['obj_trajs_last_pos'], input_dict['map_polylines_center'],
            #         input_dict['center_objects_type'])

            # visualize(batch_pred_dicts)

    
    # model = model_utils.MotionTransformer(config=cfg.MODEL)
    # with torch.no_grad():
    #     if args.eval_all:
    #         repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=dist_test)
    #     else:
    #         eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=dist_test)


if __name__ == '__main__':
    # onnx_engine()
    main()

