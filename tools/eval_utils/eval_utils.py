# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved

import pickle
import time

import numpy as np
import torch
import tqdm

from mtr.utils import common_utils

from pathlib import Path
import matplotlib.pyplot as plt

def visualize(batch_pred_dicts):
    root_dir = Path("/home/nihua/code/auto/MTR/visualize/")
    
    root_dir.mkdir(parents=True, exist_ok=True)
    pred_scores = batch_pred_dicts['pred_scores'].cpu().numpy()
    pred_trajs = batch_pred_dicts['pred_trajs'].cpu().numpy()
    
    scenario_id = batch_pred_dicts['input_dict']['scenario_id']
    obj_trajs = batch_pred_dicts['input_dict']['obj_trajs'].cpu().numpy()
    obj_trajs_mask = batch_pred_dicts['input_dict']['obj_trajs_mask'].cpu().numpy()
    obj_ids = batch_pred_dicts['input_dict']['obj_ids']
    map_polylines = batch_pred_dicts['input_dict']['map_polylines'].cpu().numpy()
    map_polylines_mask = batch_pred_dicts['input_dict']['map_polylines_mask'].cpu().numpy()
    track_index_to_predict = batch_pred_dicts['input_dict']['track_index_to_predict'].cpu().numpy()
    
    batch_size = len(pred_scores)
    for idx in range(batch_size):
        file_name = root_dir / f"{scenario_id[idx]}_{obj_ids[idx]}.png"
        cur_map_polylines = map_polylines[idx]
        cur_map_polylines_mask = map_polylines_mask[idx]
        cur_obj_trajs = obj_trajs[idx]
        cur_obj_trajs_mask = obj_trajs_mask[idx]
        center_obj_track_index = track_index_to_predict[idx]
        plt.figure(figsize=(15,10))
        plt.scatter([-10,30],[-20,20], marker='+')
        plt.xlim(-10, 90)
        plt.ylim(-30, 30)
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
            plt.plot(xs,ys, marker='.', color = 'yellow')

        center_traj = cur_obj_trajs[center_obj_track_index]
        center_traj_mask = cur_obj_trajs_mask[center_obj_track_index]
        xs = center_traj[center_traj_mask][:,0]
        ys = center_traj[center_traj_mask][:,1]
        plt.plot(xs,ys, marker='.', color = 'blue') 
        
        cur_pred_trajs = pred_trajs[idx]
        for pred_traj in cur_pred_trajs:
            plt.plot(pred_traj[:,0],pred_traj[:,1], marker='.', color = 'red')
            break
        
        plt.savefig(file_name)
        plt.close()
    
    

def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None, logger_iter_interval=50):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    dataset = dataloader.dataset

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
            num_gpus = torch.cuda.device_count()
            local_rank = cfg.LOCAL_RANK % num_gpus
            model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[local_rank],
                    broadcast_buffers=False
            )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()

    pred_dicts = []
    for i, batch_dict in enumerate(dataloader):
        with torch.no_grad():
            batch_pred_dicts = model(batch_dict)
            final_pred_dicts = dataset.generate_prediction_dicts(batch_pred_dicts, output_path=final_output_dir if save_to_file else None)
            pred_dicts += final_pred_dicts
            # visualize(batch_pred_dicts)

        disp_dict = {}

        if cfg.LOCAL_RANK == 0 and (i % logger_iter_interval == 0 or i == 0 or i + 1== len(dataloader)):
            past_time = progress_bar.format_dict['elapsed']
            second_each_iter = past_time / max(i, 1.0)
            remaining_time = second_each_iter * (len(dataloader) - i)
            disp_str = ', '.join([f'{key}={val:.3f}' for key, val in disp_dict.items() if key != 'lr'])
            batch_size = batch_dict.get('batch_size', None)
            logger.info(f'eval: epoch={epoch_id}, batch_iter={i}/{len(dataloader)}, batch_size={batch_size}, iter_cost={second_each_iter:.2f}s, '
                        f'time_cost: {progress_bar.format_interval(past_time)}/{progress_bar.format_interval(remaining_time)}, '
                        f'{disp_str}')

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        logger.info(f'Total number of samples before merging from multiple GPUs: {len(pred_dicts)}')
        pred_dicts = common_utils.merge_results_dist(pred_dicts, len(dataset), tmpdir=result_dir / 'tmpdir')
        logger.info(f'Total number of samples after merging from multiple GPUs (removing duplicate): {len(pred_dicts)}')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(pred_dicts, f)

    result_str, result_dict = dataset.evaluation(
        pred_dicts,
        output_path=final_output_dir, 
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')

    return ret_dict


if __name__ == '__main__':
    pass
