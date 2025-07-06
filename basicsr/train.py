import os
import torch
import argparse
import logging
import random
import numpy as np
from torch.utils.data import DataLoader
from basicsr.utils.parse_options import parse_options 
from basicsr.utils import get_root_logger, check_resume, make_exp_dirs, mkdir_and_rename, set_random_seed
from basicsr.models import build_model
from basicsr.data import create_dataloader, create_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    args = parser.parse_args()

    opt = parse_options(args.opt, is_train=True)
    opt['dist'] = False

    opt['datasets']['train']['phase'] = 'train'
    opt['datasets']['val']['phase'] = 'val'

    # create necessary folders
    make_exp_dirs(opt)

    # set random seed
    seed = opt.get('manual_seed')
    if seed is not None:
        set_random_seed(seed)

    # get logger
    log_file = os.path.join(opt['path']['log'], 'train.log')
    logger = get_root_logger(log_level=logging.INFO, log_file=log_file)
    logger.info('Options:\n' + str(opt))

    # create train and val dataloaders
    train_set = create_dataset(opt['datasets']['train'])
    train_loader = create_dataloader(
        train_set,
        opt['datasets']['train'],
        num_gpu=opt['num_gpu']
    )

    if opt['val'].get('val_freq'):
        val_set = create_dataset(opt['datasets']['val'])
        val_loader = create_dataloader(
            val_set,
            opt['datasets']['val'],
            num_gpu=opt['num_gpu']
        )


    # create model
    model = build_model(opt)

    # training loop
    current_iter = 0
    start_epoch = 0
    total_iter = int(opt['train']['total_iter'])

    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    model.print_network(model.net_g)

    for epoch in range(start_epoch, 9999999):
        for train_data in train_loader:
            current_iter += 1
            if current_iter > total_iter:
                break

            model.feed_train_data(train_data)
            model.optimize_parameters(current_iter)

            if current_iter % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                log_str = f'Train Iter: {current_iter}'
                for k, v in logs.items():
                    log_str += f' | {k}: {v:.4f}'
                logger.info(log_str)

            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            if opt['val'].get('val_freq') and current_iter % int(opt['val']['val_freq']) == 0:
                logger.info('Running validation.')
                model.validation(val_loader, current_iter, tb_logger=None)

        if current_iter > total_iter:
            logger.info('End of training.')
            break
            
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
