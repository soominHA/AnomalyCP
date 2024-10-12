import AnomalyCLIP_lib
import torch
import argparse
import torch.nn.functional as F
from prompt_ensemble import AnomalyCLIP_PromptLearner
from loss import FocalLoss, BinaryDiceLoss
from utils import normalize
from dataset import Dataset
from logger import get_logger
from tqdm import tqdm

import os
import random
import numpy as np
from tabulate import tabulate
from utils import get_transform

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from visualization import visualizer

from metrics import image_level_metrics, pixel_level_metrics
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from utils import *

from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora
from loralib import layers as lora_layers

def evaluate_lora(args, model, loader, dataset):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'\n ===== {device} ===== \n')
    model.eval()
    acc = []
    tot_samples = 0
    with torch.no_grad():
        for items in tqdm(loader):
            images = items['img'].to(device)
            target =  items['anomaly'].to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
                image_features, _ = model.encode_image(images)

                AnomalyCLIP_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}
                prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
                prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id = None)

                text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
                text_features = torch.stack(torch.chunk(text_features, dim = 0, chunks = 2), dim = 1)

            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.float().to(device)
            
            model = model.cuda() 

            text_features = text_features/text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.float().to(device)

            text_features = text_features.to(dtype=torch.float32)
            image_features = image_features.to(dtype=torch.float32)

            text_probs = image_features.unsqueeze(1) @ text_features.permute(0, 2, 1)
            text_probs = text_probs[:, 0, ...]/0.07
            acc.append(cls_acc(text_probs, target) * len(text_probs))
            # visualizer(items['img_path'], anomaly_map.detach().cpu().numpy(), args.image_size, args.save_path, cls_name, text_probs)

            
            # tot_samples += len(text_probs)
    # print(f' acc ===> {acc} \n mean ===> {sum(acc)/len(acc)} \n\n')
    # acc /= tot_samples

    return sum(acc)/len(acc)


def test(args):
    VALIDATION = False
    logit_scale = 100

    logger = get_logger(args.save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    AnomalyCLIP_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}
    
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details = AnomalyCLIP_parameters)
    model.eval()
    model.to(device)

    preprocess, target_transform = get_transform(args)
    train_data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name = args.dataset, mode='train')
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    test_data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name = args.dataset)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    # test_features, test_labels = pre_load_features(model, train_dataloader)

    # obj_list = test_data.obj_list

    # test_features = test_features.cpu()
    # test_labels = test_labels.cpu()

    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    checkpoint = torch.load(args.checkpoint_path)
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device)
    
    model = model.cuda() 
    model.visual.DAPM_replace(DPAM_layer = 20)

    # prompt 생성 (인코딩, 정규화)
    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id = None)
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim = 0, chunks = 2), dim = 1)
    text_features = text_features/text_features.norm(dim=-1, keepdim=True)

    list_lora_layers = apply_lora(args, model)

    if True:
        model = model.cuda() 
        load_lora(args, list_lora_layers)
        print('\n\n dskfjdslkfj \n\n')
        acc_test = evaluate_lora(args, model, test_dataloader, test_data)
        print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
        return

    mark_only_lora_as_trainable(model)  # lora layer 만 학습 가능하게 변경

    total_iters = args.n_iters * args.shots
    # total_iters = 100

    optimizer = torch.optim.AdamW(get_lora_parameters(model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)
    
    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    

    # training LoRA
    scaler = torch.cuda.amp.GradScaler()
    count_iters = 0
    finish = False
    while count_iters < total_iters:
        model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        loss_list = []
        image_loss_list = []

        # args.encoder == 'both' 가 디폴트
        if args.encoder == 'vision': 
            text_features = text_features.t().half()

        for items in tqdm(train_dataloader):
            images = items['img'].to(device)
            target =  items['anomaly'].to(device)
            gt = items['img_mask'].squeeze().to(device)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id = None)

            if args.encoder == 'text' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
                    model = model.cuda() 
                    class_embeddings = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
                    class_embeddings = torch.stack(torch.chunk(class_embeddings, dim = 0, chunks = 2), dim = 1)
                text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True) # torch.float32
                
            if args.encoder == 'vision' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
                    model = model.cuda() 
                    image_features, patch_features = model.encode_image(images, args.features_list)  # img 임베딩(torch.float16)

            else:  
                # args.encoder == 'text' 
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
                        model = model.to(device)
                        image_features, patch_features = model.encode_image(images, args.features_list, DPAM_layer = 20)

            image_features = image_features/image_features.norm(dim=-1, keepdim=True)

            text_probs = image_features.unsqueeze(1) @ text_features.permute(0, 2, 1)
            text_probs = text_probs[:, 0, ...]/0.07
            image_loss = F.cross_entropy(text_probs.squeeze(), target.long().cuda())
            image_loss_list.append(image_loss.item())

            similarity_map_list = []

            for idx, patch_feature in enumerate(patch_features):
                if idx >= args.feature_map_layer[0]:
                    patch_feature = patch_feature/ patch_feature.norm(dim = -1, keepdim = True)
                    similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
                    similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size).permute(0, 3, 1, 2)
                    similarity_map_list.append(similarity_map)
            loss = 0
            for i in range(len(similarity_map_list)):
                loss += loss_focal(similarity_map_list[i], gt)
                loss += loss_dice(similarity_map_list[i][:, 1, :, :], gt)
                loss += loss_dice(similarity_map_list[i][:, 0, :, :], 1-gt)
            # cosine_similarity = logit_scale * image_features @ text_features[0].t() # img, prompt 간 유사도
            # loss = F.cross_entropy(cosine_similarity, target) # 계산한 유사도 - gt 비교해서 loss 계산
            acc_train += cls_acc(text_probs, target) * target.shape[0]
            loss_epoch += loss * target.shape[0]
            loss_epoch += loss * target.shape[0]
            tot_samples += target.shape[0]
            
            optimizer.zero_grad()
            scaler.scale(loss + image_loss).backward()
            scaler.step(optimizer)

            scaler.update()
            scheduler.step()
            
            count_iters += 1

            loss_list.append(loss)
            
            if count_iters == total_iters:
                break
            
        if count_iters < total_iters:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()[0]
            print(f'*** {count_iters} ***')
            print('LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(current_lr, acc_train, loss_epoch))

            
    
        # Eval
        if VALIDATION:
            model = model.cuda() 
            model.eval()
            test_data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name = args.dataset)
            test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
            acc_val = evaluate_lora(args, model, test_dataloader, test_data)
            print("**** Val accuracy: {:.2f}. ****\n".format(acc_val))

    model = model.cuda() 
    model.eval()
    test_data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name = args.dataset)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    acc_test = evaluate_lora(args, model, test_dataloader, test_data)
    print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
    
    if args.save_path != None:
        save_lora(args, list_lora_layers)


    # results = {}
    # metrics = {}
    # for obj in obj_list:
    #     results[obj] = {}
    #     results[obj]['gt_sp'] = []
    #     results[obj]['pr_sp'] = []
    #     results[obj]['imgs_masks'] = []
    #     results[obj]['anomaly_maps'] = []
    #     metrics[obj] = {}
    #     metrics[obj]['pixel-auroc'] = 0
    #     metrics[obj]['pixel-aupro'] = 0
    #     metrics[obj]['image-auroc'] = 0
    #     metrics[obj]['image-ap'] = 0

    # prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    # checkpoint = torch.load(args.checkpoint_path)
    # prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    # prompt_learner.to(device)
    # model.to(device)
    # model.visual.DAPM_replace(DPAM_layer = 20)

    # model.to(device)
    # for idx, items in enumerate(tqdm(test_dataloader)):
        
    #     image = items['img'].to(device)
    #     cls_name = items['cls_name']
    #     cls_id = items['cls_id']
    #     gt_mask = items['img_mask']
    #     gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
    #     results[cls_name[0]]['imgs_masks'].append(gt_mask)  # px
    #     results[cls_name[0]]['gt_sp'].extend(items['anomaly'].detach().cpu())

    #     with torch.no_grad():
    #         image_features, patch_features = model.encode_image(image, features_list, DPAM_layer = 20)
    #         image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    #         text_probs = image_features @ text_features.permute(0, 2, 1)
    #         text_probs = (text_probs/0.07).softmax(-1)
    #         text_probs = text_probs[:, 0, 1]
    #         anomaly_map_list = []
    #         for idx, patch_feature in enumerate(patch_features):
    #             if idx >= args.feature_map_layer[0]:
    #                 patch_feature = patch_feature/ patch_feature.norm(dim = -1, keepdim = True)
    #                 similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
    #                 similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size)
    #                 anomaly_map = (similarity_map[...,1] + 1 - similarity_map[...,0])/2.0
    #                 anomaly_map_list.append(anomaly_map)

    #         anomaly_map = torch.stack(anomaly_map_list)
            
    #         anomaly_map = anomaly_map.sum(dim = 0)
    #         results[cls_name[0]]['pr_sp'].extend(text_probs.detach().cpu())
    #         anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma = args.sigma)) for i in anomaly_map.detach().cpu()], dim = 0 )
    #         results[cls_name[0]]['anomaly_maps'].append(anomaly_map)
    #         visualizer(items['img_path'], anomaly_map.detach().cpu().numpy(), args.image_size, args.save_path, cls_name, results[cls_name[0]]['pr_sp'])

    # table_ls = []
    # image_auroc_list = []
    # image_ap_list = []
    # pixel_auroc_list = []
    # pixel_aupro_list = []
    # for obj in tqdm(obj_list):
    #     table = []
    #     table.append(obj)
    #     results[obj]['imgs_masks'] = torch.cat(results[obj]['imgs_masks'])
    #     results[obj]['anomaly_maps'] = torch.cat(results[obj]['anomaly_maps']).detach().cpu().numpy()
    #     try:
    #         image_auroc = image_level_metrics(results, obj, "image-auroc")
    #         image_ap = image_level_metrics(results, obj, "image-ap")
    #     except:
    #         image_auroc = 0
    #         image_ap = 0
    #     try:
    #         pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
    #         pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
    #     except:
    #         pixel_auroc = 0
    #         pixel_aupro = 0
    #     table.append(str(np.round(pixel_auroc * 100, decimals=1)))
    #     table.append(str(np.round(pixel_aupro * 100, decimals=1)))
    #     table.append(str(np.round(image_auroc * 100, decimals=1)))
    #     table.append(str(np.round(image_ap * 100, decimals=1)))
    #     image_auroc_list.append(image_auroc)
    #     image_ap_list.append(image_ap) 
    #     pixel_auroc_list.append(pixel_auroc)
    #     pixel_aupro_list.append(pixel_aupro)
    #     table_ls.append(table)
    # table_ls.append(['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
    #                 str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1)), 
    #                 str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
    #                 str(np.round(np.mean(image_ap_list) * 100, decimals=1))])
    # results = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro', 'image_auroc', 'image_ap'], tablefmt="pipe")
    # logger.info("\n%s", results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("AnomalyCLIP", add_help=True)
    # paths
    parser.add_argument("--data_path", type=str, default="./data/visa", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoint/', help='path to checkpoint')
    # model
    parser.add_argument('--backbone', default='ViT-L/14', type=str)
    parser.add_argument("--dataset", type=str, default='mvtec')
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int,  nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--metrics", type=str, default='image-pixel-level')
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--sigma", type=int, default=4, help="zero shot")
    # Training arguments
    parser.add_argument('--shots', default=16, type=int)
    parser.add_argument('--lr', default=2e-1, type=float)
    parser.add_argument('--n_iters', default=500, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    # LoRA arguments
    parser.add_argument('--position', type=str, default='all', choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'], help='where to put the LoRA modules')
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='both')
    parser.add_argument('--params', metavar='N', type=str, nargs='+', default=['q', 'k', 'v'], help='list of attention matrices where putting a LoRA') 
    parser.add_argument('--r', default=2, type=int, help='the rank of the low-rank matrices')
    parser.add_argument('--alpha', default=1, type=int, help='scaling (see LoRA paper)')
    parser.add_argument('--dropout_rate', default=0.25, type=float, help='dropout rate applied before the LoRA module')
    
    # parser.add_argument('--save_path', default=None, help='path to save the lora modules after training, not saved if None')
    parser.add_argument('--filename', default='lora_weights', help='file name to save the lora weights (.pt extension will be added)')
    
    parser.add_argument('--eval_only', default=False, action='store_true', help='only evaluate the LoRA modules (save_path should not be None)')
    
    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)
    test(args)
