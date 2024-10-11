import torchvision.transforms as transforms
# from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
from AnomalyCLIP_lib.transform import image_transform
from AnomalyCLIP_lib.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)

def get_transform(args):
    preprocess = image_transform(args.image_size, is_train=False, mean = OPENAI_DATASET_MEAN, std = OPENAI_DATASET_STD)
    target_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor()
    ])
    preprocess.transforms[0] = transforms.Resize(size=(args.image_size, args.image_size), interpolation=transforms.InterpolationMode.BICUBIC,
                                                    max_size=None, antialias=None)
    preprocess.transforms[1] = transforms.CenterCrop(size=(args.image_size, args.image_size))
    return preprocess, target_transform

def cls_acc(output, target, topk=1):
    # print(f' \n output ===> {output} \n\n target ===> {target} \n\n')
    if isinstance(output, list):
        output = torch.tensor(output)
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    print(f' \n pred ===> {pred} \n\n target ===> {target} \n\n correct ===> {correct} \n\n')

    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]

    return acc


from tqdm import tqdm
import torch
# import clip

# def clip_classifier(classnames, template, clip_model):
#     with torch.no_grad():
#         clip_weights = []
#         for classname in classnames:
#             # Tokenize the prompts
#             classname = classname.replace('_', ' ')
#             texts = [t.format(classname) for t in template]
#             texts = clip.tokenize(texts).cuda()
#             class_embeddings = clip_model.encode_text(texts)
#             class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
#             class_embedding = class_embeddings.mean(dim=0)
#             class_embedding /= class_embedding.norm()
#             clip_weights.append(class_embedding)
#         clip_weights = torch.stack(clip_weights, dim=1).cuda()
        
#     return clip_weights


def pre_load_features(clip_model, loader):
    features, labels = [], []
    with torch.no_grad():
        for i, items in enumerate(tqdm(loader)):
            images = items['img']
            target =  items['anomaly']
            images, target = images.cuda(), target.cuda()
            image_features, _ = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features.cpu())
            labels.append(target.cpu())
        features, labels = torch.cat(features), torch.cat(labels)
    
    return features, labels
