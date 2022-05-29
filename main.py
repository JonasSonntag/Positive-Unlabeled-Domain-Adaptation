import os
import argparse
import tqdm
import argparse
import numpy as np
import tqdm
from itertools import chain
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import json
from utils import CLEFImage, weights_init, print_args

from model.net import ResNet50_mod_name, ResClassifier
from pretrained_model_rename import rename_key_resnet
import glob
from pu_loss import PULoss
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import torch.utils.data as data_utils

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", default='data/OfficeCaltech/list')

parser.add_argument("--source", default='Product', type=str)
parser.add_argument("--target", default='Real World',type=str)

parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--epoch", default=50, type=int)

parser.add_argument("--lr1", default=0.0016102287018430657, type=float)
parser.add_argument("--lr2", default=0.00012266834924870657, type=float)

parser.add_argument("--decay_epoch", default=35, type=int)
parser.add_argument("--decay", default=0.0035, type=float)

parser.add_argument("--class_num", default=2, type=int)

parser.add_argument("--weight_L2norm", default=0.05, type=float)
parser.add_argument("--weight_entropy", default=0.05, type=float)
parser.add_argument("--dropout_p", default=0.1, type=float)

parser.add_argument("--repeat", default='20', type=int)
parser.add_argument("--alpha", default=0.93, type=float)
parser.add_argument("--beta", default=0.86, type=float)

#parser.add_argument("--alpha", default=0.0013805225879326784, type=float)
#parser.add_argument("--beta", default=11.656549135997235, type=float)
#parser.add_argument("--gamma", default=7.14834918536929, type=float)

parser.add_argument("--alpha_dfa", default=0.049, type=float)
parser.add_argument("--beta_dfa", default=0.2, type=float)


parser.add_argument("--c", default=1, type=int)
parser.add_argument("--experiment_name", default ="", type=str)
parser.add_argument("--sm_mode", default =0, type=int, help='Set to one if run in SageMaker')



args = parser.parse_args()
print_args(args)
#enbale reproducability 
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

print('--Renaming Modell')
__all__ = ['resnet50']

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}
torch.manual_seed(0) 
torch.use_deterministic_algorithms(True)
writer = SummaryWriter()
new_state_dict,state_dict = rename_key_resnet()

    
result_dict = {metric : [] for metric in ['Accuracy' ]}

def transforms_images(image_tensor):
    """transform tensor for consistency regularization
    :param image_tensor: torch.Tensor [B, C, H, W]
    """

    device = image_tensor.device
    _, _, H, W = image_tensor.shape

    cr_transforms = transforms.Compose([
        transforms.ToPILImage(),
        #transforms.ColorJitter(brightness = 0.5, contrast=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize((0.5, ), (0.5, ))
    ])

    image_tensor = torch.cat([cr_transforms(image).unsqueeze_(0) 
                              for image in image_tensor.cpu()], dim=0)

    return image_tensor.to(device=device)

def get_cls_loss(pred, gt):
    cls_loss = F.nll_loss(F.log_softmax(pred), gt)
    return cls_loss

def get_L2norm_loss_self_driven(x):
    radius = x.norm(p=2, dim=1).detach()
    assert radius.requires_grad == False
    radius = radius + 1.0
    l = ((x.norm(p=2, dim=1) - radius) ** 2).mean()
    return args.weight_L2norm * l

def get_entropy_loss(p_softmax):
    mask = p_softmax.ge(0.000001)
    mask_out = torch.masked_select(p_softmax, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return args.weight_entropy * (entropy / float(p_softmax.size(0)))   



#Check if we are in sagemaker mode
if args.sm_mode:
    root_path = '/opt/ml/input/data/training/'
else:
    root_path = 'data/OfficeHome/'




for rep in range(args.repeat):
    random_state = np.random.RandomState(rep)
    torch.manual_seed(rep) 
    source_root = root_path + args.source
    source_label = os.path.join(root_path + "list/", args.source+'.txt')

    
    target_root = root_path + args.target
    target_label = os.path.join(root_path + "list/", args.target+'.txt')

    train_transform = transforms.Compose([
        transforms.Scale((256, 256)),
        transforms.CenterCrop((221, 221)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    source_set = CLEFImage(source_root, source_label, train_transform, pos_label_percent=args.c, random_state=random_state)

    target_set = CLEFImage(target_root, target_label, train_transform, pos_label_percent=args.c, random_state=random_state)
    
    prior = torch.tensor(np.mean([target_set. __getitem__(i)[2] for i in range(len(target_set))]))
    t_imgs_all = torch.stack([target_set. __getitem__(i)[0] for i in range(len(target_set))])
    t_labels_pu_all = torch.tensor([target_set. __getitem__(i)[1] for i in range(len(target_set))])
    
    t_imgs_pos = t_imgs_all[t_labels_pu_all==1]
    
    t_imgs_pos_augmented = transforms_images(t_imgs_pos)
    
    if t_imgs_pos.shape[0] == 1:
        t_imgs_pos = torch.cat([t_imgs_pos, t_imgs_pos_augmented])
    num_pos_examples = len(t_imgs_pos)
 
    source_loader = torch.utils.data.DataLoader(source_set, batch_size=args.batch_size,
        shuffle=1, num_workers=0)

    target_loader = torch.utils.data.DataLoader(target_set, batch_size=args.batch_size,
        shuffle=1, num_workers=0)

    netG = ResNet50_mod_name().cuda()
    state_dict = torch.load('model/resnet_model.pth')
    netG.load_state_dict(state_dict)
    
    netF = ResClassifier(class_num=args.class_num, dropout_p=args.dropout_p).cuda()
    netF.apply(weights_init)
    
    pu_loss =  PULoss(prior=prior, nnPU=True)

    # initialize tensors
    feat_t_recon = torch.ones([args.batch_size+num_pos_examples, 3, 221, 221]).cuda()
    feat_zn_recon = torch.ones([args.batch_size+num_pos_examples, 3, 221, 221]).cuda()
    # initialize a L1 loss for DAL
    CriterionDAL = nn.L1Loss().cuda()

    Tensor = torch.cuda.FloatTensor

    opt_g = optim.SGD(netG.parameters(), lr=args.lr1, weight_decay=0.0005)
    opt_f = optim.SGD(netF.parameters(), lr=args.lr1, momentum=0.9, weight_decay=0.0005)
    pos_threshold_source = [Tensor([0.5])]
    neg_threshold_source = [Tensor([0.5])]
    pos_threshold_list = []
    print(f'Pos Examples: {num_pos_examples}')
    new_dataset = {'images' : [], 'noise_labels': [], 'labels': [] }
    for epoch in range(1, args.epoch+1):

        pos_emb = netG(t_imgs_pos.cuda())
        pos_pred = netF(pos_emb)[1]
        pos_threshold_target = torch.mean(F.softmax(pos_pred)[:,1])
        #torch.cat(pos_threshold_source)
        pos_threshold_source = torch.mean(Tensor(pos_threshold_source))
        pos_threshold_source_prev = pos_threshold_source.clone()
        neg_threshold_source = torch.mean(Tensor(neg_threshold_source))
        neg_threshold_source_prev = neg_threshold_source.clone()        
        #print(f'pos_threshold_target: {pos_threshold_target}')
        pos_threshold_list = pos_threshold_list + [float(pos_threshold_target)]
        
        #print(f'pos_threshold_source: {pos_threshold_source}')
        pos_threshold_source = []
        neg_threshold_source = []
        source_loader_iter = iter(source_loader)
        target_loader_iter = iter(target_loader)
    

        correct = 0.
        count = 0
        append_pos = 0

        for i, (t_imgs, t_labels_pu, t_labels, t_index) in tqdm.tqdm(enumerate(target_loader_iter)):
            test_only = False
            try:
                s_imgs, s_labels_pu, s_labels, s_index = source_loader_iter.next()
            except:
                source_loader_iter = iter(source_loader)
                s_imgs, s_labels_pu, s_labels, s_index = source_loader_iter.next()

            t_imgs_test = t_imgs
            t_labels_pu_test = t_labels_pu
            t_labels_test = t_labels
            
            #Concat positive examples fior PU setting
            t_imgs = torch.cat((t_imgs, t_imgs_pos))
            s_imgs = torch.cat((s_imgs, s_imgs[:t_imgs_pos.shape[0]]))
            t_labels_pu = torch.cat((t_labels_pu, torch.tensor([1]*num_pos_examples)))
            t_index = torch.cat((t_index, torch.tensor([-1]*num_pos_examples)))
            t_labels = torch.cat((t_labels, torch.tensor([1]*num_pos_examples)))
            s_labels = torch.cat((s_labels, s_labels[:t_imgs_pos.shape[0]]))
        

            if (s_imgs.size(0) != (args.batch_size+num_pos_examples)) or( t_imgs.size(0) != (args.batch_size+num_pos_examples)):
                test_only = True
                if( s_imgs.size(0)<=1) or  (t_imgs.size(0) <=1):
                    continue            
           
            s_imgs = Variable(s_imgs.cuda())
            s_labels = Variable(s_labels.cuda())     
            t_imgs = Variable(t_imgs.cuda())
            t_labels = Variable(t_labels.cuda())
            t_labels_pu = Variable(t_labels_pu.cuda())
            t_imgs_test = Variable(t_imgs_test.cuda())
            t_labels_test = Variable(t_labels_test.cuda())
            t_labels_pu_test = Variable(t_labels_pu_test.cuda())
            zn = Variable(Tensor(np.random.normal(0,1, (args.batch_size+num_pos_examples, 2048))))
        
            opt_g.zero_grad()
            opt_f.zero_grad()


            
            s_bottleneck = netG(s_imgs)
            t_bottleneck = netG(t_imgs)   
            s_fc2_emb, s_logit = netF(s_bottleneck)
            t_fc2_emb, t_logit = netF(t_bottleneck)
            s_cls_loss = get_cls_loss(s_logit, s_labels)
            t_prob = F.softmax(t_logit)
            
            pos_pred_source = s_logit[s_labels==1] #netF(s_bottleneck[s_labels==1])[1] 
            neg_pred_source = s_logit[s_labels==0] #netF(s_bottleneck[s_labels==0])[1] 
            
            pos_threshold_source = pos_threshold_source + [torch.mean(F.softmax(pos_pred_source)[:,1]).item()]
            neg_threshold_source = neg_threshold_source + [torch.mean(F.softmax(neg_pred_source)[:,1]).item()]

            #Find likely negative and positive#
            unlabelled_pred = F.softmax(t_logit[t_labels_pu==-1], dim=1)[:,1]
            #print(unlabelled_pred)
            likely_pos_mask = (unlabelled_pred > ((pos_threshold_source_prev + pos_threshold_target) / 2))
            likely_neg_mask = (unlabelled_pred < neg_threshold_source_prev)
            
            correct_pos = torch.sum(t_labels[t_labels_pu==-1][likely_pos_mask] == 1)/float(torch.sum(likely_pos_mask))
            correct_neg = torch.sum(t_labels[t_labels_pu==-1][likely_neg_mask] == 0)/float(torch.sum(likely_neg_mask))
            
            likely_pos_imgs = t_index[t_labels_pu==-1][likely_pos_mask]
            likely_neg_imgs = t_index[t_labels_pu==-1][likely_neg_mask]
            if epoch > 20:
                new_dataset['images'] = new_dataset['images'] + likely_pos_imgs.detach().tolist() + likely_neg_imgs.detach().tolist()
                new_dataset['noise_labels'] = new_dataset['noise_labels'] + unlabelled_pred[likely_pos_mask].detach().tolist() + unlabelled_pred[likely_neg_mask].detach().tolist()
                new_dataset['labels'] = new_dataset['labels'] + t_labels[t_labels_pu==-1][likely_pos_mask].detach().tolist() + t_labels[t_labels_pu==-1][likely_neg_mask].detach().tolist()
                   

            pu_loss_ = pu_loss(t_logit[:,1].view(-1), t_labels_pu)
            writer.add_scalar("Loss/pu", pu_loss_, epoch)
                

        
            s_fc2_L2norm_loss = get_L2norm_loss_self_driven(s_fc2_emb)
            t_fc2_L2norm_loss = get_L2norm_loss_self_driven(t_fc2_emb)
            
            writer.add_scalar("Loss/sL2", s_fc2_L2norm_loss, epoch)
            writer.add_scalar("Loss/tL2", t_fc2_L2norm_loss, epoch)
            #kl-divergence
            feat_s_kl = s_bottleneck.view(-1,2048)
            if not test_only:
                loss_kld_s = F.kl_div(F.log_softmax(feat_s_kl), F.softmax(zn))

                #distribution alignment loss (DAL)
                loss_dal= CriterionDAL(feat_t_recon, feat_zn_recon)
                writer.add_scalar("Loss/dal", loss_dal, epoch)
                
                t_entropy_loss = get_entropy_loss(t_prob)
                writer.add_scalar("Loss/entropy", t_entropy_loss, epoch)
                writer.add_scalar("Loss/s_cls", s_cls_loss, epoch)
                #updated loss function
                

                loss = args.alpha * s_cls_loss + args.beta * (s_fc2_L2norm_loss + t_fc2_L2norm_loss + t_entropy_loss + args.alpha_dfa * loss_kld_s + args.beta_dfa * loss_dal) +  pu_loss_ 
                #loss = args.gamma * (s_cls_loss + s_fc2_L2norm_loss + t_fc2_L2norm_loss + t_entropy_loss + args.alpha * loss_kld_s + args.beta * loss_dal) +  pu_loss_

                loss.backward()
        
        
                opt_g.step()
                opt_f.step()

                # calculate decoded samples for the next iteration
                feat_t_recon = netG(t_imgs, is_deconv=True).detach()
                feat_zn_recon = netG.decode(zn).detach()
            else:
                print('Skipping train')
            count = count + 1
            #Eval model predictions for the unlabeled target batch
            netG.eval()
            netF.eval()
            t_bottleneck = netG(t_imgs_test)      
            t_fc2_emb, t_logit = netF(t_bottleneck)

            pred_test = t_logit[t_labels_pu_test==-1].data.max(1)[1]
            correct += float(pred_test.eq(t_labels_test[t_labels_pu_test==-1].data).cpu().sum().detach().numpy())/float(pred_test.shape[0])

            netG.train()
            netF.train()    
        res = correct/count
        print(f'Result first task: {res}')
        
    #------Pseudo-Label extractor    
    target_label_df = pd.DataFrame({'image':new_dataset['images'], 'soft_label': new_dataset['noise_labels'], 'real_labels': new_dataset['labels']})
    pos_thres = np.mean(pos_threshold_list)
    neg_thres = 1 - pos_thres
    print(pos_thres)
    temp = target_label_df.groupby('image').apply(lambda x: -1 if len(x)<10 else x.soft_label.mean() if ((x.soft_label.max() < neg_thres) or (x.soft_label.min() > pos_thres )) else -1).reset_index(name='soft_label')
    
    #target_label_df.to_csv('soft_label.csv')
    #Check accuracy
    real_label_df = target_label_df.groupby('image').apply(lambda x: x.real_labels.mean()).reset_index(name='real_label')
    acc_df = real_label_df.merge(temp, on ='image')
    acc_df = acc_df[acc_df.soft_label!=-1]
    print(f"Accuracy of training dataset: {((acc_df.soft_label>0.5).astype(int) ==acc_df.real_label).mean()} ")
    print(f" Considered Items{(temp.soft_label!=-1).sum()}")
    
    #---------------Build new tensordataset 
    temp = temp[temp.soft_label != -1]
    if len(temp)>0:
        
        images = torch.cat([torch.unsqueeze(target_set.__getitem__(index)[0],0) for index in temp.image.values])
        labels = torch.Tensor(temp.soft_label.values)
        images = torch.cat([images, t_imgs_pos])
        labels = torch.cat([labels, torch.Tensor([1]*t_imgs_pos.shape[0]).type(torch.FloatTensor)])

        #-------------2nd step-------------

        netG2 = ResNet50_mod_name().cuda()
        state_dict = torch.load('model/resnet_model.pth')
        netG2.load_state_dict(state_dict)

        netF2 = ResClassifier(class_num=args.class_num, dropout_p=args.dropout_p).cuda()
        netF2.apply(weights_init)

        opt_g2 = optim.SGD(netG2.parameters(), lr=args.lr2, weight_decay=0.0005)
        opt_f2 = optim.SGD(netF2.parameters(), lr=args.lr2, momentum=0.9, weight_decay=0.0005)

        g_scheduler = torch.optim.lr_scheduler.StepLR(opt_g2, args.decay_epoch, args.decay)
        f_scheduler = torch.optim.lr_scheduler.StepLR(opt_f2, args.decay_epoch, args.decay)

        train = data_utils.TensorDataset(images, labels)
        train_loader_2step = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss().cuda()
        for epoch in range(50):
            correct = 0
            count = 0
            for image, soft_label in train_loader_2step:
                image = Variable(image.cuda())
                soft_label = Variable(soft_label.cuda())
                embedding = netG2(image)
                _, logit = netF2(embedding)
                pred_prob = F.softmax(logit)[:,1]

                soft_label = (soft_label>0.5).type(torch.LongTensor).cuda()
                loss = criterion(logit, soft_label)
                loss.backward()
                opt_g2.step()
                opt_f2.step()
                g_scheduler.step()
                f_scheduler.step()

                #Eval on unlabeled target data            
                netG2.eval()
                netF2.eval()

                for t_imgs, t_labels_pu, t_labels, t_index in target_loader: 
                    t_imgs = Variable(t_imgs.cuda())
                    t_labels = Variable(t_labels.cuda())
                    t_labels_pu = Variable(t_labels_pu.cuda())

                    t_bottleneck = netG2(t_imgs)      
                    t_fc2_emb, t_logit = netF2(t_bottleneck)
                    pred_test = t_logit[t_labels_pu==-1].data.max(1)[1]
                    correct += float(pred_test.eq(t_labels[t_labels_pu==-1].data).cpu().sum().detach().numpy())/float(pred_test.shape[0])
                    count = count +1
                res1 = correct/count
                print(f'Result in second step: {res1}')
                #pred_pos_target = netF2(netG2(t_imgs_pos.cuda()))[1].data.max(1)[1]
                #res_labelled_target =  float(torch.sum(pred_pos_target==1))/float(pred_pos_target.shape[0])
                #print(f'Result in second step for pos labelled: {res_labelled_target}')
                netG.train()
                netF.train()
    else:
        res1 = 0

    print(f'Val-Accuracy: {np.round(res1,4)};')
    result_dict['Accuracy'] = result_dict['Accuracy'] +  [res1]
    
    record_file = f"record/{args.experiment_name}/{'result_PU.json'}"
    if args.sm_mode:
        record_file = "/opt/ml/model/test_result.json"
    if not os.path.exists(f"record/{args.experiment_name}"):
        os.makedirs(f"record/{args.experiment_name}")
    with open(record_file, 'w') as outfile:
        json.dump(result_dict, outfile)





