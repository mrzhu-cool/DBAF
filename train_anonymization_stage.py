import argparse
from argparse import Namespace
from argparse import Namespace
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import save_image
import os
from PIL import Image
from tqdm import tqdm
import math
from torch import autograd
from models.stylegan2.op import conv2d_gradfix
from models.cbam import CBAM
from models.psp import pSp
from models.stylegan2.model_revise_c3 import Generator,Discriminator,EqualLinear
from models.ranger import Ranger
from models.disentanglement_id_step1 import DisentanglementId
from losses.parse_related_loss import bg_loss
from losses.id_loss_new import IDLossExtractor
from losses.lpips.lpips import LPIPS

to_tensor_transform = transforms.ToTensor()
parser = argparse.ArgumentParser()

# file dir
parser.add_argument("--train_img_dir", type=str, default="./dataset/train")
parser.add_argument("--test_img_dir", type=str, default="./dataset/test")
parser.add_argument("--save_dir", type=str, default="./output/stage2")
parser.add_argument("--device", type=str, default="cuda:0")

# pretrained weights
parser.add_argument("--e4e_path", type=str, default="./pretrain/e4e_ffhq_encode_256.pt")
parser.add_argument("--stylegan_path", type=str, default="./pretrain/stylegan2-ffhq-256.pt")
parser.add_argument("--id_encoder_path", type=str, default="./pretrain/model_ir_se50.pth")
parser.add_argument("--parsenet_weights", type=str, default="./pretrain/parsenet.pth")

# pretrained weights in stage 1
parser.add_argument("--finetune_stylegan_path", type=str, default="./output/stage1/best_result/finetune_decoder_best_model.pth")
parser.add_argument("--w_id_disentanglement_path", type=str, default="./output/stage1/best_result/w_disentanglement_best_model.pth")
parser.add_argument("--attr_block_path", type=str, default="./output/stage1/best_result/attr_block_best_model.pth")
parser.add_argument("--discriminator_path", type=str, default="./output/stage1/best_result/discriminator_best_model.pth")

# Parameter settings
parser.add_argument("--stylegan_size", type=int, default=256)
parser.add_argument("--batchsize", type=int, default=8)
parser.add_argument("--test_batchsize", type=int, default=8)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--epoch", type=int, default=5)
parser.add_argument("--lambdaid", type=float, default=2.0)
parser.add_argument("--lambdarec", type=float, default=0.05)
parser.add_argument("--lambdalpips", type=float, default=1.0)
parser.add_argument("--id_cos_margin", type=float, default=0.1)
parser.add_argument("--lambdalatent", type=float, default=0.1)
parser.add_argument("--lambdaparse", type=float, default=0.1)
parser.add_argument("--r1", type=float, default=10.0)
parser.add_argument("--morefc_num", type=int, default=4)

args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
best_result_dir = os.path.join(args.save_dir, "best_result")
os.makedirs(best_result_dir, exist_ok=True)
epoch_dir = os.path.join(args.save_dir, "epoch")
os.makedirs(epoch_dir, exist_ok=True)
writer = SummaryWriter(args.save_dir)

def load_e4e(opts):
        ckpt = torch.load(opts.e4e_path, map_location=opts.device)
        args = ckpt['opts']
        args['checkpoint_path'] = opts.e4e_path
        args['stylegan_weights'] = opts.stylegan_path
        args = Namespace(**args)
        net = pSp(args)
        return net.to(opts.device)


class SimpleDataset(Dataset):
    
    def __init__(self,path,transform=None):
        print('Image path',path)
        self.file_path=path
        self.file_list=os.listdir(path)
        self.file_list.sort()
        if transform is not None:
            self.transform=transform
        else:
            self.transform=transforms.Compose([transforms.ToTensor()])

    def __getitem__(self,index):
        file_name=self.file_list[index]
        image=Image.open(os.path.join(self.file_path,file_name)).convert('RGB')
        if self.transform is not None:
            image=self.transform(image)
        return image,file_name
        
    def __len__(self):
        return len(self.file_list)
    
class AttrBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.cbam_c3 = CBAM(gate_channels=512)
        self.cbam_c2 = CBAM(gate_channels=512)
        self.cbam_c1 = CBAM(gate_channels=512)

        self.conv_c3 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.conv_c2 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.conv_c1 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def forward(self, input_c3,input_c2,input_c1):
        out_c3 = self.conv_c3(input_c3)
        out_c2 = self.conv_c2(input_c2)
        out_c1 = self.conv_c1(input_c1)

        out_c3 = self.cbam_c3(out_c3)
        out_c2 = self.cbam_c2(out_c2)
        out_c1 = self.cbam_c1(out_c1)

        return out_c3,out_c2,out_c1
        
class cMLP(nn.Module):
    def __init__(self, morefc_num=4, lr_mlp=0.01):
        super(cMLP, self).__init__()
        self.module_list = [EqualLinear(1024, 2048, lr_mul=lr_mlp, activation="fused_lrelu"),
                            EqualLinear(2048, 1024, lr_mul=lr_mlp, activation="fused_lrelu"),
                            EqualLinear(1024, 512, lr_mul=lr_mlp, activation="fused_lrelu")]
        for i in range(morefc_num):
            self.module_list.append(EqualLinear(512, 512, lr_mul=lr_mlp, activation="fused_lrelu"))
        self.module_list.append(nn.Linear(512, 512))
        self.model = nn.Sequential(*(self.module_list))

    def forward(self, input_tensor):
        return self.model(input_tensor)

class AnonymizationNet(nn.Module):
    def __init__(self,n_latent):
        super().__init__()
        self.mapper = nn.ModuleList([cMLP() for _ in range(3)]
        )

    def forward(self,w_id):
        outputs = []
        for i in range(w_id.size(1)):
            if i < 4:
                out = self.mapper[0](w_id[:,i,:])
            elif i < 8:
                out = self.mapper[1](w_id[:,i,:])
            else:
                out = self.mapper[2](w_id[:,i,:]) 
            outputs.append(out.unsqueeze(1))
        outputs = torch.cat(outputs,dim=1)

        return outputs

def sample_w(args,decoder,latents_num):
        z_1 = torch.randn(args.batchsize, 512, device=args.device)
        z_2 = torch.randn(args.batchsize, 512, device=args.device)
        z_3 = torch.randn(args.batchsize, 512, device=args.device)
        w_1 = decoder.style(z_1).unsqueeze(1).repeat(1,4,1)
        w_2 = decoder.style(z_2).unsqueeze(1).repeat(1,4,1)
        w_3 = decoder.style(z_3).unsqueeze(1).repeat(1,6,1)
        w = torch.cat([w_1,w_2,w_3],dim=1)
        return w

def test_sample_w(args,decoder,latents_num):
        z_1 = torch.randn(args.test_batchsize, 512, device=args.device)
        z_2 = torch.randn(args.test_batchsize, 512, device=args.device)
        z_3 = torch.randn(args.test_batchsize, 512, device=args.device)
        w_1 = decoder.style(z_1).unsqueeze(1).repeat(1,4,1)
        w_2 = decoder.style(z_2).unsqueeze(1).repeat(1,4,1)
        w_3 = decoder.style(z_3).unsqueeze(1).repeat(1,6,1)
        w = torch.cat([w_1,w_2,w_3],dim=1)
        return w

def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()

if __name__ == '__main__':
    id_encoder= IDLossExtractor(args).to(args.device).eval()
    lpips_loss = LPIPS(args,net_type='alex').to(args.device).eval()
    parse_loss = bg_loss.ParseLoss(args).to(args.device).eval()
    e4e = load_e4e(args).to(args.device).eval()
    
    w_disentanglement = DisentanglementId().to(args.device).train()
    w_disentanglement.load_state_dict(torch.load(args.w_id_disentanglement_path, map_location=args.device))
    
    attr_block = AttrBlock().to(args.device).train()
    attr_block.load_state_dict(torch.load(args.attr_block_path, map_location=args.device))
    
    decoder = Generator(args.stylegan_size, 512, 8).to(args.device).eval()
    decoder.load_state_dict(torch.load(args.stylegan_path, map_location=args.device)['g_ema'], strict=False)
    
    decoder_finetune = Generator(args.stylegan_size, 512, 8).to(args.device).train()
    decoder_finetune.load_state_dict(torch.load(args.finetune_stylegan_path, map_location=args.device))
    
    latents_num=(int(math.log(args.stylegan_size,2))-1)*2
    w_id_mlp = AnonymizationNet(latents_num).to(args.device).train()

    discriminator = Discriminator(args.stylegan_size).to(args.device).train()
    discriminator.load_state_dict(torch.load(args.discriminator_path, map_location=args.device))
    
    for model in [e4e, decoder, lpips_loss, id_encoder, parse_loss]:
        for _, p in model.named_parameters():
            p.requires_grad = False

    trans=transforms.Compose([
                    transforms.Resize((args.stylegan_size, args.stylegan_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    
    train_image_dataset = SimpleDataset(args.train_img_dir,transform=trans)
    test_image_dataset = SimpleDataset(args.test_img_dir,transform=trans)

    train_loader = DataLoader(dataset=train_image_dataset, batch_size=args.batchsize, shuffle=True,drop_last=True)
    test_loader = DataLoader(dataset=test_image_dataset, batch_size=args.test_batchsize, shuffle=False)

    test_loader_data = iter(test_loader)
    images_id,_ = next(test_loader_data)
    test_id_images = images_id.to(args.device)
    best_model_loss = float("inf")

    optimizer_params = [{'params': w_id_mlp.parameters()},
                        {'params': decoder_finetune.parameters()},
                        {'params': attr_block.parameters()},
                        {'params': w_disentanglement.parameters()}]

    optimizer = Ranger(optimizer_params, lr=args.lr)
    d_optim = Ranger(discriminator.parameters(),lr=args.lr)
    step = 0
    smooth_l1_loss = torch.nn.SmoothL1Loss().to(args.device)

    with tqdm(total=args.epoch * len(train_loader)) as pbar:
        for epoch in range(args.epoch):
            for idx, (data,_) in enumerate(train_loader):
                id_images = data.to(args.device)
                
                with torch.no_grad():
                    w_pwd1, w_pwd2 = sample_w(args, decoder, latents_num), sample_w(args, decoder, latents_num)
                    w_pwd1 = w_pwd1.to(args.device)
                    w_pwd2 = w_pwd2.to(args.device)
                    w_rand1,w_rand2 = sample_w(args,decoder,latents_num),sample_w(args,decoder,latents_num)
                    w_rand1 = w_rand1.to(args.device)
                    w_rand2 = w_rand2.to(args.device)

                discriminator.train()
                requires_grad(w_id_mlp, False)
                requires_grad(decoder_finetune, False)
                requires_grad(attr_block, False)
                requires_grad(w_disentanglement, False)
                requires_grad(discriminator, True)

                id_origin_codes,id_transform_codes,c3_coarse,c2_middle,c1_fine = w_disentanglement(id_images,e4e,args.device)
                out_c3, out_c2,out_c1 = attr_block(c3_coarse, c2_middle,c1_fine)

                w_cat1_enc = torch.cat([(id_origin_codes - id_transform_codes), w_pwd1], dim=-1)
                w_cat2_enc = torch.cat([(id_origin_codes - id_transform_codes), w_pwd2], dim=-1)

                w_enc1 = w_id_mlp(w_cat1_enc)
                w_enc2 = w_id_mlp(w_cat2_enc)
            
                w_enc1_new = w_enc1 + id_transform_codes
                w_enc2_new = w_enc2 + id_transform_codes

                out_image_anony1,_ = decoder_finetune([w_enc1_new],input_is_w=True, randomize_noise=False, truncation=1,attr_noise=[out_c3,out_c2,out_c1])
                out_image_anony2, _ = decoder_finetune([w_enc2_new], input_is_w=True, randomize_noise=False, truncation=1, attr_noise=[out_c3, out_c2, out_c1])

                w_cat1_dec_rand = torch.cat([w_enc1, w_rand1], dim=-1)
                w_cat2_dec_rand = torch.cat([w_enc1, w_rand2], dim=-1)
                w_cat1_dec = torch.cat([w_enc1, w_pwd1], dim=-1)

                w_dec_rand1 = w_id_mlp(w_cat1_dec_rand)
                w_dec_rand2 = w_id_mlp(w_cat2_dec_rand)
                w_dec = w_id_mlp(w_cat1_dec)
            
                w_dec1_new = w_dec_rand1 + id_transform_codes
                w_dec2_new = w_dec_rand2 + id_transform_codes
                w_dec_new = w_dec + id_transform_codes

                out_image_recover1,_ = decoder_finetune([w_dec1_new],input_is_w=True, randomize_noise=False, truncation=1,attr_noise=[out_c3,out_c2,out_c1])
                out_image_recover2, _ = decoder_finetune(
                    [w_dec2_new], input_is_w=True, randomize_noise=False, truncation=1, attr_noise=[out_c3, out_c2, out_c1])
                out_image_recover, _ = decoder_finetune(
                    [w_dec_new], input_is_w=True, randomize_noise=False, truncation=1, attr_noise=[out_c3, out_c2, out_c1])

                D_total_loss = torch.tensor(0, dtype=torch.float, device=args.device)
                
                id_images.requires_grad = True
                real_pred = discriminator(id_images)

                d_loss_enc1 = discriminator(out_image_anony1)
                d_loss_enc2 = discriminator(out_image_anony2)
                d_loss_dec = discriminator(out_image_recover)
                d_loss_dec1 = discriminator(out_image_recover1)
                d_loss_dec2 = discriminator(out_image_recover2)

                d_loss_1 = d_logistic_loss(real_pred, d_loss_enc1)
                d_loss_12 = d_logistic_loss(real_pred, d_loss_enc2)
                d_loss_11 = d_logistic_loss(real_pred, d_loss_dec)
                d_loss_2 = d_logistic_loss(real_pred, d_loss_dec1)
                d_loss_21 = d_logistic_loss(real_pred, d_loss_dec2)

                r1_loss = d_r1_loss(real_pred, id_images)
                d_loss = (d_loss_1 + d_loss_12 + d_loss_11 + d_loss_2 + d_loss_21)/5
                r1_loss = args.r1/2 * r1_loss
                D_total_loss = d_loss + r1_loss

                discriminator.zero_grad()
                D_total_loss.backward()
                d_optim.step()

                w_id_mlp.train()
                decoder_finetune.train()
                attr_block.train()
                w_disentanglement.train()

                requires_grad(w_id_mlp, True)
                requires_grad(decoder_finetune, True)
                requires_grad(attr_block, True)
                requires_grad(w_disentanglement, True)

                requires_grad(discriminator, False)

                id_images = data.to(args.device)

                id_origin_codes,id_transform_codes,c3_coarse,c2_middle,c1_fine = w_disentanglement(id_images,e4e,args.device)

                out_c3, out_c2,out_c1 = attr_block(c3_coarse, c2_middle,c1_fine)

                w_cat1_enc = torch.cat([(id_origin_codes - id_transform_codes), w_pwd1], dim=-1)
                w_cat2_enc = torch.cat([(id_origin_codes - id_transform_codes), w_pwd2], dim=-1)

                w_enc1 = w_id_mlp(w_cat1_enc)
                w_enc2 = w_id_mlp(w_cat2_enc)
            
                w_enc1_new = w_enc1 + id_transform_codes
                w_enc2_new = w_enc2 + id_transform_codes

                out_image_anony1,_ = decoder_finetune([w_enc1_new],input_is_w=True, randomize_noise=False, truncation=1,attr_noise=[out_c3,out_c2,out_c1])
                out_image_anony2, _ = decoder_finetune(
                    [w_enc2_new], input_is_w=True, randomize_noise=False, truncation=1, attr_noise=[out_c3, out_c2, out_c1])


                w_cat1_dec_rand = torch.cat([w_enc1, w_rand1], dim=-1)
                w_cat2_dec_rand = torch.cat([w_enc1, w_rand2], dim=-1)
                w_cat1_dec = torch.cat([w_enc1, w_pwd1], dim=-1)

                w_dec_rand1 = w_id_mlp(w_cat1_dec_rand)
                w_dec_rand2 = w_id_mlp(w_cat2_dec_rand)
                w_dec = w_id_mlp(w_cat1_dec)
            
                w_dec1_new = w_dec_rand1 + id_transform_codes
                w_dec2_new = w_dec_rand2 + id_transform_codes
                w_dec_new = w_dec + id_transform_codes

                out_image_recover1,_ = decoder_finetune([w_dec1_new],input_is_w=True, randomize_noise=False, truncation=1,attr_noise=[out_c3,out_c2,out_c1])
                out_image_recover2,_ = decoder_finetune([w_dec2_new],input_is_w=True, randomize_noise=False, truncation=1,attr_noise=[out_c3,out_c2,out_c1])
                out_image_recover, _ = decoder_finetune([w_dec_new], input_is_w=True, randomize_noise=False, truncation=1, attr_noise=[out_c3, out_c2, out_c1])

                
                loss_rev_id_enc = 0
                _, loss_rev_id_enc1, _= id_encoder(id_images,out_image_anony1)
                _, loss_rev_id_enc2, _= id_encoder(id_images,out_image_anony2)
                _, loss_rev_id_enc12, _= id_encoder(out_image_anony1,out_image_anony2)
                loss_rev_id_enc =  (loss_rev_id_enc1 + loss_rev_id_enc2 + loss_rev_id_enc12)/3
                id_loss = loss_rev_id_enc

                loss_rev_id_rev = 0
                _, loss_rev_id_dec1, _= id_encoder(out_image_recover,out_image_recover1)
                _, loss_rev_id_dec2, _= id_encoder(out_image_recover,out_image_recover2)
                _, loss_rev_id_dec12, _= id_encoder(out_image_recover1,out_image_recover2)
                _, loss_rev_id_dec1ori, _ = id_encoder(id_images,out_image_recover1)
                _, loss_rev_id_dec2ori, _ = id_encoder(id_images,out_image_recover2)
                loss_rev_id_rev = (loss_rev_id_dec1 + loss_rev_id_dec2 +loss_rev_id_dec12 + loss_rev_id_dec1ori + loss_rev_id_dec2ori)/5
                id_loss += loss_rev_id_rev
                loss_recon_id, _, _ = id_encoder(id_images,out_image_recover)
                id_loss += loss_recon_id

                outputs = [out_image_anony1, out_image_anony2, out_image_recover, out_image_recover1, out_image_recover2]
                g_loss = sum(g_nonsaturating_loss(discriminator(output)) for output in outputs) / len(outputs)
                rec_loss = args.lambdarec * sum( smooth_l1_loss(id_images, out) for out in outputs )
                lp_loss  = args.lambdalpips * sum( lpips_loss(id_images, out) for out in outputs )
                loss_parse = args.lambdaparse * sum( parse_loss(id_images, out) for out in outputs )
                latent_loss = args.lambdalatent * (sum((w**2).mean() for w in (w_enc1_new, w_enc2_new, w_dec1_new, w_dec2_new, w_dec_new)) / 5)
                G_total_loss = torch.tensor(0, dtype=torch.float, device=args.device)
                G_total_loss = id_loss + rec_loss + lp_loss + latent_loss + loss_parse + g_loss

                optimizer.zero_grad()
                G_total_loss.backward()
                optimizer.step()

                loss_file_path = os.path.join(args.save_dir, 'loss.txt')
                file_save = open(loss_file_path, mode='a')
                file_save.write('epoch/idx:'+str(epoch)+'/'+str(idx)+' G_total_loss:' +
                            str(G_total_loss)+'\n'+'epoch/idx:'+str(epoch)+'/'+str(idx)+' D_total_loss:' +
                            str(D_total_loss)+'\n')
                file_save.close()

                writer.add_scalar('Loss/G_total_loss', G_total_loss,step)
                writer.add_scalar('Loss/id_loss', id_loss, step)
                writer.add_scalar('Loss/rec_loss', rec_loss,step)
                writer.add_scalar('Loss/lp_loss', lp_loss, step)
                writer.add_scalar('Loss/latent_loss', latent_loss,step)
                writer.add_scalar('Loss/loss_parse', loss_parse,step)
                writer.add_scalar('Loss/g_loss', g_loss,step)

                writer.add_scalar('Loss/D_total_loss', D_total_loss,step)
                writer.add_scalar('Loss/d_loss', d_loss,step)
                writer.add_scalar('Loss/r1_loss', r1_loss,step)

                step += 1
                pbar.update(1)

                if idx % 10 == 0  and epoch == 4:
                    with torch.no_grad():
                        w_id_mlp.eval()
                        decoder_finetune.eval()
                        attr_block.eval()
                        discriminator.eval()
                        w_disentanglement.eval()

                        w_pwd1, w_pwd2 = test_sample_w(args, decoder, latents_num), test_sample_w(
                            args, decoder, latents_num)

                        w_rand1, w_rand2 = test_sample_w(args, decoder, latents_num), test_sample_w(
                            args, decoder, latents_num)
                        
                        id_origin_codes,id_transform_codes,c3_coarse,c2_middle,c1_fine = w_disentanglement(test_id_images,e4e,args.device)

                        w_cat1_enc = torch.cat([(id_origin_codes - id_transform_codes), w_pwd1], dim=-1)
                        w_cat2_enc = torch.cat([(id_origin_codes - id_transform_codes), w_pwd2], dim=-1)

                        w_enc1 = w_id_mlp(w_cat1_enc)
                        w_enc2 = w_id_mlp(w_cat2_enc)
                        
                        w_enc1_new = w_enc1 + id_transform_codes
                        w_enc2_new = w_enc2 + id_transform_codes

                        out_c3,out_c2,out_c1 = attr_block(c3_coarse,c2_middle,c1_fine)

                        out_image_anony1,_ = decoder_finetune([w_enc1_new],input_is_w=True, randomize_noise=False, truncation=1,attr_noise=[out_c3,out_c2,out_c1])
                        out_image_anony2, _ = decoder_finetune(
                            [w_enc2_new], input_is_w=True, randomize_noise=False, truncation=1, attr_noise=[out_c3, out_c2, out_c1])

                        w_cat1_dec_rand = torch.cat([w_enc1, w_rand1], dim=-1)
                        w_cat2_dec_rand = torch.cat([w_enc1, w_rand2], dim=-1)
                        w_cat1_dec = torch.cat([w_enc1, w_pwd1], dim=-1)

                        w_dec_rand1 = w_id_mlp(w_cat1_dec_rand)
                        w_dec_rand2 = w_id_mlp(w_cat2_dec_rand)
                        w_dec = w_id_mlp(w_cat1_dec)
                        
                        w_dec1_new = w_dec_rand1 + id_transform_codes
                        w_dec2_new = w_dec_rand2 + id_transform_codes
                        w_dec_new = w_dec + id_transform_codes

                        out_image_recover1,_ = decoder_finetune([w_dec1_new],input_is_w=True, randomize_noise=False, truncation=1,attr_noise=[out_c3,out_c2,out_c1])
                        out_image_recover2,_ = decoder_finetune([w_dec2_new],input_is_w=True, randomize_noise=False, truncation=1,attr_noise=[out_c3,out_c2,out_c1])
                        out_image_recover, _ = decoder_finetune([w_dec_new], input_is_w=True, randomize_noise=False, truncation=1, attr_noise=[out_c3, out_c2, out_c1])

                        loss_rev_id_enc = 0
                        _, loss_rev_id_enc1, _= id_encoder(test_id_images,out_image_anony1)
                        _, loss_rev_id_enc2, _= id_encoder(test_id_images,out_image_anony2)
                        _, loss_rev_id_enc12, _= id_encoder(out_image_anony1,out_image_anony2)
                        loss_rev_id_enc =  (loss_rev_id_enc1 + loss_rev_id_enc2 + loss_rev_id_enc12)/3
                        id_loss = loss_rev_id_enc

                        loss_rev_id_rev = 0
                        _, loss_rev_id_dec1, _= id_encoder(out_image_recover,out_image_recover1)
                        _, loss_rev_id_dec2, _= id_encoder(out_image_recover,out_image_recover2)
                        _, loss_rev_id_dec12, _= id_encoder(out_image_recover1,out_image_recover2)
                        _, loss_rev_id_dec1ori, _ = id_encoder(test_id_images,out_image_recover1)
                        _, loss_rev_id_dec2ori, _ = id_encoder(test_id_images,out_image_recover2)
                        loss_rev_id_rev = (loss_rev_id_dec1 + loss_rev_id_dec2 +loss_rev_id_dec12 + loss_rev_id_dec1ori + loss_rev_id_dec2ori)/5
                        id_loss += loss_rev_id_rev

                        loss_recon_id, _, _ = id_encoder(test_id_images,out_image_recover)
                        id_loss += loss_recon_id

                        outputs = [out_image_anony1, out_image_anony2, out_image_recover, out_image_recover1, out_image_recover2]
                        g_loss = sum(g_nonsaturating_loss(discriminator(output)) for output in outputs) / len(outputs)
                        rec_loss = args.lambdarec * sum( smooth_l1_loss(test_id_images, out) for out in outputs )
                        lp_loss  = args.lambdalpips * sum( lpips_loss(test_id_images, out) for out in outputs )
                        loss_parse = args.lambdaparse * sum( parse_loss(test_id_images, out) for out in outputs )
                        latent_loss = args.lambdalatent * (sum((w**2).mean() for w in (w_enc1_new, w_enc2_new, w_dec1_new, w_dec2_new, w_dec_new)) / 5)
                        G_total_loss = torch.tensor(0, dtype=torch.float, device=args.device)
                        G_total_loss = id_loss + rec_loss + lp_loss + latent_loss + loss_parse + g_loss

                    if G_total_loss < best_model_loss:
                        best_model_loss = G_total_loss
                            
                        with torch.no_grad():

                            best_loss_save_path = os.path.join(args.save_dir, 'best_result', 'best_model.txt')
                            file_save = open(best_loss_save_path, mode='w')
                            file_save.write('epoch/idx:'+str(epoch)+'/'+str(idx)+' G_test_loss:'+str(G_total_loss))
                            file_save.close()

                            save_batch=torch.zeros((args.test_batchsize,3,256,256*6))
                            save_batch[:,:,0:256,0:256]=test_id_images
                            save_batch[:,:,0:256,256:512]=out_image_anony1
                            save_batch[:,:,0:256,512:768]=out_image_anony2
                            save_batch[:,:,0:256,768:1024]=out_image_recover
                            save_batch[:,:,0:256,1024:1280]=out_image_recover1
                            save_batch[:,:,0:256,1280:1536]=out_image_recover2

                            os.makedirs(os.path.join(args.save_dir, 'best_result'), exist_ok=True)
                            for k in range(args.test_batchsize):
                                image_path=os.path.join(args.save_dir, 'best_result', f'anonymization_{k}.png')
                                save_image(save_batch[k], image_path, normalize=True)
                            
                            best_model_base=os.path.join(args.save_dir, 'best_result', 'best_model')
                            torch.save(w_id_mlp.state_dict(), f"{best_model_base}_id_mlp.pth")
                            torch.save(decoder_finetune.state_dict(), f"{best_model_base}_decoder_finetune.pth")
                            torch.save(attr_block.state_dict(), f"{best_model_base}_attr_block.pth")
                            torch.save(discriminator.state_dict(), f"{best_model_base}_discriminator.pth")
                            torch.save(w_disentanglement.state_dict(), f"{best_model_base}_w_disentanglement.pth")
                            
            with torch.no_grad():
                w_id_mlp.eval()
                decoder_finetune.eval()
                attr_block.eval()
                discriminator.eval()
                w_disentanglement.eval()

                w_pwd1, w_pwd2 = test_sample_w(args, decoder, latents_num), test_sample_w(
                    args, decoder, latents_num)

                w_rand1, w_rand2 = test_sample_w(args, decoder, latents_num), test_sample_w(
                    args, decoder, latents_num)

                id_origin_codes,id_transform_codes,c3_coarse,c2_middle,c1_fine = w_disentanglement(test_id_images,e4e,args.device)

                w_cat1_enc = torch.cat([(id_origin_codes - id_transform_codes), w_pwd1], dim=-1)
                w_cat2_enc = torch.cat([(id_origin_codes - id_transform_codes), w_pwd2], dim=-1)

                w_enc1 = w_id_mlp(w_cat1_enc)
                w_enc2 = w_id_mlp(w_cat2_enc)
                
                w_enc1_new = w_enc1 + id_transform_codes
                w_enc2_new = w_enc2 + id_transform_codes

                out_c3,out_c2,out_c1 = attr_block(c3_coarse,c2_middle,c1_fine)

                out_image_anony1,_ = decoder_finetune([w_enc1_new],input_is_w=True, randomize_noise=False, truncation=1,attr_noise=[out_c3,out_c2,out_c1])
                out_image_anony2, _ = decoder_finetune(
                    [w_enc2_new], input_is_w=True, randomize_noise=False, truncation=1, attr_noise=[out_c3, out_c2, out_c1])

                w_cat1_dec_rand = torch.cat([w_enc1, w_rand1], dim=-1)
                w_cat2_dec_rand = torch.cat([w_enc1, w_rand2], dim=-1)
                w_cat1_dec = torch.cat([w_enc1, w_pwd1], dim=-1)

                w_dec_rand1 = w_id_mlp(w_cat1_dec_rand)
                w_dec_rand2 = w_id_mlp(w_cat2_dec_rand)
                w_dec = w_id_mlp(w_cat1_dec)
                
                w_dec1_new = w_dec_rand1 + id_transform_codes
                w_dec2_new = w_dec_rand2 + id_transform_codes
                w_dec_new = w_dec + id_transform_codes

                out_image_recover1,_ = decoder_finetune([w_dec1_new],input_is_w=True, randomize_noise=False, truncation=1,attr_noise=[out_c3,out_c2,out_c1])
                out_image_recover2,_ = decoder_finetune([w_dec2_new],input_is_w=True, randomize_noise=False, truncation=1,attr_noise=[out_c3,out_c2,out_c1])
                out_image_recover, _ = decoder_finetune([w_dec_new], input_is_w=True, randomize_noise=False, truncation=1, attr_noise=[out_c3, out_c2, out_c1])

                save_batch=torch.zeros((args.test_batchsize,3,256,256*6))
                save_batch[:,:,0:256,0:256]=test_id_images
                save_batch[:,:,0:256,256:512]=out_image_anony1
                save_batch[:,:,0:256,512:768]=out_image_anony2
                save_batch[:,:,0:256,768:1024]=out_image_recover
                save_batch[:,:,0:256,1024:1280]=out_image_recover1
                save_batch[:,:,0:256,1280:1536]=out_image_recover2

                for k in range(args.test_batchsize):
                    image_filename = f"epoch{epoch}_anonymization_{k}.png"
                    image_path = os.path.join(args.save_dir, "epoch", image_filename)
                    save_image(save_batch[k], image_path, normalize=True)

                model_prefix = os.path.join(args.save_dir, "epoch", f"epoch_{epoch}_model")
                torch.save(w_id_mlp.state_dict(), f"{model_prefix}_id_mlp.pth")
                torch.save(decoder_finetune.state_dict(), f"{model_prefix}_decoder_finetune.pth")
                torch.save(attr_block.state_dict(), f"{model_prefix}_attr_block.pth")
                torch.save(discriminator.state_dict(), f"{model_prefix}_discriminator.pth")
                torch.save(w_disentanglement.state_dict(), f"{model_prefix}_w_disentanglement.pth")  


