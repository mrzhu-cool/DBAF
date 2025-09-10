import math
import os
import argparse
from argparse import Namespace
import torch
from torch import nn
from torch import autograd
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
from models.stylegan2.op import conv2d_gradfix
from models.stylegan2.model_revise_c3 import Generator, Discriminator
from models.ranger import Ranger
from models.psp import pSp
from models.cbam import CBAM
from models.disentanglement_id_step1_1 import DisentanglementId
from losses.id_loss_new import IDLossExtractor
from losses.lpips.lpips import LPIPS
from losses.parse_related_loss import bg_loss

to_tensor_transform = transforms.ToTensor()
parser = argparse.ArgumentParser()

# file dir
parser.add_argument("--train_img_dir", type=str, default="./dataset/train")
parser.add_argument("--test_img_dir", type=str, default="./dataset/test")
parser.add_argument("--save_dir", type=str, default="./output/stage1")
parser.add_argument("--device", type=str, default="cuda:0")

# pretrained weights
parser.add_argument("--e4e_path", type=str, default="./pretrain/e4e_ffhq_encode_256.pt")
parser.add_argument("--stylegan_path", type=str, default="./pretrain/stylegan2-ffhq-256.pt")
parser.add_argument("--id_encoder_path", type=str, default="./pretrain/model_ir_se50.pth")
parser.add_argument("--parsenet_weights", type=str, default="./pretrain/parsenet.pth")

# Parameter settings
parser.add_argument("--stylegan_size", type=int, default=256)
parser.add_argument("--batchsize", type=int, default=16)
parser.add_argument("--test_batchsize", type=int, default=16)
parser.add_argument("--validate_every", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--epoch", type=int, default=5)
parser.add_argument("--lambdaid", type=float, default=0.5)
parser.add_argument("--lambdarec", type=float, default=3.5)
parser.add_argument("--lambdalpips", type=float, default=1.0)
parser.add_argument("--lambdalatent", type=float, default=0.1)
parser.add_argument("--lambdaparse", type=float, default=0.1)
parser.add_argument("--id_cos_margin", type=float, default=0.0)
parser.add_argument("--lambdacontrastive", type=float, default=0.1)
parser.add_argument("--r1", type=float, default=10.0)
parser.add_argument("--cycle_num", type=int, default=3)
parser.add_argument("--same_thres", type=float, default=0.0)
parser.add_argument("--not_same_thres", type=float, default=0.5)

args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
best_result_dir = os.path.join(args.save_dir, "best_result")
os.makedirs(best_result_dir, exist_ok=True)
epoch_dir = os.path.join(args.save_dir, "epoch")
os.makedirs(epoch_dir, exist_ok=True)
writer = SummaryWriter(args.save_dir)


def cycle_images_to_create_different_order(images,device):
        batch_size = len(images)
        different_images = torch.empty_like(images, device=device)
        different_images[0] = images[batch_size - 1]
        different_images[1:] = images[:batch_size - 1]
        return different_images


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
        #make sure path has only images
        print('Image Path: ', path)
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

    def forward(self, input_c3, input_c2,input_c1):
        out_c3 = self.conv_c3(input_c3)
        out_c2 = self.conv_c2(input_c2)
        out_c1 = self.conv_c1(input_c1)

        out_c3 = self.cbam_c3(out_c3)
        out_c2 = self.cbam_c2(out_c2)
        out_c1 = self.cbam_c1(out_c1)

        return out_c3, out_c2,out_c1
    
def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(
        grad_real.shape[0], -1).sum(1).mean()
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
    decoder = Generator(args.stylegan_size, 512, 8).to(args.device).train()
    decoder.load_state_dict(torch.load(
        args.stylegan_path, map_location=args.device)['g_ema'], strict=False)
    w_disentanglement = DisentanglementId().to(args.device).train()
    attr_block = AttrBlock().to(args.device).train()
    discriminator = Discriminator(args.stylegan_size).to(args.device).train()
    discriminator.load_state_dict(torch.load(
        args.stylegan_path, map_location=args.device)['d'], strict=False)
    
    for model in [id_encoder, lpips_loss, parse_loss, e4e]:
        for _, p in model.named_parameters():
            p.requires_grad = False

    latents_num=(int(math.log(args.stylegan_size,2))-1)*2
    
    trans=transforms.Compose([
                    transforms.Resize((args.stylegan_size, args.stylegan_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    
    train_image_dataset = SimpleDataset(args.train_img_dir,transform=trans)
    test_image_dataset = SimpleDataset(args.test_img_dir,transform=trans)

    train_loader = DataLoader(dataset=train_image_dataset, batch_size=args.batchsize, shuffle=True)
    test_loader = DataLoader(dataset=test_image_dataset, batch_size=args.test_batchsize, shuffle=False)
    test_loader_data = iter(test_loader)
    images_id,_ = next(test_loader_data)
    images_attr = cycle_images_to_create_different_order(images_id,args.device)
    test_id_images = images_id.to(args.device)
    test_attr_images = images_attr.to(args.device)

    best_model_loss = float("inf")

    optimizer_params = [{'params': attr_block.parameters()},
                        {'params': decoder.parameters()},
                        {'params': w_disentanglement.parameters()}]

    optimizer = Ranger(optimizer_params, lr=args.lr)

    d_optim = Ranger(discriminator.parameters(), lr=args.lr)

    smooth_l1_loss = torch.nn.SmoothL1Loss().to(args.device)

    step = 0

    with tqdm(total=args.epoch * len(train_loader)) as pbar:
        for epoch in range(args.epoch):
            for idx, (data,_) in enumerate(train_loader):
                id_images = data.to(args.device)
                if idx % args.cycle_num == 0:
                        attr_images = id_images.to(args.device)
                else:
                        attr_images = cycle_images_to_create_different_order(id_images,args.device).to(args.device)

                w_disentanglement.zero_grad()
                attr_block.zero_grad()
                decoder.zero_grad()
                discriminator.zero_grad()

                discriminator.train()
                requires_grad(w_disentanglement, False)
                requires_grad(decoder, False)
                requires_grad(attr_block, False)
                requires_grad(discriminator, True)

                id_codes_origin,id_w_out,_,_,_ = w_disentanglement(id_images,e4e,args.device)
                attr_codes_origin,attr_w_out,c3_coarse, c2_middle,c1_fine = w_disentanglement(attr_images,e4e,args.device)

                out_c3, out_c2,out_c1 = attr_block(c3_coarse, c2_middle,c1_fine)

                w_new = (id_codes_origin - id_w_out) + attr_w_out

                out_images,_ = decoder([w_new],input_is_w=True, randomize_noise=False, truncation=1,attr_noise=[out_c3, out_c2,out_c1])

                D_total_loss = torch.tensor(0, dtype=torch.float, device=args.device)
                attr_images.requires_grad = True
                real_pred = discriminator(attr_images)

                d_loss_enc = discriminator(out_images)
                d_loss = d_logistic_loss(real_pred, d_loss_enc)
                r1_loss = d_r1_loss(real_pred, attr_images)
                r1_loss = args.r1/2 * r1_loss
                D_total_loss = d_loss + r1_loss

                d_optim.zero_grad()
                D_total_loss.backward()
                d_optim.step()

                #G
                w_disentanglement.train()
                attr_block.train()
                decoder.train()

                w_disentanglement.zero_grad()
                attr_block.zero_grad()
                decoder.zero_grad()
                discriminator.zero_grad()

                requires_grad(w_disentanglement, True)
                requires_grad(decoder, True)
                requires_grad(attr_block, True)
                requires_grad(discriminator, False)

                if idx % args.cycle_num == 0:
                    attr_images = id_images.to(args.device)
                else:
                    attr_images = cycle_images_to_create_different_order(
                        id_images, args.device).to(args.device)
                    
                id_codes_origin, id_w_out, _, _, _ = w_disentanglement(
                    id_images, e4e, args.device)
                attr_codes_origin, attr_w_out, c3_coarse, c2_middle, c1_fine = w_disentanglement(
                    attr_images, e4e, args.device)

                out_c3, out_c2, out_c1 = attr_block(
                    c3_coarse, c2_middle, c1_fine)

                w_new = (id_codes_origin - id_w_out) + attr_w_out

                out_images, _ = decoder([w_new], input_is_w=True, randomize_noise=False, truncation=1, attr_noise=[
                                        out_c3, out_c2, out_c1])

                G_total_loss = torch.tensor(0, dtype=torch.float, device=args.device)
                g_loss = g_nonsaturating_loss(
                    discriminator(out_images))

                dists = id_encoder.calculate_contrastive_loss(attr_images,out_images)

                if idx % args.cycle_num == 0:
                    contrastive_loss = torch.clamp(dists - args.same_thres,min=0.).mean()
                else:
                    contrastive_loss = torch.clamp(args.not_same_thres - dists,min=0.).mean()

                contrastive_loss = args.lambdacontrastive * contrastive_loss
                id_loss,_,_ =  id_encoder(id_images,out_images)
                id_loss = args.lambdaid * id_loss

                rec_loss = args.lambdarec * \
                    smooth_l1_loss(attr_images, out_images)
                lp_loss = args.lambdalpips * lpips_loss(attr_images,out_images)
                loss_parse = args.lambdaparse * parse_loss(attr_images,out_images)
                latent_loss =  (w_new **2).mean()
                latent_loss = args.lambdalatent * latent_loss
                G_total_loss = id_loss + rec_loss + lp_loss + latent_loss + loss_parse + contrastive_loss + g_loss

                optimizer.zero_grad()
                G_total_loss.backward()
                optimizer.step()

                loss_save = os.path.join(args.save_dir, "loss.txt")
                file_save = open(loss_save, mode='a')
                file_save.write('epoch/idx:'+str(epoch)+'/'+str(idx)+' G_total_loss:' +
                                str(G_total_loss)+'\n'+'epoch/idx:'+str(epoch)+'/'+str(idx)+' D_total_loss:' +
                                str(D_total_loss)+'\n')
                file_save.close()

                if step % 10 == 0:
                    writer.add_scalar('Loss/G_total_loss', G_total_loss, step)
                    writer.add_scalar('Loss/id_loss', id_loss, step)
                    writer.add_scalar('Loss/rec_loss', rec_loss, step)
                    writer.add_scalar('Loss/lp_loss', lp_loss, step)
                    writer.add_scalar('Loss/latent_loss', latent_loss, step)
                    writer.add_scalar('Loss/loss_parse', loss_parse,step)
                    writer.add_scalar('Loss/contrastive_loss', contrastive_loss, step)
                    writer.add_scalar('Loss/g_loss', g_loss, step)
                    writer.add_scalar('Loss/D_total_loss', D_total_loss, step)
                    writer.add_scalar('Loss/d_loss', d_loss, step)
                    writer.add_scalar('Loss/r1_loss', r1_loss, step)

                step += 1
                pbar.update(1)

                if idx % args.validate_every == 0:
                    test_id_images, _ = next(test_loader_data)
                    test_attr_images = cycle_images_to_create_different_order(test_id_images, args.device)
                    with torch.no_grad():
                        w_disentanglement.eval()
                        attr_block.eval()
                        decoder.eval()

                        G_total_loss = torch.tensor(0, dtype=torch.float, device=args.device)

                        id_codes_origin,id_w_out,_,_,_ = w_disentanglement(test_id_images,e4e,args.device)
                        attr_codes_origin,attr_w_out,c3_coarse,c2_middle,c1_fine = w_disentanglement(test_attr_images,e4e,args.device)

                        out_c3, out_c2,out_c1 = attr_block(c3_coarse, c2_middle,c1_fine)

                        w_new = (id_codes_origin - id_w_out) + attr_w_out

                        out_images,_ = decoder([w_new],input_is_w=True, randomize_noise=False, truncation=1,attr_noise=[out_c3, out_c2,out_c1])

                        g_loss = g_nonsaturating_loss(discriminator(out_images))
                        dists = id_encoder.calculate_contrastive_loss(test_attr_images,out_images)
                        contrastive_loss = torch.clamp(args.not_same_thres - dists,min=0.).mean()
                        contrastive_loss = args.lambdacontrastive * contrastive_loss
                        id_loss,_,_ =  id_encoder(test_id_images,out_images)
                        id_loss = args.lambdaid * id_loss
                        rec_loss = args.lambdarec * smooth_l1_loss(test_attr_images, out_images)
                        lp_loss = args.lambdalpips * lpips_loss(test_attr_images,out_images)
                        loss_parse = args.lambdaparse * parse_loss(test_attr_images,out_images)
                        latent_loss =  (w_new **2).mean()
                        latent_loss = args.lambdalatent * latent_loss
                        G_total_loss = id_loss + rec_loss + lp_loss + latent_loss + loss_parse + contrastive_loss + g_loss
                        
                    if G_total_loss < best_model_loss:
                        best_model_loss = G_total_loss
                            
                        with torch.no_grad():
                            save_batch=torch.zeros((args.test_batchsize,3,256,256*3))
                            save_batch[:,:,0:256,0:256]=test_id_images
                            save_batch[:,:,0:256,256:512]=test_attr_images
                            save_batch[:,:,0:256,512:768]=out_images

                            best_loss_save_path = os.path.join(args.save_dir, 'best_result', 'best_model.txt')
                            file_save = open(best_loss_save_path, mode='w')
                            file_save.write('epoch/idx:'+str(epoch)+'/'+str(idx)+' test_total_loss:'+str(G_total_loss))
                            file_save.close()

                            for k in range(args.test_batchsize):
                                image_path = os.path.join(args.save_dir, 'best_result', f'disentanglement{k}.png')
                                save_image(save_batch[k], image_path, normalize=True)

                            w_dis_path = os.path.join(args.save_dir, 'best_result', 'w_disentanglement_best_model.pth')
                            attr_block_path = os.path.join(args.save_dir, 'best_result', 'attr_block_best_model.pth')
                            decoder_path = os.path.join(args.save_dir, 'best_result', 'finetune_decoder_best_model.pth')
                            discri_path = os.path.join(args.save_dir, 'best_result', 'discriminator_best_model.pth')
                            torch.save(w_disentanglement.state_dict(), w_dis_path)
                            torch.save(attr_block.state_dict(), attr_block_path)
                            torch.save(decoder.state_dict(), decoder_path)
                            torch.save(discriminator.state_dict(), discri_path)

            with torch.no_grad():
                w_disentanglement.eval()
                attr_block.eval()
                decoder.eval()

                id_codes_origin,id_w_out,_,_,_ = w_disentanglement(test_id_images,e4e,args.device)
                attr_codes_origin,attr_w_out,c3_coarse,c2_middle,c1_fine = w_disentanglement(test_attr_images,e4e,args.device)
                out_c3, out_c2,out_c1 = attr_block(c3_coarse, c2_middle,c1_fine)
                w_new = (id_codes_origin - id_w_out) + attr_w_out
                out_images,_ = decoder([w_new],input_is_w=True, randomize_noise=False, truncation=1,attr_noise=[out_c3, out_c2,out_c1])

                save_batch=torch.zeros((args.test_batchsize,3,256,256*3))
                save_batch[:,:,0:256,0:256]=test_id_images
                save_batch[:,:,0:256,256:512]=test_attr_images
                save_batch[:,:,0:256,512:768]=out_images

                for k in range(args.test_batchsize):
                    image_path = os.path.join(args.save_dir, 'epoch', f'disentanglement_epoch_{epoch}_sample_{k}.png')
                    save_image(save_batch[k], image_path, normalize=True)

                model_prefix = os.path.join(args.save_dir, 'epoch', f'epoch_{epoch}_model')
                torch.save(w_disentanglement.state_dict(), f'{model_prefix}.pth')
                torch.save(attr_block.state_dict(), f'{model_prefix}_attr_block.pth')
                torch.save(decoder.state_dict(), f'{model_prefix}_finetune_decoder.pth')
                torch.save(discriminator.state_dict(), f'{model_prefix}_discriminator.pth')
