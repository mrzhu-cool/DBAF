
import argparse
from argparse import Namespace
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import os
import math
from models.cbam import CBAM
from models.psp import pSp
from models.stylegan2.model_revise_c3 import Generator,EqualLinear
from models.disentanglement_id_step1 import DisentanglementId
from losses.id_loss_new import IDLossExtractor

to_tensor_transform = transforms.ToTensor()

parser = argparse.ArgumentParser()

# file dir
parser.add_argument("--test_img_dir", type=str, default="./dataset/test")
parser.add_argument("--save_dir", type=str, default="./output/test")
parser.add_argument("--device", type=str, default="cuda:0")

# pretrained weights
parser.add_argument("--e4e_path", type=str, default="./pretrain/e4e_ffhq_encode_256.pt")
parser.add_argument("--stylegan_path", type=str, default="./pretrain/stylegan2-ffhq-256.pt")
parser.add_argument("--id_encoder_path", type=str, default="./pretrain/model_ir_se50.pth")

# pretrained weights in stage 2
parser.add_argument("--finetune_stylegan_path", type=str, default="./output/stage2/best_result/best_model_decoder_finetune.pth")
parser.add_argument("--w_id_disentanglement_path", type=str, default="./output/stage2/best_result/best_model_w_disentanglement.pth")
parser.add_argument("--attr_block_path", type=str, default="./output/stage2/best_result/best_model_attr_block.pth")
parser.add_argument("--mapper_pre_path", type=str, default="./output/stage2/best_result/best_model_id_mlp.pth")


# parser.add_argument("--finetune_stylegan_path", type=str, default="./pretrain_test/best_model_decoder_finetune.pth")
# parser.add_argument("--w_id_disentanglement_path", type=str, default="./pretrain_test/best_model_w_disentanglement.pth")
# parser.add_argument("--attr_block_path", type=str, default="./pretrain_test/best_model_attr_block.pth")
# parser.add_argument("--mapper_pre_path", type=str, default="./pretrain_test/best_model_id_mlp.pth")


# Parameter settings
parser.add_argument("--stylegan_size", type=int, default=256)
parser.add_argument("--test_batchsize", type=int, default=1)
parser.add_argument("--morefc_num", type=int, default=4)
parser.add_argument("--id_cos_margin", type=float, default=0.1)
parser.add_argument('--num_samples', type=int, default=10, help='Number of generated samples per test image')

args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

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
            self.mapper = nn.ModuleList(
                [
                cMLP() for _ in range(3)
                ]
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
    
def test_sample_w(args,decoder):
        z_1 = torch.randn(args.test_batchsize, 512, device=args.device)
        z_2 = torch.randn(args.test_batchsize, 512, device=args.device)
        z_3 = torch.randn(args.test_batchsize, 512, device=args.device)
        w_1 = decoder.style(z_1).unsqueeze(1).repeat(1,4,1)
        w_2 = decoder.style(z_2).unsqueeze(1).repeat(1,4,1)
        w_3 = decoder.style(z_3).unsqueeze(1).repeat(1,6,1)
        w = torch.cat([w_1,w_2,w_3],dim=1)
        return w

if __name__ == '__main__':
    id_encoder= IDLossExtractor(args).to(args.device).eval()
    e4e = load_e4e(args).to(args.device).eval()
    decoder = Generator(
        args.stylegan_size, 512, 8).to(args.device).eval()
    decoder.load_state_dict(torch.load(
        args.stylegan_path, map_location=args.device)['g_ema'], strict=False)
    
    w_disentanglement = DisentanglementId().to(args.device).eval()
    w_disentanglement.load_state_dict(torch.load(
        args.w_id_disentanglement_path, map_location=args.device))
    
    attr_block = AttrBlock().to(args.device).eval()
    attr_block.load_state_dict(torch.load(
        args.attr_block_path, map_location=args.device))
    
    decoder_finetune = Generator(args.stylegan_size, 512, 8).to(args.device).eval()
    decoder_finetune.load_state_dict(torch.load(
        args.finetune_stylegan_path, map_location=args.device))
    
    latents_num=(int(math.log(args.stylegan_size,2))-1)*2
    w_id_mlp = AnonymizationNet(latents_num).to(args.device).eval()
    w_id_mlp.load_state_dict(torch.load(
        args.mapper_pre_path, map_location=args.device))

    trans=transforms.Compose([
                    transforms.Resize((args.stylegan_size, args.stylegan_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    
    test_image_dataset = SimpleDataset(args.test_img_dir,transform=trans)
    test_loader = DataLoader(dataset=test_image_dataset, batch_size=1, shuffle=False)
    step = 0
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as pbar:
            for idx, (data,name) in enumerate(test_loader):
                test_id_images = data.to(args.device)

                num_ = "".join(list(filter(str.isdigit, name[0])))
                w_pwds = [test_sample_w(args, decoder) for _ in range(args.num_samples)]

                id_origin_codes,id_transform_codes,c3_coarse,c2_middle,c1_fine = w_disentanglement(test_id_images,e4e,args.device)
                attr_noises = attr_block(c3_coarse, c2_middle, c1_fine)

                for i, w_pwd in enumerate(w_pwds, start=1):
                    w_cat_enc = torch.cat([(id_origin_codes - id_transform_codes), w_pwd], dim=-1)
                    w_enc = w_id_mlp(w_cat_enc)
                    w_enc_new = w_enc + id_transform_codes
                    out_image, _ = decoder_finetune(
                        [w_enc_new], 
                        input_is_w=True, 
                        randomize_noise=False, 
                        truncation=1, 
                        attr_noise=attr_noises
                    )
                    save_image(
                        out_image, 
                        os.path.join(args.save_dir, f'img{num_}_anony{i}.jpg'), 
                        normalize=True
                    )
                pbar.update(1)