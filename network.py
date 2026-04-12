import os
import re
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import time
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

def photometric_loss(left_img, right_img, disp):
    B, _, H, W = left_img.size()

    curr_grid_y, curr_grid_x = torch.meshgrid(
        torch.linspace(0, 1, H, device=disp.device),
        torch.linspace(0, 1, W, device=disp.device),
        indexing='ij'
    )

    curr_grid_x = curr_grid_x.expand(B, -1, -1)
    curr_grid_y = curr_grid_y.expand(B, -1, -1)

    shifted_grid_x = curr_grid_x - (disp.squeeze(1) / W)

    vgrid_x = 2.0 * shifted_grid_x - 1.0
    vgrid_y = 2.0 * curr_grid_y - 1.0

    vgrid = torch.stack((vgrid_x, vgrid_y), dim=-1)

    reconstructed_left = F.grid_sample(right_img, vgrid, mode='bilinear', padding_mode='border', align_corners=True)
    
    return F.l1_loss(reconstructed_left, left_img)

def readPFM(file):
    with open(file, 'rb') as f:
        header = f.readline().rstrip()
        if header == b'PF': color = True
        elif header == b'Pf': color = False
        else: raise Exception('Not a PFM file.')
        
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', f.readline().decode("ascii"))
        if dim_match: width, height = list(map(int, dim_match.groups()))
        else: raise Exception('Malformed PFM header.')
        
        scale = float(f.readline().decode("ascii").rstrip())
        if scale < 0: endian = '<'; scale = -scale
        else: endian = '>';
        
        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)
        return data.copy(), scale

def flow_to_image_smart(flow_x, flow_y, mask=None, gamma=0.4):
    magnitude = np.sqrt(flow_x**2 + flow_y**2)
    angle = np.arctan2(flow_y, flow_x)

    if mask is not None:
        valid_pixels = mask > 0.5

        if valid_pixels.sum() > 0:
            valid_mags = magnitude[valid_pixels]
            max_val = np.percentile(valid_mags, 95)
        else:
            max_val = 1.0
    else:
        max_val = np.percentile(magnitude, 95)

    if max_val < 1.0: max_val = 1.0

    mag_norm = np.clip(magnitude / max_val, 0, 1)
    mag_norm = mag_norm ** gamma

    h, w = flow_x.shape
    hsv = np.zeros((h, w, 3), dtype=np.float32)

    hsv[..., 0] = (angle / (2 * np.pi)) % 1.0 
    hsv[..., 1] = 1.0
    hsv[..., 2] = mag_norm

    return hsv_to_rgb(hsv)

class DrivingSceneFlowDataset(Dataset):
    def __init__(self, root_dir, resize_wh=(512, 256)):
        self.root_dir = root_dir
        self.resize_w, self.resize_h = resize_wh

        self.disp_dir = os.path.join(root_dir, 'disparity')
        self.disp_change_dir = os.path.join(root_dir, 'disparity_change')
        self.flow_dir = os.path.join(root_dir, 'optical_flow')
        self.img_left_dir = os.path.join(root_dir, 'frames', 'left')
        self.img_right_dir = os.path.join(root_dir, 'frames', 'right')

        if not os.path.exists(self.disp_change_dir):
            print(f"WARNING: Папка {self.disp_change_dir} не найдена! Проверьте путь.")

        files = [f for f in os.listdir(self.img_left_dir) if f.endswith('.png')]
        self.num_samples = len(files) - 1
        print(f"DrivingDataset: {self.num_samples} pairs loaded from {root_dir}")

    def __len__(self): return self.num_samples

    def _load_png(self, path):
        img = Image.open(path).convert('RGB')
        img = img.resize((self.resize_w, self.resize_h), Image.BILINEAR)
        return torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)

    def _load_pfm_target(self, path, is_flow=False):
        data, _ = readPFM(path)
        if len(data.shape) == 2: data = data[:, :, np.newaxis]
        
        tensor = torch.from_numpy(data).permute(2, 0, 1).unsqueeze(0).float()
        tensor = F.interpolate(tensor, size=(self.resize_h, self.resize_w), mode='bilinear', align_corners=False)
        
        scale_x = self.resize_w / data.shape[1]
        scale_y = self.resize_h / data.shape[0]
        
        if is_flow:
            tensor[:, 0, :, :] *= scale_x
            tensor[:, 1, :, :] *= scale_y
        else:
            tensor *= scale_x
        return tensor.squeeze(0)

    def __getitem__(self, idx):
        c_id, n_id = idx + 1, idx + 2
        name_c, name_n = f"{c_id:04d}", f"{n_id:04d}"
        
        try:
            iL_t = self._load_png(os.path.join(self.img_left_dir, f"{name_c}.png"))
            iR_t = self._load_png(os.path.join(self.img_right_dir, f"{name_c}.png"))
            iL_t1 = self._load_png(os.path.join(self.img_left_dir, f"{name_n}.png"))
            iR_t1 = self._load_png(os.path.join(self.img_right_dir, f"{name_n}.png"))

            flow = self._load_pfm_target(os.path.join(self.flow_dir, f"OpticalFlowIntoFuture_{name_c}_L.pfm"), True)[:2]

            disp0 = self._load_pfm_target(os.path.join(self.disp_dir, f"{name_c}.pfm"), False)

            change_path = os.path.join(self.disp_change_dir, f"{name_c}.pfm")
            
            if os.path.exists(change_path):
                disp_change = self._load_pfm_target(change_path, False)
                disp1 = disp0 + disp_change
            else:
                disp1 = self._load_pfm_target(os.path.join(self.disp_dir, f"{name_n}.pfm"), False)

        except Exception as e:
            print(f"Error reading idx {idx}: {e}")
            raise e

        inputs = torch.cat([iL_t, iR_t, iL_t1, iR_t1], dim=0)
        targets = torch.cat([flow, disp0, disp1], dim=0)
        return inputs, targets

class KittiSceneFlowDataset(Dataset):
    def __init__(self, root_dir, indices, resize_wh=(512, 256)):
        self.root_dir = root_dir
        self.indices = indices
        self.resize_w, self.resize_h = resize_wh
        
        self.img2 = os.path.join(root_dir, 'image_2')
        self.img3 = os.path.join(root_dir, 'image_3')
        self.disp0 = os.path.join(root_dir, 'disp_occ_0')
        self.disp1 = os.path.join(root_dir, 'disp_occ_1')
        self.flow = os.path.join(root_dir, 'flow_occ')
        print(f"KittiDataset: {len(indices)} samples.")

    def __len__(self): return len(self.indices)

    def _load_img(self, path):
        img = Image.open(path).convert('RGB')
        img = img.resize((self.resize_w, self.resize_h), Image.BILINEAR)
        return torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)

    def _load_disp_kitti(self, path):
        if not os.path.exists(path):
            return torch.zeros((1, self.resize_h, self.resize_w)), torch.zeros((1, self.resize_h, self.resize_w))
        
        data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if data is None: return torch.zeros((1, self.resize_h, self.resize_w)), torch.zeros((1, self.resize_h, self.resize_w))
        
        h, w = data.shape
        mask = (data > 0).astype(np.float32)
        val = data.astype(np.float32) / 256.0
        
        val_t = torch.from_numpy(val).view(1, 1, h, w)
        mask_t = torch.from_numpy(mask).view(1, 1, h, w)
        
        val_r = F.interpolate(val_t, (self.resize_h, self.resize_w), mode='bilinear', align_corners=False)
        mask_r = F.interpolate(mask_t, (self.resize_h, self.resize_w), mode='nearest')
        
        val_r *= (self.resize_w / w)
        return val_r.squeeze(0), mask_r.squeeze(0)

    def _load_flow_kitti(self, path):
        if not os.path.exists(path):
            return torch.zeros((2, self.resize_h, self.resize_w)), torch.zeros((2, self.resize_h, self.resize_w))
        
        data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if data is None: return torch.zeros((2, self.resize_h, self.resize_w)), torch.zeros((2, self.resize_h, self.resize_w))
        
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        h, w, _ = data.shape
        u, v, valid = data[:,:,0], data[:,:,1], data[:,:,2]
        
        u = (u.astype(np.float32) - 32768.0) / 64.0
        v = (v.astype(np.float32) - 32768.0) / 64.0
        mask = (valid > 0).astype(np.float32)
        
        flow_t = torch.from_numpy(np.stack([u, v], axis=0)).unsqueeze(0)
        mask_t = torch.from_numpy(mask).view(1, 1, h, w)
        mask_t = torch.cat([mask_t, mask_t], dim=1)
        
        flow_r = F.interpolate(flow_t, (self.resize_h, self.resize_w), mode='bilinear', align_corners=False)
        mask_r = F.interpolate(mask_t, (self.resize_h, self.resize_w), mode='nearest')
        
        flow_r[:,0] *= (self.resize_w / w)
        flow_r[:,1] *= (self.resize_h / h)
        
        return flow_r.squeeze(0), mask_r.squeeze(0)

    def __getitem__(self, idx):
        kid = self.indices[idx]
        name_t, name_t1 = f"{kid:06d}_10.png", f"{kid:06d}_11.png"
        
        inputs = torch.cat([
            self._load_img(os.path.join(self.img2, name_t)),
            self._load_img(os.path.join(self.img3, name_t)),
            self._load_img(os.path.join(self.img2, name_t1)),
            self._load_img(os.path.join(self.img3, name_t1))
        ], dim=0)
        
        f, mf = self._load_flow_kitti(os.path.join(self.flow, name_t))
        d0, md0 = self._load_disp_kitti(os.path.join(self.disp0, name_t))
        d1, md1 = self._load_disp_kitti(os.path.join(self.disp1, name_t))
        
        targets = torch.cat([f, d0, d1], dim=0)
        masks = torch.cat([mf, md0, md1], dim=0)
        
        return inputs, targets, masks

class SceneFlowNet(nn.Module):
    def __init__(self):
        super(SceneFlowNet, self).__init__()
        self.enc1 = self.cb(12, 64)
        self.enc2 = self.cb(64, 128)
        self.enc3 = self.cb(128, 256)
        self.enc4 = self.cb(256, 512)
        
        self.bottleneck = self.cb(512, 512)
        
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = self.cb(1024, 256)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = self.cb(512, 128)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = self.cb(256, 64)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = self.cb(128, 32)
        
        self.final_flow = nn.Conv2d(32, 2, kernel_size=1)
        self.final_disp = nn.Conv2d(32, 2, kernel_size=1)

    def cb(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1), nn.BatchNorm2d(out_c), nn.ReLU(True),
            nn.Conv2d(out_c, out_c, 3, 1, 1), nn.BatchNorm2d(out_c), nn.ReLU(True)
        )

    def forward(self, x):
        e1 = self.enc1(x); p1 = self.pool(e1)
        e2 = self.enc2(p1); p2 = self.pool(e2)
        e3 = self.enc3(p2); p3 = self.pool(e3)
        e4 = self.enc4(p3); p4 = self.pool(e4)
        b = self.bottleneck(p4)
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))

        flow = self.final_flow(d1)
        disp = self.final_disp(d1)

        h = flow.shape[2]
        gate = torch.linspace(0.0, 1.0, h, device=x.device).view(1, 1, h, 1)
        gate = torch.clamp(gate * 5.0, 0.0, 1.0) 

        flow = torch.clamp(flow * gate, -512.0, 512.0)
        disp = torch.clamp(F.relu(disp) * gate, 0.0, 256.0)

        return torch.cat([flow, disp], dim=1)

class MaskedL1Loss(nn.Module):
    def forward(self, pred, target, mask):
        diff = torch.abs(pred - target) * mask
        return diff.sum() / (mask.sum() + 1e-6)

class SmoothnessLoss(nn.Module):
    def forward(self, pred, image):
        pred_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        pred_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])

        img_dy = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]), 1, keepdim=True)
        img_dx = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]), 1, keepdim=True)

        weights_x = torch.exp(-img_dx)
        weights_y = torch.exp(-img_dy)

        loss_x = torch.mean(pred_dx * weights_x)
        loss_y = torch.mean(pred_dy * weights_y)
        
        return loss_x + loss_y

def run_universal_training(dataset_name, root_dir, pretrained=None, lr=1e-3, epochs=50, save_pref="model", kitti_idxs=None):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nStart {dataset_name.upper()} on {DEVICE} | LR={lr} | Pretrained={pretrained}")

    if dataset_name == 'driving':
        dataset = DrivingSceneFlowDataset(root_dir)
        use_mask = False
        criterion = nn.L1Loss()
    elif dataset_name == 'kitti':
        if kitti_idxs is None: raise ValueError("Kitti indices required")
        dataset = KittiSceneFlowDataset(root_dir, kitti_idxs)
        use_mask = True
        criterion_mask = MaskedL1Loss()
        criterion_smooth = SmoothnessLoss()
    else: raise ValueError("Unknown dataset")
    
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)

    model = SceneFlowNet().to(DEVICE)
    if pretrained and os.path.exists(pretrained):
        model.load_state_dict(torch.load(pretrained))
        print(f"Loaded weights: {pretrained}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs/2), gamma=0.5)

    model.train()
    for ep in range(1, epochs+1):
        st = time.time()
        loss_sum = 0.0
        
        for batch in loader:
            batch = [b.to(DEVICE) for b in batch]
            optimizer.zero_grad()
            
            if use_mask:
                inp, tgt, msk = batch

                out = model(inp) 
            
                loss_flow = criterion_mask(out[:, :2], tgt[:, :2], msk[:, :2])
                loss_disp = criterion_mask(out[:, 2:], tgt[:, 2:], msk[:, 2:])

                ls_flow = criterion_smooth(out[:, :2], inp[:, :3]) * 0.1 
                ls_disp = criterion_smooth(out[:, 2:], inp[:, :3]) * 1.0

                unm = (1 - msk[:, 2:])
                h = out.shape[2]
                yw = torch.linspace(4.0, 0.0, h, device=DEVICE).view(1, 1, h, 1)
                l_sky = (torch.abs(out[:, 2:]) * unm * yw).mean() * 1.5
                l_ph = photometric_loss(inp[:, :3], inp[:, 3:6], out[:, 2:3]) * 0.2

                loss = (loss_flow + loss_disp) + ls_flow + ls_disp + l_sky + l_ph
            else: 
                inp, tgt = batch
                out = model(inp)
                loss = criterion(out, tgt)
                
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

            if ep % 5 == 0 and loss_sum == loss.item():
                with torch.no_grad():
                    fx = out[0,0].cpu().numpy()
                    fy = out[0,1].cpu().numpy()

                    cur_mask = msk[0,0].cpu().numpy() if use_mask else None

                    rgb_flow = flow_to_image_smart(fx, fy, mask=cur_mask)
                    
                    plt.imsave(f"debug_{save_pref}_ep{ep}.png", rgb_flow)

        print(f"Ep [{ep}/{epochs}] | Loss: {loss_sum/len(loader):.4f} | LR: {scheduler.get_last_lr()[0]:.1e} | T: {time.time()-st:.1f}s")
        scheduler.step()
        
        if ep % 10 == 0:
            torch.save(model.state_dict(), f"{save_pref}_ep{ep}.pth")
            print(f"Saved {save_pref}_ep{ep}.pth")

KITTI_TRAIN_IDXS = [2, 43, 44, 158, 78, 102, 56, 13, 107, 99, 31, 55, 54, 129, 85, 151, 173, 186, 195, 130, 48, 196, 154, 28, 165, 63, 60, 161, 140, 194, 104, 114, 35, 16, 152, 77, 126, 23, 125, 10, 86, 124, 160, 80, 98, 193, 69, 118, 115, 30, 92, 134, 71, 57, 8, 178, 38, 182, 27, 67, 36, 139, 91, 6, 49, 179, 184, 84, 81, 188, 101, 5, 141, 166, 113, 12, 199, 65, 128, 18, 41, 82, 53, 146, 187, 14, 19, 34, 21, 46, 180, 172, 106, 137, 145, 153, 191, 20, 22, 144, 70, 183, 190, 29, 156, 119, 25, 135, 1, 176, 103, 42, 33, 3, 17, 64, 108, 75, 164, 11, 143, 88, 117, 26, 4, 162, 177, 83, 73, 171, 109, 111, 15, 50, 100, 181, 167, 148, 79, 168, 76, 94, 121, 89, 198, 68, 138, 112, 170, 72, 120, 155, 66, 149, 47, 59, 90, 185, 189, 105, 52, 132, 45, 110, 127, 7, 157, 96, 24, 122, 147, 116, 0, 9, 58, 97, 62, 192, 142, 123]

if __name__ == "__main__":
    # Этап 1 — обучение с нуля на датасете Driving
    #run_universal_training(
    #    dataset_name='driving',
    #    root_dir='./driving',
    #    pretrained=None,
    #    lr=1e-3,
    #    epochs=50,
    #    save_pref='driving_new'
    #)

    # Этап 2 — дообучение на датасете Driving Fast
    #run_universal_training(
    #    dataset_name='driving',
    #    root_dir='./driving_fast',
    #    pretrained='driving_new.pth',
    #    lr=1e-5,
    #    epochs=10,
    #    save_pref='driving_driving_fast_new'
    #)

    # Этап 3 — дообучение на KITTI
    run_universal_training(
        dataset_name='kitti',
        root_dir='./kitti',
        pretrained='driving_driving_fast_new.pth',
        lr=1e-4,
        epochs=200,
        save_pref='driving_driving_fast_kitti_new',
        kitti_idxs=KITTI_TRAIN_IDXS
    )