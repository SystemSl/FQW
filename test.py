import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from network import SceneFlowNet 

def flow_to_image(flow_x, flow_y, gamma=0.4):
    magnitude = np.sqrt(flow_x**2 + flow_y**2)
    angle = np.arctan2(flow_y, flow_x)
    max_val = np.percentile(magnitude, 95)
    if max_val < 1e-5: max_val = 1.0
    mag_norm = np.clip(magnitude / max_val, 0, 1) ** gamma
    h, w = flow_x.shape
    hsv = np.zeros((h, w, 3), dtype=np.float32)
    hsv[..., 0] = (angle / (2 * np.pi)) % 1.0
    hsv[..., 1] = 1.0
    hsv[..., 2] = mag_norm
    return hsv_to_rgb(hsv)

def save_kitti_data(output_tensor, original_wh, save_dir, prefix):
    orig_w, orig_h = original_wh
    output_resized = F.interpolate(output_tensor, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
    data = output_resized[0].cpu().numpy()
    
    flow_x, flow_y = data[0], data[1]
    disp0, disp1   = data[2], data[3]
    
    scale_x = orig_w / 512.0
    scale_y = orig_h / 256.0
    
    flow_x *= scale_x
    flow_y *= scale_y
    disp0  *= scale_x
    disp1  *= scale_x

    cv2.imwrite(os.path.join(save_dir, f"{prefix}_disp_0.png"), (disp0 * 256.0).astype(np.uint16))
    cv2.imwrite(os.path.join(save_dir, f"{prefix}_disp_1.png"), (disp1 * 256.0).astype(np.uint16))

    u_u16 = np.clip((flow_x * 64.0) + 32768.0, 0, 65535).astype(np.uint16)
    v_u16 = np.clip((flow_y * 64.0) + 32768.0, 0, 65535).astype(np.uint16)
    valid_u16 = np.ones_like(u_u16, dtype=np.uint16)

    flow_data_img = cv2.merge([valid_u16, v_u16, u_u16])
    cv2.imwrite(os.path.join(save_dir, f"{prefix}_flow_data.png"), flow_data_img)

def save_visualization(flow_x, flow_y, save_path):
    rgb_flow = flow_to_image(flow_x, flow_y)
    bgr_flow = cv2.cvtColor(rgb_flow, cv2.COLOR_RGB2BGR)
    bgr_flow = (bgr_flow * 255.0).astype(np.uint8)
    
    cv2.imwrite(save_path, bgr_flow)
    print(f"Vis saved: {save_path}")

def run_inference(left_img, right_img, left_next, right_next, weights_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SceneFlowNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    with Image.open(left_img) as tmp: w, h = tmp.size

    def prep(p):
        img = Image.open(p).convert('RGB').resize((512, 256), Image.BILINEAR)
        return torch.from_numpy(np.array(img)/255.0).permute(2,0,1).unsqueeze(0).float().to(device)

    inp = torch.cat([prep(left_img), prep(right_img), prep(left_next), prep(right_next)], 1)
    
    with torch.no_grad():
        out = model(inp)
        
    os.makedirs("results", exist_ok=True)
    save_kitti_data(out, (w, h), "results", "kitti_res")
    fx = out[0, 0].cpu().numpy()
    fy = out[0, 1].cpu().numpy()
    save_visualization(fx, fy, "results/kitti_res_vis.png")

if __name__ == "__main__":
    run_inference("1L.png", "1R.png", "2L.png", "2R.png", "driving_driving_fast_kitti.pth")