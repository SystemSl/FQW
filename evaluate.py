import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image

from network import SceneFlowNet 

def load_kitti_disp_gt(path):
    if not os.path.exists(path): return None, None
    data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if data is None: return None, None
    val = data.astype(np.float32) / 256.0
    mask = (data > 0)
    return val, mask

def load_kitti_flow_gt(path):
    if not os.path.exists(path): return None, None, None
    data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if data is None: return None, None, None

    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    
    u_raw = data[:, :, 0]
    v_raw = data[:, :, 1]
    valid_raw = data[:, :, 2]
    
    u = (u_raw.astype(np.float32) - 32768.0) / 64.0
    v = (v_raw.astype(np.float32) - 32768.0) / 64.0
    mask = (valid_raw > 0)
    
    return u, v, mask

def compute_epe(pred, gt, mask):
    if gt is None or mask is None or mask.sum() == 0: return 0.0, 0
    diff = pred - gt
    if len(diff.shape) == 3:
        error = np.sqrt(diff[0]**2 + diff[1]**2)
    else:
        error = np.abs(diff)
    valid_error = error[mask]
    return valid_error.sum(), len(valid_error)

def save_error_heatmap(error_map, mask, filename, max_err=30.0):

    err_clipped = np.clip(error_map, 0, max_err)

    err_norm = (err_clipped / max_err * 255.0).astype(np.uint8)

    heatmap = cv2.applyColorMap(err_norm, cv2.COLORMAP_JET)

    if mask is not None:
        heatmap[~mask] = 0
        
    cv2.imwrite(filename, heatmap)

def evaluate_and_save(data_root, weights_path, output_dir="results"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting evaluation on {device}...")
    
    model = SceneFlowNet().to(device)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("Weights loaded successfully.")
    else:
        print(f"Error: Weights {weights_path} not found.")
        return
    model.eval()

    dir_img2 = os.path.join(data_root, "image_2") 
    dir_img3 = os.path.join(data_root, "image_3") 
    dir_gt_d0 = os.path.join(data_root, "disp_occ_0")
    dir_gt_d1 = os.path.join(data_root, "disp_occ_1")
    dir_gt_flow = os.path.join(data_root, "flow_occ")

    save_disp0 = os.path.join(output_dir, "disp_0")
    save_disp1 = os.path.join(output_dir, "disp_1")
    save_flow = os.path.join(output_dir, "flow")

    save_hm_disp0 = os.path.join(output_dir, "heatmap_disp_0")
    save_hm_disp1 = os.path.join(output_dir, "heatmap_disp_1")
    save_hm_flow = os.path.join(output_dir, "heatmap_flow")
    
    os.makedirs(save_disp0, exist_ok=True)
    os.makedirs(save_disp1, exist_ok=True)
    os.makedirs(save_flow, exist_ok=True)
    os.makedirs(save_hm_disp0, exist_ok=True)
    os.makedirs(save_hm_disp1, exist_ok=True)
    os.makedirs(save_hm_flow, exist_ok=True)
    
    files = [f for f in os.listdir(dir_img2) if f.endswith("_10.png")]
    files.sort()

    stats = {
        'flow': [0, 0], 
        'd0':   [0, 0], 
        'd1':   [0, 0],
        'sf':   [0, 0]
    } 
    
    print(f"Found {len(files)} files. Processing...")
    
    for fname in files:
        name_t = fname
        name_t1 = fname.replace("_10.png", "_11.png")
        
        path_L_t = os.path.join(dir_img2, name_t)
        path_R_t = os.path.join(dir_img3, name_t)
        path_L_t1 = os.path.join(dir_img2, name_t1)
        path_R_t1 = os.path.join(dir_img3, name_t1)

        with Image.open(path_L_t) as tmp: orig_w, orig_h = tmp.size
        
        def prep(p):
            img = Image.open(p).convert('RGB').resize((512, 256), Image.BILINEAR)
            return torch.from_numpy(np.array(img)/255.0).permute(2,0,1).unsqueeze(0).float().to(device)

        inputs = torch.cat([prep(path_L_t), prep(path_R_t), prep(path_L_t1), prep(path_R_t1)], dim=1)
        
        with torch.no_grad():
            output = model(inputs)
            
        output_resized = F.interpolate(output, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        data = output_resized[0].cpu().numpy()
        
        pred_flow_x = data[0] * (orig_w / 512.0)
        pred_flow_y = data[1] * (orig_h / 256.0)
        pred_d0 = data[2] * (orig_w / 512.0)
        pred_d1 = data[3] * (orig_w / 512.0)

        gt_u, gt_v, gt_m_flow = load_kitti_flow_gt(os.path.join(dir_gt_flow, name_t))
        gt_d0, gt_m_d0 = load_kitti_disp_gt(os.path.join(dir_gt_d0, name_t))
        gt_d1, gt_m_d1 = load_kitti_disp_gt(os.path.join(dir_gt_d1, name_t))

        if gt_u is not None:
            es, cnt = compute_epe(np.stack([pred_flow_x, pred_flow_y]), np.stack([gt_u, gt_v]), gt_m_flow)
            stats['flow'][0] += es; stats['flow'][1] += cnt

            diff_u = pred_flow_x - gt_u
            diff_v = pred_flow_y - gt_v
            flow_err_map = np.sqrt(diff_u**2 + diff_v**2)
            save_error_heatmap(flow_err_map, gt_m_flow, os.path.join(save_hm_flow, name_t))

        if gt_d0 is not None:
            es, cnt = compute_epe(pred_d0, gt_d0, gt_m_d0)
            stats['d0'][0] += es; stats['d0'][1] += cnt
            
            d0_err_map = np.abs(pred_d0 - gt_d0)
            save_error_heatmap(d0_err_map, gt_m_d0, os.path.join(save_hm_disp0, name_t))

        if gt_d1 is not None:
            es, cnt = compute_epe(pred_d1, gt_d1, gt_m_d1)
            stats['d1'][0] += es; stats['d1'][1] += cnt
            
            d1_err_map = np.abs(pred_d1 - gt_d1)
            save_error_heatmap(d1_err_map, gt_m_d1, os.path.join(save_hm_disp1, name_t))

        if (gt_u is not None) and (gt_d0 is not None) and (gt_d1 is not None):
            mask_sf = gt_m_flow & gt_m_d0 & gt_m_d1
            
            if mask_sf.sum() > 0:
                diff_u = (pred_flow_x - gt_u)[mask_sf]
                diff_v = (pred_flow_y - gt_v)[mask_sf]

                pred_change = pred_d1 - pred_d0
                gt_change = gt_d1 - gt_d0
                diff_change = (pred_change - gt_change)[mask_sf]

                sf_error = np.sqrt(diff_u**2 + diff_v**2 + diff_change**2)
                
                stats['sf'][0] += sf_error.sum()
                stats['sf'][1] += len(sf_error)

        cv2.imwrite(os.path.join(save_disp0, name_t), (pred_d0 * 256.0).astype(np.uint16))
        cv2.imwrite(os.path.join(save_disp1, name_t), (pred_d1 * 256.0).astype(np.uint16))
        
        u_16 = np.clip(pred_flow_x * 64.0 + 32768.0, 0, 65535).astype(np.uint16)
        v_16 = np.clip(pred_flow_y * 64.0 + 32768.0, 0, 65535).astype(np.uint16)
        valid_16 = np.ones_like(u_16)

        out_flow = cv2.merge([valid_16, v_16, u_16]) 
        cv2.imwrite(os.path.join(save_flow, name_t), out_flow)
        
        print(f"Processed {name_t}")

    print("\n" + "="*40)
    print(" KITTI EVALUATION RESULTS (EPE)")
    print("="*40)
    if stats['d0'][1]: print(f"Disparity (t)   : {stats['d0'][0]/stats['d0'][1]:.4f} px")
    if stats['d1'][1]: print(f"Disparity (t+1) : {stats['d1'][0]/stats['d1'][1]:.4f} px")
    if stats['flow'][1]: print(f"Optical Flow    : {stats['flow'][0]/stats['flow'][1]:.4f} px")
    print("-" * 40)
    if stats['sf'][1]: print(f"SCENE FLOW (3D) : {stats['sf'][0]/stats['sf'][1]:.4f} px")
    print("="*40)

if __name__ == "__main__":
    evaluate_and_save("test", "driving_driving_fast_kitti.pth")