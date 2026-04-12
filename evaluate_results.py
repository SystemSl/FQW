import os
import cv2
import numpy as np

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

def evaluate_folders(gt_root, pred_root, output_dir="results_eval"):
    dir_gt_d0 = os.path.join(gt_root, "disp_occ_0")
    dir_gt_d1 = os.path.join(gt_root, "disp_occ_1")
    dir_gt_flow = os.path.join(gt_root, "flow_occ")

    dir_pred_d0 = os.path.join(pred_root, "disp_0")
    dir_pred_d1 = os.path.join(pred_root, "disp_1")
    dir_pred_flow = os.path.join(pred_root, "flow")

    save_hm_disp0 = os.path.join(output_dir, "heatmap_disp_0")
    save_hm_disp1 = os.path.join(output_dir, "heatmap_disp_1")
    save_hm_flow = os.path.join(output_dir, "heatmap_flow")
    save_hm_dc = os.path.join(output_dir, "heatmap_disp_change")
    
    for d in [save_hm_disp0, save_hm_disp1, save_hm_flow, save_hm_dc]:
        os.makedirs(d, exist_ok=True)
    
    files = [f for f in os.listdir(dir_gt_d0) if f.endswith("_10.png")]
    files.sort()

    stats = {
        'flow': [0, 0], 
        'd0':   [0, 0], 
        'd1':   [0, 0],
        'dc':   [0, 0],
        'sf':   [0, 0]
    } 
    
    print(f"Starting evaluation... Found {len(files)} files.")
    
    for fname in files:
        gt_u, gt_v, gt_m_flow = load_kitti_flow_gt(os.path.join(dir_gt_flow, fname))
        gt_d0, gt_m_d0 = load_kitti_disp_gt(os.path.join(dir_gt_d0, fname))
        gt_d1, gt_m_d1 = load_kitti_disp_gt(os.path.join(dir_gt_d1, fname))

        pred_d0, _ = load_kitti_disp_gt(os.path.join(dir_pred_d0, fname))
        pred_d1, _ = load_kitti_disp_gt(os.path.join(dir_pred_d1, fname))
        pred_u, pred_v, _ = load_kitti_flow_gt(os.path.join(dir_pred_flow, fname))

        if pred_d0 is None or pred_d1 is None or pred_u is None:
            print(f"Skipping {fname}: Results not found in pred_root.")
            continue

        if gt_d0 is not None:
            es, cnt = compute_epe(pred_d0, gt_d0, gt_m_d0)
            stats['d0'][0] += es; stats['d0'][1] += cnt
            save_error_heatmap(np.abs(pred_d0 - gt_d0), gt_m_d0, os.path.join(save_hm_disp0, fname))

        if gt_d1 is not None:
            es, cnt = compute_epe(pred_d1, gt_d1, gt_m_d1)
            stats['d1'][0] += es; stats['d1'][1] += cnt
            save_error_heatmap(np.abs(pred_d1 - gt_d1), gt_m_d1, os.path.join(save_hm_disp1, fname))

        if gt_u is not None:
            es, cnt = compute_epe(np.stack([pred_u, pred_v]), np.stack([gt_u, gt_v]), gt_m_flow)
            stats['flow'][0] += es; stats['flow'][1] += cnt
            flow_err_map = np.sqrt((pred_u - gt_u)**2 + (pred_v - gt_v)**2)
            save_error_heatmap(flow_err_map, gt_m_flow, os.path.join(save_hm_flow, fname))

        if gt_d0 is not None and gt_d1 is not None:
            mask_dc = gt_m_d0 & gt_m_d1
            if mask_dc.sum() > 0:
                gt_change = gt_d1 - gt_d0
                pred_change = pred_d1 - pred_d0
                dc_err_map = np.abs(pred_change - gt_change)
                stats['dc'][0] += dc_err_map[mask_dc].sum()
                stats['dc'][1] += mask_dc.sum()
                save_error_heatmap(dc_err_map, mask_dc, os.path.join(save_hm_dc, fname), max_err=10.0)

        if (gt_u is not None) and (gt_d0 is not None) and (gt_d1 is not None):
            mask_sf = gt_m_flow & gt_m_d0 & gt_m_d1
            if mask_sf.sum() > 0:
                diff_u = (pred_u - gt_u)[mask_sf]
                diff_v = (pred_v - gt_v)[mask_sf]
                diff_change = ((pred_d1 - pred_d0) - (gt_d1 - gt_d0))[mask_sf]
                sf_error = np.sqrt(diff_u**2 + diff_v**2 + diff_change**2)
                stats['sf'][0] += sf_error.sum()
                stats['sf'][1] += len(sf_error)

        print(f"Processed {fname}")

    print("\n" + "="*40)
    print(" KITTI EVALUATION RESULTS (EPE)")
    print("="*40)
    if stats['d0'][1]: print(f"Disparity (t)   : {stats['d0'][0]/stats['d0'][1]:.4f} px")
    if stats['d1'][1]: print(f"Disparity (t+1) : {stats['d1'][0]/stats['d1'][1]:.4f} px")
    if stats['dc'][1]: print(f"Disp. Change    : {stats['dc'][0]/stats['dc'][1]:.4f} px")
    if stats['flow'][1]: print(f"Optical Flow    : {stats['flow'][0]/stats['flow'][1]:.4f} px")
    print("-" * 40)
    if stats['sf'][1]: print(f"SCENE FLOW (3D) : {stats['sf'][0]/stats['sf'][1]:.4f} px")
    print("="*40)

if __name__ == "__main__":
    evaluate_folders(gt_root="test", pred_root="methods_results/PWOC-3D-Finetuned")