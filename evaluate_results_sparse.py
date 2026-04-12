import os
import cv2
import numpy as np

def load_kitti_disp_with_mask(path):
    if not os.path.exists(path): return None, None
    data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if data is None: return None, None
    val = data.astype(np.float32) / 256.0
    mask = (data > 0)
    return val, mask

def load_kitti_flow_with_mask(path):
    if not os.path.exists(path): return None, None, None
    data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if data is None: return None, None, None

    u = (data[:, :, 2].astype(np.float32) - 32768.0) / 64.0
    v = (data[:, :, 1].astype(np.float32) - 32768.0) / 64.0
    mask = (data[:, :, 0] > 0)
    
    return u, v, mask

def compute_epe_sparse(pred, gt, gt_mask, pred_mask):
    combined_mask = gt_mask & pred_mask
    if combined_mask.sum() == 0: return 0.0, 0
    
    diff = pred - gt
    if len(diff.shape) == 3:
        error = np.sqrt(diff[0]**2 + diff[1]**2)
    else:
        error = np.abs(diff)
    return error[combined_mask].sum(), combined_mask.sum()

def evaluate_sparse(gt_root, pred_root):
    dir_gt_d0 = os.path.join(gt_root, "disp_occ_0")
    dir_gt_d1 = os.path.join(gt_root, "disp_occ_1")
    dir_gt_flow = os.path.join(gt_root, "flow_occ")

    dir_pred_d0 = os.path.join(pred_root, "disp_0")
    dir_pred_d1 = os.path.join(pred_root, "disp_1")
    dir_pred_flow = os.path.join(pred_root, "flow")

    files = [f for f in os.listdir(dir_gt_d0) if f.endswith("_10.png")]
    files.sort()

    stats = {
        'd0':   [0.0, 0, 0, 0], 
        'd1':   [0.0, 0, 0, 0], 
        'dc':   [0.0, 0, 0, 0],
        'flow': [0.0, 0, 0, 0], 
        'sf':   [0.0, 0, 0, 0]
    } 
    
    print(f"Starting evaluation... Found {len(files)} files.")
    
    for fname in files:
        gt_d0, m_gt_d0 = load_kitti_disp_with_mask(os.path.join(dir_gt_d0, fname))
        gt_d1, m_gt_d1 = load_kitti_disp_with_mask(os.path.join(dir_gt_d1, fname))
        gt_u, gt_v, m_gt_flow = load_kitti_flow_with_mask(os.path.join(dir_gt_flow, fname))

        p_d0, m_p_d0 = load_kitti_disp_with_mask(os.path.join(dir_pred_d0, fname))
        p_d1, m_p_d1 = load_kitti_disp_with_mask(os.path.join(dir_pred_d1, fname))
        p_u, p_v, m_p_flow = load_kitti_flow_with_mask(os.path.join(dir_pred_flow, fname))

        if p_d0 is None: continue
        h, w = p_d0.shape
        total_px = h * w

        es, cnt = compute_epe_sparse(p_d0, gt_d0, m_gt_d0, m_p_d0)
        stats['d0'][0]+=es; stats['d0'][1]+=cnt; stats['d0'][2]+=m_p_d0.sum(); stats['d0'][3]+=total_px

        es, cnt = compute_epe_sparse(p_d1, gt_d1, m_gt_d1, m_p_d1)
        stats['d1'][0]+=es; stats['d1'][1]+=cnt; stats['d1'][2]+=m_p_d1.sum(); stats['d1'][3]+=total_px

        es, cnt = compute_epe_sparse(np.stack([p_u, p_v]), np.stack([gt_u, gt_v]), m_gt_flow, m_p_flow)
        stats['flow'][0]+=es; stats['flow'][1]+=cnt; stats['flow'][2]+=m_p_flow.sum(); stats['flow'][3]+=total_px

        m_gt_dc = m_gt_d0 & m_gt_d1
        m_p_dc = m_p_d0 & m_p_d1
        es, cnt = compute_epe_sparse(p_d1 - p_d0, gt_d1 - gt_d0, m_gt_dc, m_p_dc)
        stats['dc'][0]+=es; stats['dc'][1]+=cnt; stats['dc'][2]+=m_p_dc.sum(); stats['dc'][3]+=total_px

        m_gt_sf = m_gt_flow & m_gt_d0 & m_gt_d1
        m_p_sf = m_p_flow & m_p_d0 & m_p_d1
        combined_sf = m_gt_sf & m_p_sf
        if combined_sf.sum() > 0:
            diff_u = (p_u - gt_u)[combined_sf]
            diff_v = (p_v - gt_v)[combined_sf]
            diff_dc = ((p_d1 - p_d0) - (gt_d1 - gt_d0))[combined_sf]
            stats['sf'][0] += np.sqrt(diff_u**2 + diff_v**2 + diff_dc**2).sum()
            stats['sf'][1] += combined_sf.sum()
        stats['sf'][2] += m_p_sf.sum(); stats['sf'][3] += total_px

        print(f"Processed {fname}")

    print("\n" + "="*40)
    print(" KITTI EVALUATION RESULTS (EPE : DENSITY)")
    print("="*40)
    
    def print_res(label, key):
        if stats[key][1] > 0:
            epe = stats[key][0] / stats[key][1]
            dens = (stats[key][2] / stats[key][3]) * 100
            print(f"{label:<15} : {epe:.4f} px : {dens:.2f}%")

    print_res("Disparity (t)",   'd0')
    print_res("Disparity (t+1)", 'd1')
    print_res("Disp. Change",    'dc')
    print_res("Optical Flow",    'flow')
    print("-" * 40)
    print_res("SCENE FLOW (3D)", 'sf')
    print("="*40)

if __name__ == "__main__":
    evaluate_sparse(gt_root="test", pred_root="methods_results/STEREO-RSSF")