import cv2
import torch
import numpy as np
import torch.nn.functional as F
import os
import time
from tqdm import tqdm
from network import SceneFlowNet, flow_to_image_smart

class StereoInference:
    def __init__(self, weights_path, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SceneFlowNet().to(self.device)
        
        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device), strict=False)
            self.model.eval()
            print(f"Модель загружена: {weights_path}")
        else:
            raise FileNotFoundError(f"Веса не найдены: {weights_path}")

    def _prep_frame(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 256), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def run(self, path_L, path_R=None, output_dir="results", frame_skip=3):
        cap_L = cv2.VideoCapture(path_L)
        is_sbs = path_R is None
        cap_R = cv2.VideoCapture(path_R) if not is_sbs else None

        total_frames = int(cap_L.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap_L.get(cv2.CAP_PROP_FPS))
        full_w = int(cap_L.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_L.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = full_w // 2 if is_sbs else full_w
        
        os.makedirs(output_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_flow = cv2.VideoWriter(f'{output_dir}/flow.mp4', fourcc, fps, (width, height))
        out_disp0 = cv2.VideoWriter(f'{output_dir}/disp0.mp4', fourcc, fps, (width, height))
        out_disp1 = cv2.VideoWriter(f'{output_dir}/disp1.mp4', fourcc, fps, (width, height))

        def get_stereo_frames():
            if is_sbs:
                ret, frame = cap_L.read()
                if not ret: return False, None, None
                return True, frame[:, :width], frame[:, width:]
            else:
                retL, fL = cap_L.read()
                retR, fR = cap_R.read()
                if not retL or not retR: return False, None, None
                return True, fL, fR

        buffer_L = []
        buffer_R = []

        print(f"Инициализация буфера (интервал {frame_skip})...")
        for _ in range(frame_skip + 1):
            success, fL, fR = get_stereo_frames()
            if not success: break
            buffer_L.append(fL)
            buffer_R.append(fR)

        if len(buffer_L) < frame_skip + 1:
            print("Ошибка: Видео слишком короткое для такого интервала.")
            return

        print(f"Обработка {'Side-by-Side' if is_sbs else 'двух файлов'}: {total_frames} кадров")
        pbar = tqdm(total=total_frames - frame_skip, desc="Обработка видео", unit="кадр")

        while True:
            fL_t, fR_t = buffer_L[0], buffer_R[0]
            fL_t1, fR_t1 = buffer_L[-1], buffer_R[-1]

            tensors = [self._prep_frame(f) for f in [fL_t, fR_t, fL_t1, fR_t1]]
            input_tensor = torch.cat(tensors, dim=1)

            with torch.no_grad():
                output = self.model(input_tensor)
                output_res = F.interpolate(output, size=(height, width), mode='bilinear', align_corners=False)
                data = output_res[0].cpu().numpy()

            fx = data[0] * (width / 512.0)
            fy = data[1] * (height / 256.0)
            flow_bgr = cv2.cvtColor((flow_to_image_smart(fx, fy) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            out_flow.write(flow_bgr)

            def to_gray_disp(d_map):
                d_norm = np.clip(d_map / 128.0 * 255.0, 0, 255).astype(np.uint8)
                return cv2.cvtColor(d_norm, cv2.COLOR_GRAY2BGR)

            out_disp0.write(to_gray_disp(data[2] * (width / 512.0)))
            out_disp1.write(to_gray_disp(data[3] * (width / 512.0)))

            success, fL_next, fR_next = get_stereo_frames()
            if not success: break

            buffer_L.pop(0)
            buffer_R.pop(0)
            buffer_L.append(fL_next)
            buffer_R.append(fR_next)

            pbar.update(1)

        pbar.close()
        cap_L.release()
        if cap_R: cap_R.release()
        out_flow.release()
        out_disp0.release()
        out_disp1.release()
        print(f"\nГотово! Результаты сохранены в папку: {output_dir}")

if __name__ == "__main__":
    inf = StereoInference(weights_path="driving_driving_fast_kitti_new.pth")
    # Два файла
    #inf.run(path_L="cam_left.avi", path_R="cam_right.avi", output_dir="scene_flow_result")
    
    # Side-by-Side
    inf.run(path_L="video.mp4", output_dir="scene_flow_result", frame_skip=3)