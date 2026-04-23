import cv2
import os

def create_video(image_folder, output_video, fps=24):
    images = [f"{str(i).zfill(10)}.png" for i in range(447)]
    first_image_path = os.path.join(image_folder, images[0])
    if not os.path.exists(first_image_path):
        print(f"Ошибка: Файл {first_image_path} не найден!")
        return

    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    print(f"Начинаю сборку видео из {len(images)} кадров...")

    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        if os.path.exists(image_path):
            frame = cv2.imread(image_path)
            video.write(frame)
        else:
            print(f"Предупреждение: Кадр {image_name} пропущен (не найден).")

    video.release()
    print(f"Готово! Видео сохранено как: {output_video}")

folder = 'data'
output = 'result_02.mp4'
frame_rate = 10

create_video(folder, output, frame_rate)