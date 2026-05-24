import os
import deeplabcut


config_path = r"C:\Users\Usuario\Desktop\CIN-1\01_dlc_project\Proyecto1-Laura-2026-04-13\config.yaml"
video_folder = r"C:\Users\Usuario\Desktop\CIN-2\new_videos"


videos_all = [
    f for f in os.listdir(video_folder)
    if f.lower().endswith(".mp4")
]

videos_to_process = []

for video_file in videos_all:

    base_name = os.path.splitext(video_file)[0]

  
    h5_exists = any(
        f.startswith(base_name) and f.endswith(".h5")
        for f in os.listdir(video_folder)
    )

    if not h5_exists:
        videos_to_process.append(os.path.join(video_folder, video_file))


print(f"Total videos: {len(videos_all)}")
print(f"Ya procesados: {len(videos_all) - len(videos_to_process)}")
print(f"Faltantes: {len(videos_to_process)}")


if len(videos_to_process) > 0:

    deeplabcut.analyze_videos(
        config_path,
        videos_to_process,
        save_as_csv=True
    )

    deeplabcut.filterpredictions(config_path, videos_to_process)

    print("Procesamiento completado")

else:
    print(" ya estaba procesado")