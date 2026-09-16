import cv2
import os

def downsample_video_fps(
    input_path,
    output_path,
    target_fps=2
):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_path}")

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        target_fps,
        (width, height)
    )

    step = fps_in / target_fps
    next_frame = 0.0
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx >= int(round(next_frame)):
            writer.write(frame)
            next_frame += step

        idx += 1

    cap.release()
    writer.release()
    print("Done:", output_path)

downsample_video_fps(
    input_path="input_video.mov",
    output_path="input_video_example.mov",
    target_fps=2
)
