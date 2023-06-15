import cv2
import os
from vision.tools.video_wrapper import video_wrapper
from vision.misc.help_func import validate_output_path


def get_depth_video(filepath, rotate=0, index=0):
    cam = video_wrapper(filepath, rotate=rotate, depth_minimum=0, depth_maximum=8)
    width = cam.get_width()
    height = cam.get_height()
    number_of_frames = cam.get_number_of_frames()

    output_filename = os.path.join(os.path.dirname(filepath), "DEPTH.mkv")
    output_video = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*"X264"), 15, (width, height))

    while True:
        print(index)
        cam.grab(index)
        frame_bgr, frame_depth = cam.get_zed(frame_number=index, exclude_depth=False, exclude_point_cloud=True, far_is_black = False)

        frame_depth3D = cv2.cvtColor(frame_depth, cv2.COLOR_GRAY2BGR) # depth to 3 channels
        frame_dgr = cv2.merge(frame_depth, frame_bgr[:, :, 1:])

        merged_frame = cv2.hconcat([frame_bgr, frame_depth3D, frame_dgr])
        cv2.imshow('merged_frame', cv2.resize(merged_frame, None, fx=0.5, fy=0.5))
        cv2.waitKey(1)  # 1 millisecond delay

        # save:
        output_video.write(frame_depth)

        index += 1
        if index >= number_of_frames:
            break

    output_video.release()
    cam.close()
    cv2.destroyAllWindows()




if __name__ == "__main__":

    import torch

    if torch.cuda.is_available():
        print("CUDA is available on this system.")
    else:
        print("CUDA is not available on this system.")
        
    fp = "/home/fruitspec-lab-3/FruitSpec/Data/customers/DEWAGD/190123/DWDBLE33/R11A/ZED_1.svo"
    output_path = "/home/fruitspec-lab-3/FruitSpec/Data/customers/DEWAGD/190123/DWDBLE33/R11A/debugLihi/"
    validate_output_path(output_path)
    get_depth_video(fp, output_path, rotate=2, index=0, resize_factor=3)

