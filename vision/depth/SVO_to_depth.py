import cv2
import os
from vision.tools.video_wrapper import video_wrapper
from tqdm import tqdm


def convert_svo_to_depth_bgr_dgr(filepath, rotate=0, index=0, save = False):
    cam = video_wrapper(filepath, rotate=rotate, depth_minimum=0, depth_maximum=8)
    width = cam.get_width()
    height = cam.get_height()
    number_of_frames = cam.get_number_of_frames()
    print (f'Found {number_of_frames} frames in {filepath}')

    if save:
        output_path_depth = os.path.join(os.path.dirname(filepath), "zed_depth.avi")
        output_depth_video = cv2.VideoWriter(output_path_depth, cv2.VideoWriter_fourcc(*"MJPG"), 15, (width, height)) # TODO - it doesnt let me save as .mkv file

        output_path_BGR = os.path.join(os.path.dirname(filepath), "zed_bgr.avi")
        output_BGR_video = cv2.VideoWriter(output_path_BGR, cv2.VideoWriter_fourcc(*"MJPG"), 15, (width, height))

        output_path_DGR = os.path.join(os.path.dirname(filepath), "zed_dgr.avi")
        output_DGR_video = cv2.VideoWriter(output_path_DGR, cv2.VideoWriter_fourcc(*"MJPG"), 15, (width, height))

    with tqdm(total=number_of_frames) as pbar:
        while True:
            # #########
            if index != 0:
                if index % 30 == 0:   # save the previous frame to prevent frame shift
                    if save:
                        output_depth_video.write(frame_depth)
                        output_BGR_video.write(frame_bgr)
                        output_DGR_video.write(frame_dgr)
                    index += 1
            # #########
            cam.grab(index)
            frame_bgr, frame_depth = cam.get_zed(frame_number=index, exclude_depth=False, exclude_point_cloud=True, far_is_black = False, handle_nan = False, blur = False)

            b = frame_bgr[:, :, 0].copy()
            frame_depth[b > 170] = 255

            frame_dgr = cv2.merge([frame_depth, frame_bgr[:, :, 1], frame_bgr[:, :, 2]])

            if not save:
                frame_depth3D = cv2.cvtColor(frame_depth, cv2.COLOR_GRAY2BGR)  # depth to 3 channels
                merged_frame = cv2.hconcat([frame_bgr, frame_depth3D, frame_dgr])
                cv2.imshow('merged_frame', cv2.resize(merged_frame, None, fx=0.5, fy=0.5))
                cv2.waitKey(1)  # 1 millisecond delay

            if save:
                output_depth_video.write(frame_depth)
                output_BGR_video.write(frame_bgr)
                output_DGR_video.write(frame_dgr)

            index += 1
            pbar.update(1)
            if index >= number_of_frames:
                break


    cam.close()
    if save:
        output_depth_video.release()
        output_BGR_video.release()
        output_DGR_video.release()
        print (f'Saved to {os.path.dirname(filepath)}')
    else:
        cv2.destroyAllWindows()

    return output_path_depth, output_path_BGR, output_path_DGR

if __name__ == "__main__":

    import torch

    if torch.cuda.is_available():
        print("CUDA is available on this system.")
    else:
        print("CUDA is not available on this system.")
        
    fp = "/home/fruitspec-lab-3/FruitSpec/Data/customers/DEWAGD/190123/DWDBLE33/R29A/ZED_1.svo"
    #validate_output_path(output_path)
    local_path_depth, local_path_BGR, local_path_DGR = convert_svo_to_depth_bgr_dgr(fp, rotate=2, index=0, save = False)

