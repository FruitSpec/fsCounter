import cv2
import os
from vision.tools.video_wrapper import video_wrapper
from tqdm import tqdm


def get_depth_video(filepath, rotate=0, index=0):
    cam = video_wrapper(filepath, rotate=rotate, depth_minimum=0, depth_maximum=8)
    width = cam.get_width()
    height = cam.get_height()
    number_of_frames = cam.get_number_of_frames()

    output_filename = os.path.join(os.path.dirname(filepath), "DEPTH.avi")
    output_depth_video = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*"MJPG"), 15, (width, height)) # TODO - it doesnt let me save as .mkv file
    output_BGR_video = cv2.VideoWriter(os.path.join(os.path.dirname(filepath), "rgb_zed.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 15, (width, height))
    output_DGR_video = cv2.VideoWriter(os.path.join(os.path.dirname(filepath), "dgb_zed.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 15, (width, height))
    with tqdm(total=number_of_frames) as pbar:
        while True:
            cam.grab(index)
            frame_bgr, frame_depth = cam.get_zed(frame_number=index, exclude_depth=False, exclude_point_cloud=True, far_is_black = False, handle_nan = False, blur = False)

            b = frame_bgr[:, :, 0].copy()
            frame_depth[b > 240] = 255

            frame_depth3D = cv2.cvtColor(frame_depth, cv2.COLOR_GRAY2BGR) # depth to 3 channels
            frame_dgr = cv2.merge([frame_depth, frame_bgr[:, :, 1], frame_bgr[:, :, 2]])
            #
            # merged_frame = cv2.hconcat([frame_bgr, frame_depth3D, frame_dgr])
            # cv2.imshow('merged_frame', cv2.resize(merged_frame, None, fx=0.5, fy=0.5))
            # cv2.waitKey(1)  # 1 millisecond delay

            # save:
            output_depth_video.write(frame_depth3D)
            output_BGR_video.write(frame_bgr)
            output_DGR_video.write(frame_dgr)

            index += 1
            pbar.update(1)
            if index >= number_of_frames:
                break


    cam.close()
    output_depth_video.release()
    output_BGR_video.release()
    output_DGR_video.release()
    print (f'Saved to {output_filename}')
    cv2.destroyAllWindows()




if __name__ == "__main__":

    import torch

    if torch.cuda.is_available():
        print("CUDA is available on this system.")
    else:
        print("CUDA is not available on this system.")
        
    fp = "/home/fruitspec-lab-3/FruitSpec/Data/customers/DEWAGD/190123/DWDBLE33/R29A/ZED_1.svo"
    #validate_output_path(output_path)
    get_depth_video(fp,  rotate=2, index=0)

