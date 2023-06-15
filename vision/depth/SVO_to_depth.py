import cv2
from vision.tools.video_wrapper import video_wrapper
from vision.misc.help_func import validate_output_path


def get_depth_video(filepath, output_path, rotate=0, index=0, resize_factor=3):
    cam = video_wrapper(filepath, rotate=rotate)
    number_of_frames = cam.get_number_of_frames()
    width = cam.get_width()
    height = cam.get_height()

    # Read until video is completed
    while True:
        # Capture frame-by-frame
        print(index)
        cam.grab(index)
        frame, frame_depth = cam.get_zed(frame_number=index, exclude_depth=False, exclude_point_cloud=True)

        cv2.imshow('headline', frame_depth)

        # Display the frame for a short period of time
        cv2.waitKey(1)  # 1 millisecond delay

        # Increment the index for the next frame
        index += 1

        # Check if the end of the video is reached
        if index >= number_of_frames:
            break

    # When everything is done, release the video capture object
    cam.close()

    # Close all the frames
    cv2.destroyAllWindows()




if __name__ == "__main__":

    import torch

    if torch.cuda.is_available():
        print("CUDA is available on this system.")
    else:
        print("CUDA is not available on this system.")
        
    fp = "/home/fruitspec-lab/FruitSpec/Data/customers/DEWAGD/190123/DWDBLE33/R11A/ZED_1.svo"
    output_path = "/home/fruitspec-lab/FruitSpec/Data/customers/DEWAGD/190123/DWDBLE33/R11A/debbug/"
    validate_output_path(output_path)
    get_depth_video(fp, output_path, rotate=2, index=0, resize_factor=3)

