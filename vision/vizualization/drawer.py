import cv2


def draw_rectangle(frame, start_point, end_point, color, thickness=2):

    frame = cv2.rectangle(frame, start_point, end_point, color, thickness)

    return frame

def draw_text(frame, text, start_point, size, color, above=True, font_scale=None, thickness=1, factor=10):

    if font_scale is None:
        scale, h, w = get_optimal_font_scale(text, size[0], thickness=thickness, factor=factor)
    else:
        scale = font_scale
        h, w = get_text_size(text, font_scale, thickness, factor)

    if above:
        frame = cv2.putText(frame,
                            text,
                            (start_point[0], start_point[1] - h),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            scale / factor,
                            color,
                            thickness)
    else:
        frame = cv2.putText(frame,
                            text,
                            (start_point[0], start_point[1] + h),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            scale / factor,
                            color,
                            thickness)

    return frame

def get_optimal_font_scale(text, width, thickness=1, factor=10):

    for scale in reversed(range(0, 60, 1)):
        h, w = get_text_size(text, scale, thickness, factor)
        if (w <= width):
            return (scale / factor), h, w
    h, w = get_text_size(text, 1, thickness, factor)
    return 1, h, w


def get_text_size(text, scale, thickness=1, factor=10):
    textSize = cv2.getTextSize(text,
                               fontFace=cv2.FONT_HERSHEY_DUPLEX,
                               fontScale=scale / factor,
                               thickness=thickness)
    width = textSize[0][0]
    height = textSize[0][1]

    return height, width