import cv2

def draw_highlighted_test(frame, text, start_point, width, bgd_color, frpmt_color, above=True, font_scale=None,
                          thickness=1, factor=10, box_h_margin=3):
    _, h, w = get_text_dim(text, width, thickness, font_scale, factor)

    x_s = int(start_point[0])
    x_end = int(start_point[0] + w)
    if above:
        rect_end = int(start_point[1])
        rect_s = int(start_point[1] - h - (2 * box_h_margin))
        #txt_s = start_point[1] - h - box_h_margin
        txt_s = int(rect_s + box_h_margin)
    else:
        rect_end = int(start_point[1] + h + (2 * box_h_margin))
        rect_s = int(start_point[1])
        #txt_s = start_point[1] + box_h_margin
        txt_s = int(rect_s + box_h_margin)

    frame = draw_rectangle(frame, (x_s, rect_s), (x_end, rect_end), bgd_color, -1)
    frame = draw_text(frame, text, (x_s, txt_s), width, frpmt_color, False, font_scale, thickness, factor)

    return frame


def draw_rectangle(frame, start_point, end_point, color, thickness=2):

    frame = cv2.rectangle(frame, start_point, end_point, color, thickness)

    return frame

def draw_text(frame, text, start_point, width, color, above=True, font_scale=None, thickness=1, factor=10):
    scale, h, w = get_text_dim(text, width, thickness, font_scale=font_scale, factor=factor)

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

def get_text_dim(text, width, thickness, font_scale=None, factor=10):
    if font_scale is None:
        scale, h, w = get_optimal_font_scale(text, width, thickness=thickness, factor=factor)
    else:
        scale = font_scale
        h, w = get_text_size(text, font_scale, thickness, factor)

    return  scale, h, w


def get_text_size(text, scale, thickness=1, factor=10):
    textSize = cv2.getTextSize(text,
                               fontFace=cv2.FONT_HERSHEY_DUPLEX,
                               fontScale=scale / factor,
                               thickness=thickness)
    width = textSize[0][0]
    height = textSize[0][1]

    return height, width

def get_color(val):
    #  https://sashamaps.net/docs/resources/20-colors/
    color_list = [(230, 25, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240),
                  (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255), (170, 110, 40),
                  (128, 0, 0), (128, 128, 0), (0, 0, 128), (255, 255, 255), (0, 0, 0)]
    return color_list[val]
