import cv2

'''def binary_draw(frame, mode, first_choice, second_choice):
    """
    Draw texts on image or video frame, used only for single title and two choices

    Args:
        frame (np.ndarray|) 
        b (int): The second integer.

    Returns:
        None
    
    Raises:
        ValueError: If either input is not an integer.
    """


    #cv2.rectangle(frame, (0, 0), (600, 65), (0, 0, 0), thickness=-1)
    #cv2.putText(frame, f'Current clarity: {sharpness}', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    cv2.rectangle(frame, (0, 65), (250, 130), (0, 0, 0), thickness=-1)
    if sharpness < self.threshold:
        cv2.putText(frame, f'Not Clear', (0, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    else:
        cv2.putText(frame, f'Clear', (0, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    if out:
        out.write(frame)
    else:
        cv2.imwrite(img_name, frame)'''    


def draw_with_bg(frame, text):
    cv2.rectangle(frame, (0, 0), (600, 65), (0, 0, 0), thickness=-1)
    cv2.putText(frame, text, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)


def config_writer(src_path, file_name):
    """
    Config writer for a video and return corresponding VideoCapture and VideoWriter objects.
    
    Args:
        src_path (str): original video path
        dest_path (str): target video path

    Returns:
        cv2.VideoCapture
        cv2.VideoWriter

    Raises:
        None
    """
    cap = cv2.VideoCapture(src_path)
    # get width and height
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Config output parameters
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # use MP4 encoding
    out = cv2.VideoWriter(file_name, fourcc, 30.0, (w, h))

    return cap, out