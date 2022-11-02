#Importing Libraries
from mtcnn import MTCNN
import cv2
import numpy as np
from tkinter import *
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import argparse
import yaml


# This is to open the camera config file for webcam selection
with open("./camera_config.yaml", "r") as f:
    camera_cfg = yaml.safe_load(f)


# This is used to parse our arguments to aid with error messages
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i",
                    "--image",
                    help="path to input image containing faces")
    ap.add_argument("-m",
                    "--method",
                    type=str,
                    default="gaussian",
                    choices=["gaussian", "blackbar"],
                    help="face blurring/ anonymizing method")
    ap.add_argument("-ft",
                    "--factor",
                    type=int,
                    default=1,
                    help="factor for the Guassian blurring method, you can change")
    ap.add_argument("-c",
                    "--confidence",
                    type=float,
                    default=0.2,
                    help="minimum probability to filter weak detections")

    args = vars(ap.parse_args())
    return args


# Building the black bar function
def anonymize_face_blackbar(image):
    (h, w) = image.shape[:2]
    return np.zeros_like(image)

# Building the Guassian blur function
def anonymize_face_gaussian(image, factor=1.0):
    # automatically determine the size of the blurring kernel based on the spatial dimensions of the input image
    (h, w) = image.shape[:2]
    #kW,kH are kernel dimensions, The larger the kernel size, the more blurred the output face will be
    kW = int(w / factor)
    kH = int(h / factor)

    # ensure the width and the height of the kernel is odd so that it can placed at a central.
    if kW % 2 == 0:
        kW -= 1
    if kH % 2 == 0:
        kH -= 1
    # apply Guassian blur to the input image.
    return cv2.GaussianBlur(image, (kW, kH), 0)

# Building the Sobel blur filter
def sobel(image, ddepth, dx, dy, ksize):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered = cv2.Sobel(gray, ddepth, dx, dy, ksize)
    ret = np.dstack([filtered, filtered, filtered])
    return ret

# Running Image detection
def run_detection(image, detector, filter_registry, method):

    img_with_dets = image.copy()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # detections is an array with all the bounding boxes detected.
    detections = detector.detect_faces(image)
    count = 0
    min_conf = 0.9

    # finding the co-ordinates of facial features
    for det in detections:
        if det['confidence'] >= min_conf:
            count += 1
            x, y, width, height = det['box']
            keypoints = det['keypoints']
            cv2.rectangle(img_with_dets, (x, y),
                          (x + width, y + height), (0, 0, 255), 2)
            cv2.circle(img_with_dets,
                       (keypoints['left_eye']), 2, (0, 0, 255), 2)
            cv2.circle(img_with_dets,
                       (keypoints['right_eye']), 2, (0, 0, 255), 2)
            cv2.circle(img_with_dets, (keypoints['nose']), 2, (0, 0, 255), 2)
            cv2.circle(img_with_dets,
                       (keypoints['mouth_left']), 2, (0, 0, 255), 2)
            cv2.circle(img_with_dets,
                       (keypoints['mouth_right']), 2, (0, 0, 255), 2)

            # face ROI
            face = image[y:y + height, x:x + width]

            # anonymizer all faces
            face = filter_registry[method]['func'](
                face, **filter_registry[method]['params'])

            # replace blurred face to the original image.
            image[y:y + height, x:x + width] = face

    # display the original image and blurred image faces
    # conver the color back to RGB
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, count

# video capture takes in cv functions and runs through mtcnn
def main(filter_registry, value, detector):

    if camera_cfg['use_cv2_cap_dshow']:
        cap = cv2.VideoCapture(camera_cfg['camera_id'], cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(camera_cfg['camera_id'], )
    while cap.isOpened():
        ret, frame = cap.read()
        frame, count = run_detection(frame, detector, filter_registry, value)
        if not ret:
            break
        cv2.imshow('Real-Time Facial Anonymizer', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# returns "clicked" confirmation
def clicked(value):
    myLabel = Label(root, text=value)
    myLabel.pack()
    detector = MTCNN()
    main(filter_registry, value, detector)

# Building the Open and process image function
def open_files(run_anonymizer=False, detector=None, method=None,):
    filetypes = (
        ('png', '*.png'),
        ('jpg', '*.jpg'),
        ('jpeg', '*.jpeg'),
        ('All files', '*.*')
    )
    filename = filedialog.askopenfilename(
        initialdir="./", title="Choose An Image", filetypes=filetypes)
    open(filename, run_anonymizer=run_anonymizer,
         detector=detector, method=method)

# Building "Demo" function to test loaded image
def open(filename="./images/Alicia_Witt_0001.jpeg", run_anonymizer=False, detector=None, method=None):
    """opens single image which we should apply blurs to for illustration"""

    my_img = Image.open(filename)

    show_image(my_img, title='Facial Anonymization',
               run_anonymizer=run_anonymizer, detector=detector, method=method)

# Converting the file to pillow image to work with tkinter
def show_image(pillow_image, title="Facial Anonymization", run_anonymizer=False, detector=None, method=None):
    top = Toplevel()
    top.title(title)

    try:
        top.iconbitmap("./mask.ico")
    except Exception as e:
        print("could not load icon, likely unsupported os")

    if run_anonymizer:
        frame = np.array(pillow_image).astype(np.uint8)
        frame, _ = run_detection(
            image=frame, detector=detector, filter_registry=filter_registry, method=method)
        pillow_image = Image.fromarray(frame)

    ph = ImageTk.PhotoImage(pillow_image)
    my_label = Label(top, image=ph)
    my_label.pack()
    my_label.image = ph

# Anonimization naming for differents functions
if __name__ == "__main__":
    filter_registry = {
        'Black Bar': {
            'func': anonymize_face_blackbar,
            'params': {},
        },

        'Gaussian Blur': {
            'func': anonymize_face_gaussian,
            'params': {
                'factor': 3.0,
            }
        },

        'Median Blur': {
            'func': cv2.medianBlur,
            'params': {
                "ksize": 11,

            }
        },

        'Scharr': {
            'func': cv2.Sobel,
            'params': {
                "dx": 1,
                "dy": 1,
                "ddepth": -1,
                "ksize": 3,
                "scale": 1,
                "delta": 0,
                "borderType": cv2.BORDER_CONSTANT

            }
        },

        'Sobel': {
            'func': sobel,
            'params': {
                'ddepth': cv2.CV_16S,
                'dx': 0,
                'dy': 1,
                'ksize': 3,
            }
        }

    }

    # GUI for the Anonymizer
    root = Tk()
    root.title('Face Anonymizer')
    root.geometry("840x640")

    frame = tk.Frame(root)
    frame.place(relx=1, rely=1, relwidth=1, relheight=1, anchor='e')
    filters = [(txt, txt) for txt in filter_registry]

    # Ptinker buttons and gui
    filter = StringVar()
    filter.set('Black Bar')

    # Render radio buttons based on available filters
    for text, filt in filters:
        Radiobutton(root, text=text, variable=filter,
                    value=filt, bd=3, pady=14).pack(anchor=W)

    detector = MTCNN()
    kernel = np.ones((45, 45), np.uint8)  # kernels to run filters
    kernel2 = np.ones((5, 5), np.float32) / 25
    ksize = (10, 10)  # a ksize to run these filters as a default

    Search_img = Button(
        root, text="Open & Process Image", command=lambda: open_files(
            run_anonymizer=True,
            method=filter.get(),
            detector=detector)).place(
        relx=.5,
        rely=.55,
        relwidth=.3,
        relheight=.08,
        anchor='n')


    # Button for Webcam Anonymizer
    myButton = Button(root, text="Real-time Facial Anonymizer",
                      command=lambda: clicked(filter.get()))
    myButton.place(relx=.5, rely=.45, relwidth=.3, relheight=.08, anchor='n')

    btn = Button(root, text="Demo", command=lambda: open(
        run_anonymizer=True,
        method=filter.get(),
        detector=detector,)).place(
        relx=.5,
        rely=.35,
        relwidth=.3,
        relheight=.08,
        anchor='n')

    mainloop()
