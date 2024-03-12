import cv2
import os
import numpy as np
import time


def capture_and_save_images(label_name, dataset_dir='dataset', width_pixels = 28, height_pixels = 28):
    """
    Capture and save images with a specific label name using the webcam.

    Parameters:
    - label_name (str): The name to associate with the captured images.
    - dataset_dir (str): Directory where the dataset will be saved. Default is 'dataset'.
    - width_pixels (int): Width of the captured images in pixels. Default is 28.
    - height_pixels (int): Height of the captured images in pixels. Default is 28.

    Returns:
    - X (numpy.ndarray): Flattened images array.
    - y (numpy.ndarray): Labels array corresponding to the images.

    Notes:
    - The function captures images from the webcam until the 'q' key is pressed to quit.
    - Press the 's' key to save the current image with the specified label name.
    - Images are saved as grayscale PNG files in the specified dataset directory.
    - The function returns flattened images array (X) and labels array (y) as numpy arrays.
    """

    # Create the dataset directory if it doesn't exist
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Initialize the camera with the specified resolution
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width_pixels)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height_pixels)

    # Variables to store images and labels
    X = []
    y = []

    try:
        img_counter = 0
        while True:
            # Capture frame-by-frame
            ret, frame = camera.read()

            # Convert the frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Resize the frame to 28x28 pixels
            resized_frame = cv2.resize(gray_frame, (width_pixels, height_pixels), interpolation=cv2.INTER_AREA)

            # Flatten the image
            flattened_frame = resized_frame.flatten()

            # Display the resulting frame
            cv2.imshow('Capturing Images', resized_frame)

            # Wait for key press
            key = cv2.waitKey(1)

            # Save the flattened image when 's' key is pressed
            if key & 0xFF == ord('s'):
                img_name = os.path.join(dataset_dir, '{}_{}.png'.format(label_name, img_counter))
                cv2.imwrite(img_name, resized_frame)
                print('{} written!'.format(img_name))

                # Append the flattened image to X and add the label to y
                X.append(flattened_frame)
                y.append(label_name)
                
                img_counter += 1

            # Quit the program when 'q' key is pressed
            elif key & 0xFF == ord('q'):
                break

    finally:
        # Release the camera and close all OpenCV windows
        camera.release()
        cv2.destroyAllWindows()

    # Convert X and y to numpy arrays
    X = np.array(X)
    y = np.array(y)
    print('X shape:', X.shape)
    print('y shape:', y.shape)

    return X, y
    

def capture_and_predict_labels(model, class_names=None, width_pixels = 28, height_pixels = 28):
    """
    Capture images from the camera and predict their labels using a trained model.

    Parameters:
    - model: Trained model for prediction.
    - class_names (list or None): List of class names for mapping predictions. Default is None.
    - width_pixels (int): Width of the captured images in pixels. Default is 28.
    - height_pixels (int): Height of the captured images in pixels. Default is 28.

    Returns:
    - None

    Notes:
    - The function captures images from the webcam until the 'q' key is pressed to quit.
    - Press the 's' key to predict the label of the current image in real-time.
    - Predictions are displayed in the console.
    - If class_names is provided, it is used for mapping predicted labels.
    """

    # Initialize the camera with the specified resolution
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width_pixels)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height_pixels)

    try:
        while True:
            # Capture frame-by-frame
            ret, frame = camera.read()

            # Convert the frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Resize the frame to 28x28 pixels
            resized_frame = cv2.resize(gray_frame, (width_pixels, height_pixels), interpolation=cv2.INTER_AREA)

            # Flatten the image
            flattened_frame = resized_frame.flatten()

            # Display the resulting frame
            cv2.imshow('Capturing Images', resized_frame)

            # Wait for key press
            key = cv2.waitKey(1)

            # Perform prediction when 's' key is pressed
            if key & 0xFF == ord('s'):
                # Perform prediction
                flattened_frame = flattened_frame.reshape(1, -1)
                prediction = model.predict(flattened_frame)
                if class_names is None:
                    print('Predicted label:', prediction[0])
                else:
                    print('Predicted label:', class_names[prediction[0]])

            # Quit the program when 'q' key is pressed
            elif key & 0xFF == ord('q'):
                break

    finally:
        # Release the camera and close all OpenCV windows
        camera.release()
        cv2.destroyAllWindows()
        
        
def capture_and_predict_labels_live(model, class_names=None, width_pixels=28, height_pixels=28, refresh_interval=0.5):
    """
    Capture live images from the camera and predict their labels using a trained model.

    Parameters:
    - model: Trained model for prediction.
    - class_names (list or None): List of class names for mapping predictions. Default is None.
    - width_pixels (int): Width of the captured images in pixels. Default is 28.
    - height_pixels (int): Height of the captured images in pixels. Default is 28.
    - refresh_interval (float): Time interval in seconds to refresh the prediction. Default is 0.5.

    Returns:
    - None

    Notes:
    - The function captures images from the webcam until the 'q' key is pressed to quit.
    - Predictions are displayed within the OpenCV window next to the camera feed.
    - If class_names is provided, it is used for mapping predicted labels.
    """

    # Initialize the camera with the specified resolution
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width_pixels)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height_pixels)

    last_refresh_time = time.time()
    last_label_text = ''  # Store the last predicted label text

    try:
        while True:
            # Capture frame-by-frame
            ret, frame = camera.read()

            # Convert the frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Resize the frame to specified dimensions
            resized_frame = cv2.resize(gray_frame, (width_pixels, height_pixels), interpolation=cv2.INTER_AREA)

            # Flatten the image
            flattened_frame = resized_frame.flatten()

            # Get the current time
            current_time = time.time()

            # Check if it's time to refresh the prediction
            if current_time - last_refresh_time >= refresh_interval:
                # Perform prediction
                flattened_frame = flattened_frame.reshape(1, -1)
                prediction = model.predict(flattened_frame)

                # Prepare text to display
                if class_names is None:
                    last_label_text = '{}'.format(prediction[0])
                else:
                    last_label_text = '{}'.format(class_names[prediction[0]])

                # Update the last refresh time
                last_refresh_time = current_time

            # Add the text to the frame
            cv2.putText(frame, last_label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the resulting frame
            cv2.imshow('Capturing Images', frame)

            # Quit the program when 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Release the camera and close all OpenCV windows
        camera.release()
        cv2.destroyAllWindows()

