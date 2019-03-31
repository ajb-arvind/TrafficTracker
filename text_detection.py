import io
import os
from google.cloud import vision_v1p3beta1 as vision
from datetime import datetime
import cv2

# Setup google authen client key
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'client_key.json'

# Source path content all images
SOURCE_PATH = "/home/laraib/Music/test/numberplate/"


def recognize_license_plate(img_path):

    start_time = datetime.now()

    # Read image with opencv
    img = cv2.imread(img_path)

    # Get image size
    height, width = img.shape[:2]

    #crp_img = img[103:748,291:1300]
    #cv2.imshow("Crop", crp_img)
    # Scale image
    img = cv2.resize(img, (800, int((height * 800) / width)))

    # Show the origin image
    cv2.imshow('Origin image', img)

    # Save the image to temp file
    cv2.imwrite(SOURCE_PATH + "output.jpg", img)

    # Create new img path for google vision
    img_path = SOURCE_PATH + "output.jpg"

    # Create google vision client
    client = vision.ImageAnnotatorClient()

    # Read image file
    with io.open(img_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    # Recognize text
    response = client.text_detection(image=image)
    texts = response.text_annotations

    for text in texts:
        print(len(text.description))
        if len(text.description) == 13:
            license_plate = text.description
            print(license_plate)
            vertices = [(vertex.x, vertex.y)
                        for vertex in text.bounding_poly.vertices]

            # Put text license plate number to image
            cv2.putText(img, license_plate, (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

            print(vertices)
            # Draw rectangle around license plate
            cv2.rectangle(img, (vertices[0][0]-10, vertices[0][1]-10), (vertices[2][0]+10, vertices[2][1]+10), (0, 255, 0), 3)
            print('Total time: {}'.format(datetime.now() - start_time))
            cv2.imshow('Recognize & Draw', img)
            cv2.waitKey(0)

print('---------- Start recognize license palate --------')
path = SOURCE_PATH + 'images2.jpg'
recognize_license_plate(path)
print('---------- End ----------')
'''291.4069633483887
1300.895091533661
103.35073247551918
748.5175788402557
103.51417601108551
1029.1144180297852
52.1240758895874
673.725757598877'''
