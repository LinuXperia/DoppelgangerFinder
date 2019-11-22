# Train multiple images per person
# Find and recognize faces in an image using a SVC with scikit-learn

# source: https://github.com/ageitgey/face_recognition/blob/master/examples/face_recognition_svm.py

"""
Structure:
        <test_image>.jpg
        <train_dir>/
            <person_1>/
                <person_1_face-1>.jpg
                <person_1_face-2>.jpg
                .
                .
                <person_1_face-n>.jpg
           <person_2>/
                <person_2_face-1>.jpg
                <person_2_face-2>.jpg
                .
                .
                <person_2_face-n>.jpg
            .
            .
            <person_n>/
                <person_n_face-1>.jpg
                <person_n_face-2>.jpg
                .
                .
                <person_n_face-n>.jpg
"""

from face_recognition import face_recognition
from sklearn import svm
import os
from PIL import Image, ImageDraw, ImageFont
import math
import textwrap
import requests
import urllib.request
from joblib import dump, load
import os.path
from os import path
import warnings

def train():
    # Training the SVC classifier

    # The training data would be all the face encodings from all the known images and the labels are their names
    encodings = []
    names = []

    # Training directory
    train_dir = os.listdir("train_dir/")

    # Loop through each person in the training directory
    for person in train_dir:
        pix = os.listdir("train_dir/" + person)
        # pix = [item for item in pix if not item.startswith('.') and os.path.isfile(os.path.join(root, item))]

        # Loop through each training image for the current person
        for person_img in pix:
            # Get the face encodings for the face in each image file
            if person_img == ".DS_Store":
                continue
            face = face_recognition.load_image_file(
                "train_dir/" + person + "/" + person_img
            )
            face_bounding_boxes = face_recognition.face_locations(face)

            if len(face_bounding_boxes) == 1:
                face_enc = face_recognition.face_encodings(face)[0]
                # Add face encoding for current image with corresponding label (name) to the training data
                encodings.append(face_enc)
                names.append(person)

    # Create and train the SVC classifier
    clf = svm.SVC(gamma="scale", probability=True)
    clf.fit(encodings, names)
    dump(clf, 'clf.joblib')
    return clf


def match(clf, filename):

    # Load the test image with unknown faces into a numpy array
    test_image = face_recognition.load_image_file(filename)

    # Find all the faces in the test image using the default HOG-based model
    face_locations = face_recognition.face_locations(test_image)
    no = len(face_locations)
    print("\nNumber of faces detected: ", no)

    # Predict all the faces in the test image using the trained classifier
    firstname = None
    print("\n(☞ﾟヮﾟ)☞   ☜(ﾟヮﾟ☜)\n")
    print("Found:")
    for i in range(no):
        test_image_enc = face_recognition.face_encodings(test_image)[i]
        name = clf.predict([test_image_enc])
        probs = clf.predict_proba([test_image_enc])
        print(*name)
        firstname = (str(name[0]))
    return (firstname, no)


def generate_image(actorname, my_img_file, no):
    if no > 1:
        my_im = Image.open(my_img_file)
        my_im.show()
        return

    canvas_width = 500
    canvas_height = 500

    # format name
    if actorname.split()[-1] == 'face':
        actorname = ' '.join(actorname.split()[:-1])
    actorname = ''.join(map(lambda x: x if x.islower() else " "+x, actorname))

    nospaces = actorname.replace(" ", "")
    nospaces = ' '.join(nospaces.split())

    # call CS API
    URL = "https://www.googleapis.com/customsearch/v1?"
    # NOTE: you need to get a Google Custom Search API Key Here: https://developers.google.com/custom-search/v1/overview
    # And A Custom Search Key Here: https://cse.google.com/cse/create/new
    # Following these specifications:
    # Image Search: On, Search Entire Web: On, Schema.org TypeL Person, Sites to Search: None
    PARAMS = {
        "key": "ADD_KEY_HERE"
        "cx": "ADD_CX_HERE",
        "q": nospaces,
        "searchType": "image",
    }

    # get resulting image
    data = requests.get(url=URL, params=PARAMS).json()
    imgurl = (data['items'][0]['link'])

    # save the image
    try:
        with urllib.request.urlopen(imgurl) as url:
            with open('temp.jpg', 'wb') as f:
                f.write(url.read())
    except Exception:
        imgurl = (data['items'][1]['link'])
        with urllib.request.urlopen(imgurl) as url:
            with open('temp.jpg', 'wb') as f:
                f.write(url.read())

    # ignore warnings
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    # open image
    im = Image.open(urllib.request.urlopen(imgurl))
    basewidth = 250

    # TODO: if proportions of an image are off (too tall), the cropping will distort the image

    # resize actor image
    wpercent = (basewidth/float(im.size[0]))
    hsize1 = int((float(im.size[1])*float(wpercent)))
    im = im.resize((basewidth,hsize1), Image.ANTIALIAS)
    old_width, old_height = im.size

    # rescale user image
    myimg = Image.open(my_img_file)
    hpercent = (old_height/float(myimg.size[0]))
    wsize = int((float(myimg.size[0])*float(hpercent)))
    myimg = myimg.resize((wsize,old_height), Image.ANTIALIAS) # resize to be same height

    # crop to be right width
    my_width, my_height = myimg.size
    if my_width > 300:
        diff = my_width - 300
        left = (my_width - basewidth) / 2
        myimg = myimg.crop((left, 0, left + basewidth, my_height))


    old_width, old_height = myimg.size

    images = [myimg, im]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    # format background of canvas
    mode = myimg.mode
    if len(mode) == 1:  # L, 1
        new_background = (255)
    if len(mode) == 3:  # RGB
        new_background = (255, 255, 255)
    if len(mode) == 4:  # RGBA, CMYK
        new_background = (255, 255, 255, 255)

    new_im = Image.new(mode, (500,500), new_background)

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,100))
        x_offset += im.size[0]

    # add image to canvas
    draw = ImageDraw.Draw(new_im)

    # get font
    fontsize = 15
    txt = f'Your Doppelgänger is {actorname.title()}!!'
    font = ImageFont.truetype("kollektif.ttf", fontsize)

    while font.getsize(txt)[0] < 480:
        # iterate until the text size is just larger than the criteria
        fontsize += 1
        font = ImageFont.truetype("kollektif.ttf", fontsize)

    # format and add text
    text = textwrap.fill(txt, width=50)
    draw.multiline_text((10,30),text, fill='black', font=font, align="center", spacing = 6) #drawing text on the black strip


    new_im.show()
    new_im.save('result.jpg')


def main():
    # didn't have time to sort data to implement this, but it's not really needed
    # gender = input('Pick the gender of your celebrity doppleganger: (F)emale, (M)ale, or (B)oth').lower()[0]
    # while (gender not in ['f', 'm', 'b']):
    #     gender = input('Invalid input. \
    #         Pick the gender of your celebrity doppleganger: (F)emale, (M)ale, or (B)oth').lower()[0]

    # industry = input('Pick a film industry for your celebrity doppleganger (all works best!): (H)ollywood, (B)ollywood, (T)ollywood')
    # while (gender not in ['h', 'b', 't']):
    #     gender = input('Invalid input. \
    #         Pick a film industry for your celebrity doppleganger (all works best!): (H)ollywood, (B)ollywood, (T)ollywood').lower()[0]

    print("\n===================================================================")
    # print("Welcome to the Celebrity Doppleganger Finder!\n\n")
    print("\nWelcome to the")
    print("""
   ____     _      _          _ _
  / ___|___| | ___| |__  _ __(_) |_ _   _
 | |   / _ \ |/ _ \ '_ \| '__| | __| | | |
 | |__|  __/ |  __/ |_) | |  | | |_| |_| |
  \____\___|_|\___|_.__/|_|  |_|\__|\__, |
  ____                         _    |___/  _
 |  _ \  ___  _ __  _ __   ___| | __ _(_)_(_)_ __   __ _  ___ _ __
 | | | |/ _ \| '_ \| '_ \ / _ \ |/ _` |/ _` | '_ \ / _` |/ _ \ '__|
 | |_| | (_) | |_) | |_) |  __/ | (_| | (_| | | | | (_| |  __/ |
 |____/ \___/| .__/| .__/ \___|_|\__, |\__,_|_| |_|\__, |\___|_|
  _____ _    |_|   |_|           |___/             |___/
 |  ___(_)_ __   __| | ___ _ __
 | |_  | | '_ \ / _` |/ _ \ '__|
 |  _| | | | | | (_| |  __/ |
 |_|   |_|_| |_|\__,_|\___|_|

    """)

    # load classifier
    if path.exists('clf.joblib'):
        clf = load('clf.joblib')
    else:
        clf = train()
    print("The Machine Learning Model has been trained! 🧠\n\nNow, let's find your match.\n")

    while True:
        filename = ""
        while not os.path.isfile(filename):
            filename = input("Which file would you like to use? Press enter to default to test_image.jpg: ")
            if filename == "":
                filename = "test_image.jpg"
                break
            elif filename == 'q': # exit out of program
                return
            elif os.path.isfile(filename):
                break
            else:
                print("Invalid file name. Try again.\n")

        res, no = match(clf, filename)
        print("\n===================================================================\n")
        generate_image(res, filename, no)

main()
