from wand.image import Image
import os

files = os.listdir("../all/answer")
for f in files:
    with Image(filename='all/answer/' + f) as img:
        img.deskew(0.4*img.quantum_range)
        img.save(filename='all/skew_corrected_answer/' + f)
        #display(img)