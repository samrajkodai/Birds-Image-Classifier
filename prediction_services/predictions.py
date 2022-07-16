import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import yaml

def model_predict(img_path, model):

    img = image.load_img(img_path, target_size=(224, 224))

    x = image.img_to_array(img)
    
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    preds = model.predict(x)
    result=np.argmax(preds, axis=1)

    if result==0:
        result="ABBOTTS BABBLER"
    elif result==1:
        result="ABBOTTS BOOBY"
    elif result==2:
        result="ABYSSINIAN GROUND HORNBILL"
    elif result==3:
        result="AFRICAN CROWNED CRANE"
    elif result==4:
        result="AFRICAN EMERALD CUCKOO"
    elif result==5:
        result="AFRICAN FIREFINCH"
    elif result==6:
        result="AFRICAN OYSTER CATCHER"
    elif result==7:
        result="ALBATROSS"
    elif result==8:
        result="ALBERTS TOWHEE"
    elif result==9:
        result="ALEXANDRINE PARAKEET"
    elif result==10:
        result="ALPINE CHOUGH"
    elif result==11:
        result="ALTAMIRA YELLOWTHROAT"
    elif result==12:
        result="AMERICAN AVOCET"
    elif result==13:
        result="AMERICAN BITTERN"
    elif result==14:
        result="AMERICAN COOT"
    elif result==16:
        result="AMERICAN GOLDFINCH"
    elif result==17:
        result="AMERICAN KESTREL"
    elif result==18:
        result="AMERICAN PIPIT"
    elif result==19:
        result="AMERICAN REDSTART"
    else:
        result="AMETHYST WOODSTAR"          
    print(result)     # Convert to string
    return result