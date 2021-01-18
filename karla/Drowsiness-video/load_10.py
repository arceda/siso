from keras.models import load_model
from models_mobile_v1 import ResearchModels
from data_10 import Dataset
import numpy as np
import os
import os.path
# import winsound

def load():
    data = Dataset(
        seq_length=10,
        class_limit=2,
    )

    X_test, y_test = data.get_all_sequences_in_memory('testing')

    print ("##################################################")
    X_test = np.ravel(X_test)
    X_test = X_test.reshape(4, 10, -1)
    print("X_test.shape", X_test.shape)
    print("y_test.shape", y_test.shape)
    print("##################################################")

    model = load_model('my_model.h5')
    predictions = model.predict(X_test)
    loss, accuracy = model.evaluate(X_test, y_test)
    # print ("==============TEST=====AFTER=====SAVE==============")
    # print("prediction: \n", predictions)
    # print("loss:", loss, "accuracy:", accuracy)
    # print ("==============TEST=====AFTER=====SAVE==============")

    for j in predictions:
        print("---------------------------------------------------------------")
        if j[0] > j[1]:
            print("Driver is alert with the confidence of", (j[0]*100), "%")
            #print(j[1]*100)
        else:
            print("Driver is drowsy with the confidence of", (j[1]*100), "%")
            #print(j[0]*100)
            print("Sounding the alarm now....")
            # duration = 10  # second
            # freq = 440  # Hz
            # os.system(
            #     'play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))
            # winsound.Beep(440, 1000)

def main():
    load()


if __name__ == '__main__':
    main()