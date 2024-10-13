import os
import shutil

import gdown

URL_BEST_MODEL = "https://drive.google.com/uc?id=1zMy1hehRC8XOkdYqtv_LfMrzMtuLgRoG"
URL_LM_MODEL = "https://drive.google.com/uc?id=1jFtw5W3znI6P9EBAgfwrzHcYtnNFDImW"
URL_VOCAB_LM = "https://drive.google.com/uc?id=185opmX4qeLLVaM89qs9ie8ZD2u5vtE5D"


def download():
    gdown.download(URL_BEST_MODEL)
    gdown.download(URL_LM_MODEL)
    gdown.download(URL_VOCAB_LM)

    os.makedirs("src/language_model", exist_ok=True)
    os.makedirs("src/model_weights", exist_ok=True)
    shutil.move("lm.arpa", "src/language_model/lm.arpa")
    shutil.move("lm_vocab.txt", "src/language_model/lm_vocab.txt")
    shutil.move("model_best.pth", "src/model_weights/model_best.pth")


if __name__ == "__main__":
    download()
