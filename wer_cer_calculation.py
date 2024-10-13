import argparse
import json
import os
from pathlib import Path

from src.metrics.utils import calc_cer, calc_wer
from src.text_encoder.ctc_text_encoder import CTCTextEncoder


def cer_wer(dir_path_pred, dir_path_gt):
    pred_json = os.path.join(dir_path_pred, "predictions.json")
    with open(pred_json, "r") as f:
        predictions = json.loads(f.read())

    cnt = 0
    wer = 0
    cer = 0

    for audio_file, prediction_data in predictions.items():
        file_name = Path(audio_file).stem

        text_file_path = Path(dir_path_gt) / f"{file_name}.txt"
        with open(text_file_path, "r", encoding="utf-8") as file:
            original_text = file.read()
        original_text = CTCTextEncoder.normalize_text(original_text)
        predicted_text = prediction_data["text_predicted"].strip()

        cer += calc_cer(original_text, predicted_text)
        wer += calc_wer(original_text, predicted_text)
        cnt += 1

    print("WER: {:.4f}".format(wer / cnt))
    print("CER: {:.4f}".format(cer / cnt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir_path_pred",
        required=True,
        type=str,
        help="Directory path for the predictions",
    )
    parser.add_argument(
        "--dir_path_gt",
        required=True,
        type=str,
        help="Directory path for the ground truth",
    )
    args = parser.parse_args()
    cer_wer(dir_path_gt=args.dir_path_gt, dir_path_pred=args.dir_path_pred)
