# Based on seminar materials

# Don't forget to support cases when target_text == ''
import editdistance


def calc_cer(target_text, predicted_text) -> float:
    if not len(target_text):
        return 1
    splitted_target = target_text.split()
    return editdistance.eval(splitted_target, predicted_text.split()) / len(
        splitted_target
    )


def calc_wer(target_text, predicted_text) -> float:
    if not len(target_text):
        return 1
    return editdistance.eval(target_text, predicted_text) / len(target_text)
