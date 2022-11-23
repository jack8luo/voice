import os
import random


def get_data_list(wav_path, meta_txt_path, bi=0.3):
    speakers = os.listdir(wav_path)
    for si in speakers:
        si_wi_list = []
        wav_si_path = os.path.join(wav_path, si)
        wi_list = os.listdir(wav_si_path)
        for wi in wi_list:
            wi_path = os.path.join(wav_si_path, wi)
            si_wi_list.append(wi_path + f"\t{si}\n")
        test_len = int(len(si_wi_list) * bi)
        random.shuffle(si_wi_list)
        train_list = si_wi_list[test_len:]
        test_list = si_wi_list[:test_len]
        with open(os.path.join(meta_txt_path, 'train.txt'), 'a', encoding='utf-8') as f:
            f.writelines(train_list)
        with open(os.path.join(meta_txt_path, 'test.txt'), 'a', encoding='utf-8') as f:
            f.writelines(test_list)


if __name__ == '__main__':
    get_data_list('data_man', 'meta_txt_man')
