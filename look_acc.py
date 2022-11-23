from matplotlib import pyplot as plt


def look_acc(acc_path):
    with open(acc_path, 'r', encoding='utf-8') as f:
        acc_s = f.readline().split(',')
    acc_f = []
    for s in acc_s:
        acc_f.append(float(s))
    x = range(len(acc_f))
    plt.figure()
    plt.xticks(x)
    plt.plot(acc_f)
    plt.xlabel('epoch')
    plt.ylabel('acc_value')
    plt.show()


if __name__ == '__main__':
    look_acc('meta_txt/eval_acc.txt')
