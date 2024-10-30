import json
import matplotlib.pyplot as plt


def read_json(path_to_json: str):
    loss_history = []
    with open(path_to_json, 'r', encoding='utf-8') as file:
        for line in file:
            line = json.loads(line)
            try:
                loss = line['loss']
                loss_history.append(loss)
            except KeyError:
                pass
    return loss_history

def draw_plot(loss_history):
    epochs = range(1, len(loss_history) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_history, marker='o', color='b', label='Loss')
    plt.title('Зависимость ошибки от эпохи')
    plt.xlabel('Эпоха')
    plt.ylabel('Ошибка (Loss)')
    plt.grid(True)
    plt.legend()
    plt.show()


def main():
    path_to_json = "../mmsegmentation/work_dirs/pspnet_r50-d8_4xb4-20k_coco-stuff10k-512x512/20241024_145129/vis_data/20241024_145129.json"
    loss_history = read_json(path_to_json)
    draw_plot(loss_history)


if __name__ == "__main__":
    main()
