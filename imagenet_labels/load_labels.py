def load_labels(labels_file):
    with open(labels_file) as reader:
        f = reader.read()
        labels = f.splitlines()
    return labels

imagenet_labels = load_labels(r"imagenet_labels.txt")