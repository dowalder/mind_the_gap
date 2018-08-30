import PIL.Image


def load_image(filename, size=None, scale=None):
    img = PIL.Image.open(filename)
    if size is not None:
        img = img.resize((size, size), PIL.Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), PIL.Image.ANTIALIAS)
    return img


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = PIL.Image.fromarray(img)
    img.save(filename)
