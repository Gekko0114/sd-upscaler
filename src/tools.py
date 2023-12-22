# Save your samples to google drive (disabled by default)
# Save location format:
# %T: timestamp
# %S: seed
# %I: image index
# %P: prompt (will be truncated to avoid overly long filenames)

import hashlib
import io
import os
import re

import requests
from PIL import Image


def clean_prompt(prompt):
    badchars = re.compile(r"[/\\]")
    prompt = badchars.sub("_", prompt)
    if len(prompt) > 100:
        prompt = prompt[:100] + "…"
    prompt = prompt.replace(" ", "_")
    return prompt


def format_filename(timestamp, seed, index, prompt, save_location):
    string = save_location
    string = string.replace("%T", f"{timestamp}")
    string = string.replace("%S", f"{seed}")
    string = string.replace("%I", f"{index:02}")
    string = string.replace("%P", clean_prompt(prompt))
    return string


def save_image(image, **kwargs):
    filename = format_filename(**kwargs)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    print("saving file: " + filename)
    image.save(filename)
    print("saved file.")


def pil_to_bytes(image):
    with io.BytesIO() as fp:
        image.save(fp, format="png")
        return fp.getvalue()


def on_upload(change, image_widget):
    global input_image
    if change["name"] == "value":
        (value,) = change["new"].values()
        filename = value["metadata"]["name"]
        assert "/" not in filename
        print(f"Upscaling {filename}")
        input_image = Image.open(io.BytesIO(value["content"])).convert("RGB")
        image_widget.value = value["content"]
        image_widget.width = input_image.size[0]
        image_widget.height = input_image.size[1]


# Fetch models
def fetch(url_or_path):
    if url_or_path.startswith("http:") or url_or_path.startswith("https:"):
        _, ext = os.path.splitext(os.path.basename(url_or_path))
        cachekey = hashlib.md5(url_or_path.encode("utf-8")).hexdigest()
        cachename = f"{cachekey}{ext}"
        if not os.path.exists(f"cache/{cachename}"):
            os.makedirs("tmp", exist_ok=True)
            os.makedirs("cache", exist_ok=True)
            response = requests.get(url_or_path)
            with open(f"tmp/{cachename}", "wb") as f:
                f.write(response.content)
            os.rename(f"tmp/{cachename}", f"cache/{cachename}")
        return f"cache/{cachename}"
    return url_or_path


def resize_image_if_needed(img):
    if img.width != img.height:
        if img.width > img.height:
            img = img.crop(((img.width - img.height) // 2, 0,
                           (img.width + img.height) // 2, img.height))
        else:
            img = img.crop((0, (img.height - img.width) // 2,
                           img.width, (img.height + img.width) // 2))
    print('resized: ', img.width, img.height)
    return img
