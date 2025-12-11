import base64

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode("utf-8")
    return encoded

# Usage
path = r"C:\Users\Hemanth Kumar\OneDrive\Desktop\cirhossis.jpg"
b64_string = image_to_base64(path)
print(b64_string[:200], "...")  # printing entire string will fry your terminal
