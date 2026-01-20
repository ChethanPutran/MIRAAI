from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image


folder = __name__
# Load image
img_path = r'data\img_left.jpg'

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

image = Image.open(img_path).convert("RGB")
pixel_values = processor(images=image, return_tensors="pt").pixel_values
output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
