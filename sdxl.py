#!/usr/bin/env python

import sys
import torch
from imgcat import imgcat
from diffusers import AutoPipelineForText2Image
from diffusers import logging

logging.set_verbosity_error()

# Let's figure out what kind of hardware torch has been built for!
if torch.cuda.is_available():
  pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16")
  device="cuda"
  pipe.to(device)
elif torch.backends.mps.is_available():
  pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo")
  device="mps"
  pipe.to(device)
else:
  pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo")
  device="cpu"

print("\nPipeline configured for " + device + " 🔥")

# configure the prompt
if len(sys.argv) > 1:
	prompt = sys.argv[1]
else:
	prompt = "a flox in a henhouse"

steps = 1

# run the pipeline
image = pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=0.0).images[0]

# save the image
image.save("img.png")
print("\nImage saved to img.png 🎆\n")

# print the image out
imgcat(image)
