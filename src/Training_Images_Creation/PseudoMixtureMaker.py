from PIL import Image
import random, glob

# Load all fish image paths
fish_paths = glob.glob("../../data/Tests/shape_fish_crops/*.png")  # adjust path
canvas_size = (4600, 3700)
num_layers = 2
fish_per_layer = 300

bundle = Image.new("RGBA", canvas_size, (0,0,0,0))

for _ in range(num_layers):
    layer = Image.new("RGBA", canvas_size, (0,0,0,0))
    for _ in range(fish_per_layer):
        fish = Image.open(random.choice(fish_paths)).convert("RGBA")

        # Random resize & rotation
        scale = random.uniform(0.5, 1.5)
        fish = fish.resize((int(fish.width*scale), int(fish.height*scale)))
        fish = fish.rotate(random.randint(0, 360), expand=True)

        # Allow partial cropping — fish can go outside frame
        x = random.randint(-fish.width//2, canvas_size[0] - fish.width//2)
        y = random.randint(-fish.height//2, canvas_size[1] - fish.height//2)

        layer.paste(fish, (x, y), fish)

    bundle = Image.alpha_composite(bundle, layer)

# Ensure final image fits canvas exactly (cropping overflow)
bundle = bundle.crop((0, 0, canvas_size[0], canvas_size[1]))
bundle.save("../../data/Tests/pseudo_mixtures/fish_bundle_filled.png")

