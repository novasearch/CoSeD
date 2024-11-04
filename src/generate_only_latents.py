from latents_singleton import Latents
from functools import partial
import os
from sequence_predictor import SoftAttention
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoProcessor, CLIPModel
import json
from diffuser import StableDiffusion


class Clip():
    def __init__(self):
        # Initialize the CLIP model, processor, and tokenizer
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")

    def get_model(self):
        # Return the CLIP model
        return self.model

    def get_preprocess(self):
        # Return the CLIP processor
        return self.processor

    def get_tokenizer(self):
        # Return the CLIP tokenizer
        return self.tokenizer

    def generate_embedding(self, data):
        # Generate embeddings for images or text
        if isinstance(data, Image.Image):
            image_processed = self.processor(images=data, return_tensors="pt").to("cuda")
            image_embedding = self.model.get_image_features(**image_processed)
            return image_embedding

        elif isinstance(data, str):
            data = data[0:77]
            text = self.tokenizer(data, return_tensors="pt").to("cuda")
            text_embedding = self.model.get_text_features(**text)
            return text_embedding


def call_inference_softattention(paths: list[str], previous_images_paths: list[str], previous_texts: list[str],
                                 recipe_id: str, i: int, current_step_text: str):
    print("CURRENT STEP:", i, current_step_text)

    current_images_embedding = []
    current_texts_embedding = []
    previous_images_embedding = []
    previous_texts_embedding = []

    # Extract methods from paths
    previous_methods = [path.split("/")[-2] for path in previous_images_paths]
    if paths[0][-1] == "/":
        methods = [path.split("/")[-2] for path in paths]
    else:
        methods = [path.split("/")[-1] for path in paths]
    current_paths = []

    # Get image and text embeddings for each method
    for index in range(len(paths)):
        print("Method:", methods[index])

        if methods[index] == "sd":
            path = paths[index] + "/" + recipe_id + "_" + str(i) + "_" + str(-1) + ".png"
            if not os.path.exists(path):
                print(path)
                raise FileNotFoundError("Image not found")
            image = Image.open(path)
            image_processed = clip_processor(images=image, return_tensors="pt").to("cuda")
            img_embedding = clip_model.get_image_features(**image_processed)
            text = clip_tokenizer(current_step_text, return_tensors="pt").to("cuda")
            txt_embedding = clip_model.get_text_features(**text)
            previous_texts_embedding.append(txt_embedding)
            previous_images_embedding.append(img_embedding)

            continue

        for previous_recipe_i in range(i):
            if previous_methods[previous_recipe_i] != methods[index]:
                continue
            # Compute the embeddings for the previous step
            previous_step = previous_methods.index(methods[index])
            previous_image = Image.open(previous_images_paths[previous_step])
            previous_image_processed = clip_processor(images=previous_image, return_tensors="pt").to("cuda")
            previous_img_embedding = clip_model.get_image_features(**previous_image_processed)
            previous_images_embedding.append(previous_img_embedding)
            previous_text = previous_texts[previous_step][:77]
            text = clip_tokenizer(previous_text, return_tensors="pt").to("cuda")
            previous_txt_embedding = clip_model.get_text_features(**text)
            previous_texts_embedding.append(previous_txt_embedding)
            break

        current_step_text = current_step_text[:77]
        text = clip_tokenizer(current_step_text, return_tensors="pt").to("cuda")
        temp_txt_embedding = clip_model.get_text_features(**text)

        # Compute the embeddings for the current step with the previous step latents
        for previous_recipe_i in range(i):
            path_to_image = paths[index] + "/" + recipe_id + "_" + str(i) + "_" + str(previous_recipe_i) + ".png"
            current_paths.append(path_to_image)
            if not os.path.exists(path_to_image):
                print(path_to_image)
                raise FileNotFoundError("Image not found")
            current_image = Image.open(path_to_image)
            image_processed = clip_processor(images=current_image, return_tensors="pt").to("cuda")
            temp_img_embedding = clip_model.get_image_features(**image_processed)
            current_images_embedding.append(temp_img_embedding)
            current_texts_embedding.append(temp_txt_embedding)

    # Concatenate the embeddings
    previous_texts_tensor = torch.cat(previous_texts_embedding, dim=0).to("cpu")
    previous_images_tensor = torch.cat(previous_images_embedding, dim=0).to("cpu")
    current_images_tensor = torch.cat(current_images_embedding, dim=0).to("cpu")
    current_texts_tensor = torch.cat(current_texts_embedding, dim=0).to("cpu")

    # Perform inference with the SoftAttention model
    with torch.no_grad():
        softmax, logit = model(current_texts_tensor.unsqueeze(0),
                               current_images_tensor.unsqueeze(0),
                               previous_texts_tensor.unsqueeze(0),
                               previous_images_tensor.unsqueeze(0)
                               )

    # Determine the index to save based on the softmax output
    if len(softmax.shape) == 1:
        index_to_save = softmax.argmax().item()
    else:
        sum_of_probs = torch.sum(softmax, dim=-1)
        index_to_save = sum_of_probs.argmax().item()
    return index_to_save, current_paths[index_to_save]


def capture_latents(latents_store: Latents, step: int, timestep: int, latents: torch.FloatTensor):
    if DEBUG:
        print("Inside callback")
        print("Step:", step)
        print("Timestep:", timestep)
        print("Latents Shape:", latents.shape)
    latents_store.add_latents(latents)
    return


def generate_with_latents(generated_steps, recipe_id, selected_latents, default_to_fixed_seed=True):
    latents_store_array = []
    stats = []
    image_name_dict = {}
    paths_selected_latents = []

    # Map latents to their corresponding paths
    latents_to_path = {-1: "sd", 1: "latent_1", 2: "latent_2", 3: "latent_3", 4: "latent_4", 5: "latent_5",
                       6: "latent_6", 7: "latent_7", 8: "latent_8", 9: "latent_9", 10: "latent_10", 20: "latent_20",
                       40: "latent_40"}
    paths_for_softattention = []
    for latent in selected_latents:
        paths_for_softattention.append(
            "{}{}".format(PATH, latents_to_path[latent]))

    # Create a fixed seed generator if required
    if default_to_fixed_seed:
        generator = torch.Generator("cuda").manual_seed(int(recipe_id))
    else:
        generator = None

    previous_texts = []
    previous_image_methods = []
    for i, step in enumerate(generated_steps):
        print("\nRecipe", recipe_id, "Step", i)
        latents_store = Latents()

        # First step of the task
        if i == 0:
            path_without_filename = f"{PATH}sd/"
            path = f"{PATH}sd/" + "{}_{}_{}.png".format(recipe_id, i, -1)
            previous_image_methods.append(path)
            previous_texts.append(step)
            latents_stores = []
            images = []
            for i in range(4):
                f = partial(capture_latents, latents_store)
                image = sd.call(step, generator=None, callback=f, callback_steps=1)
                latents_stores.append(latents_store.dump_and_clear())
                images.append(image)
            inputs = clip_processor(text=step, images=images, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=0)
            print("Probs")
            print(probs)
            print(probs.argmax().item())

            selected_latent_store = latents_stores[probs.argmax().item()]
            image = images[probs.argmax().item()]
            latents_store_array.append(selected_latent_store)
            image_name_dict["{}_{}_{}.png".format(recipe_id, i, -1)] = image
            if not os.path.exists(path_without_filename):
                print("Creating directory")
                os.makedirs(path_without_filename)
            image.save(path)
            splitted_path = path.split("/")
            stored_path = splitted_path[-2] + "/" + splitted_path[-1]
            paths_selected_latents.append(stored_path)
            continue

        temp_latent_store_array = []
        # Use all the previous latents to generate the next step
        print("Previous steps", i)
        for previous_step in range(i):

            for latent in selected_latents:
                print("Step", i, "Latent", latent, "Previous Step", previous_step)
                f = partial(capture_latents, latents_store)

                # Normal generation
                if latent == -1:
                    path = f"{PATH}sd/" + "{}_{}_{}.png".format(recipe_id, i, -1)
                    if os.path.exists(path):
                        print("Skipping step", i, "latent", latent, "previous step", previous_step)
                        continue
                    image = sd.call(step, generator=generator, callback=f, callback_steps=1)
                    temp_latent_store_array.append(latents_store.dump_and_clear())
                    image_name_dict["{}_{}_{}.png".format(recipe_id, i, -1)] = image

                # Latent Conditioned Generation
                else:
                    path_without_filename = "{}latent_{}/".format(PATH, latent)
                    path = "{}latent_{}/".format(PATH, latent) + "{}_{}_{}.png".format(
                        recipe_id, i, previous_step)
                    previous_latents = latents_store_array[previous_step][latent]
                    image = sd.call(step, latents=previous_latents, callback=f, callback_steps=1)
                    temp_latent_store_array.append(latents_store.dump_and_clear())
                    image_name_dict["{}_{}_{}.png".format(recipe_id, i, previous_step)] = image

                # Save the image
                if not os.path.exists(path_without_filename):
                    print("Creating directory")
                    os.makedirs(path_without_filename)
                image.save(path)

        print("Calling inference of softattention")
        path_index, path_selected = call_inference_softattention(paths_for_softattention,  # paths
                                                                 previous_image_methods,  # previous_images_method
                                                                 previous_texts,  # previous_texts
                                                                 recipe_id,  # recipe_id
                                                                 i,  # i
                                                                 step  # current_step_text
                                                                 )
        latents_store_array.append(temp_latent_store_array[path_index])
        # Save the path of the selected latent
        previous_image_methods.append(path_selected)
        previous_texts.append(step)
        splitted_path = path_selected.split("/")
        stored_path = splitted_path[-2] + "/" + splitted_path[-1]
        paths_selected_latents.append(stored_path)

    return paths_selected_latents, stats


if __name__ == '__main__':
    # Load the SoftAttention model from checkpoint
    model = SoftAttention.load_from_checkpoint(
        "./9q3eu8vi/checkpoints/epoch=7-step=15.ckpt")

    model.eval()

    # Initialize CLIP components
    clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")

    # Define constants
    STEPS_FILE = "./captions.json"
    PATH = "./images/"
    DEBUG = False
    NUM_RECIPES = 100
    selected_latents = [-1, 1, 2, 3]

    # Initialize StableDiffusion
    sd = StableDiffusion()

    # Load recipes from JSON file
    with open(STEPS_FILE, "r") as f:
        recipes = json.load(f)

    name_to_save = "recipe_image_selection_all.json"
    recipes = list(recipes.items())[:min(NUM_RECIPES, len(recipes))]

    recipe_image_selection = {}
    for recipe_id, recipe in recipes:
        real_steps = recipe["steps"]
        generated_steps = recipe["steps_generated"]
        sd.set_use_negative_prompts(True)

        print("Generating images for recipe", recipe_id, "with", len(generated_steps), "steps")

        img_dict, stats = generate_with_latents(generated_steps,
                                                recipe_id,
                                                selected_latents,
                                                default_to_fixed_seed=True)
        recipe_image_selection[recipe_id] = img_dict

    # Save the recipe image selection to a JSON file
    json.dump(recipe_image_selection,
              open("{}{}".format(PATH, name_to_save), "w"))
