# ### Creating Welcome Image with dall-e-3

import base64
from io import BytesIO
from PIL import Image
import os
from openai import OpenAI

def welcome_image():
    """Function to generate the welcome image in the chatbot making reference to the Booking experience."""
    
    image_name = "welcome_image_booking.png"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, image_name)
    openai_client = OpenAI()

    if os.path.exists(image_path):
        image = Image.open(image_path)

    else:
        image_response = openai_client.images.generate(
                model="dall-e-3",
                prompt=f"""Create a photorealistic welcome image with the text 
                "Welcome!"
                in a clear, elegant, and formal font.             
                In the center, there's a modern glass hut. 
                In front of the hut, a couple is peacefully sitting on the ground, gently illuminated by sun rays 
                filtering through the canopy. Around the border of the image, the subtle silhouettes of curious 
                jungle animals are partially visible, watching the couple with interest. 
                The overall mood is serene and inviting.
                The background features a lush jungle with vivid greenery.
                """,
                size="1024x1024",
                quality="standard",
                n=1,
                response_format="b64_json",
                style="vivid"       
            )
        image_base64 = image_response.data[0].b64_json
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data))

        # Save the image
        image.save(image_path, format="PNG")
        print("Image generated and saved.")
    # Display the image
    # display(image)
    
    return image