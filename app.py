import streamlit as st
from vertexai.vision_models import ImageTextModel

from vertexai.preview.generative_models import (
    GenerationConfig,
    GenerativeModel,
    Image,
    Part,
)
import vertexai
from vertexai.preview.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmCategory,
    HarmBlockThreshold,
    Part,
)
from vertexai.preview.vision_models import (
    Image, 
    ImageGenerationModel
)

import gcs as gcs
import config as config

vertexai.init(project=config.PROJECT_ID, location=config.REGION)

st.set_page_config(layout="wide")

@st.cache_resource
def load_models():
    try:
        text_model_pro = GenerativeModel("gemini-pro")
    except:
        print("ERROR  GenerativeModel(gemini-pro)")
        st.write("ERROR  GenerativeModel(gemini-pro)")
        text_model_pro = None

    try:
        multimodal_model_pro = GenerativeModel("gemini-pro-vision")
    except:
        print("ERROR      GenerativeModel(gemini-pro-vision)")
        st.write("ERROR      GenerativeModel(gemini-pro-vision)")
        multimodal_model_pro = None
    try:
        image_model = ImageGenerationModel.from_pretrained("imagegeneration@005")
    except:
        print("ERROR     ImageGenerationModel.from_pretrained(imagegeneration@005)")
        st.write("ERROR     ImageGenerationModel.from_pretrained(imagegeneration@005)")
        image_model = None

    try:
        
        image2text_model = ImageTextModel.from_pretrained("imagetext@001")
    except:
        print("ERROR     ImageTextModel.from_pretrained(imagetext@001)")
        st.write("ERROR     ImageTextModel.from_pretrained(imagetext@001)")
        image2text_model = None

    return text_model_pro, multimodal_model_pro, image_model, image2text_model


@st.cache_resource
def cache_data():
    from data import Story
    story = Story()
    return story

def get_gemini_pro_text_response(
    model: GenerativeModel,
    contents: str,
    generation_config: GenerationConfig,
    stream: bool = True,
):
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    responses = model.generate_content(
        prompt,
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=stream,
    )

    final_response = []
    for response in responses:
        try:
            # st.write(response.text)
            final_response.append(response.text)
        except IndexError:
            # st.write(response)
            final_response.append("")
            continue
    return " ".join(final_response)


def get_gemini_pro_vision_response(
    model, prompt_list, generation_config={}, stream: bool = True
):
    generation_config = {"temperature": 0.1, "max_output_tokens": 2048}
    responses = model.generate_content(
        prompt_list, generation_config=generation_config, stream=stream
    )
    final_response = []
    for response in responses:
        try:
            final_response.append(response.text)
        except IndexError:
            pass
    return "".join(final_response)

def get_image_2_text_response(
    model, source_bytes, language="en"
):
    from PIL import Image as PIL_image
    from vertexai.vision_models import Image

    source_image = Image(source_bytes)
    captions = model.get_captions(
        image=source_image,
        # Optional:
        number_of_results=2,
        language=language,
    )
    return captions


import data as data

print("init page")

st.header("Vertex AI Demo", divider="rainbow",)
text_model_pro, multimodal_model_pro, image_model, image2text_model = load_models()

story_data = cache_data()

tab3, tab1,   = st.tabs(
    ["Image Playground", "Generate story"  ]
)


with tab1:
    st.write("Using Gemini Pro - Text only model")
    st.subheader("Generate a story")

    # Story premise
    character_name = st.text_input(
        "Enter character name: \n\n", key="character_name", value="Mittens"
    )
    character_type = st.text_input(
        "What type of character is it? \n\n", key="character_type", value="Cat"
    )
    character_persona = st.text_input(
        "What personality does the character have? \n\n",
        key="character_persona",
        value="Mitten is a very friendly cat.",
    )
    character_location = st.text_input(
        "Where does the character live? \n\n",
        key="character_location",
        value="Andromeda Galaxy",
    )
    story_premise = st.multiselect(
        "What is the story premise? (can select multiple) \n\n",
        [
            "Love",
            "Adventure",
            "Mystery",
            "Horror",
            "Comedy",
            "Sci-Fi",
            "Fantasy",
            "Thriller",
        ],
        key="story_premise",
        default=["Love", "Adventure"],
    )
    creative_control = st.radio(
        "Select the creativity level: \n\n",
        ["Low", "High"],
        key="creative_control",
        horizontal=True,
    )
    length_of_story = st.radio(
        "Select the length of the story: \n\n",
        ["Short", "Long"],
        key="length_of_story",
        horizontal=True,
    )

    if creative_control == "Low":
        temperature = 0.30
    else:
        temperature = 0.95

    max_output_tokens = 2048

    prompt = f"""Write a {length_of_story} story based on the following premise: \n
    character_name: {character_name} \n
    character_type: {character_type} \n
    character_persona: {character_persona} \n
    character_location: {character_location} \n
    story_premise: {",".join(story_premise)} \n
    If the story is "short", then make sure to have 5 chapters or else if it is "long" then 10 chapters.
    Important point is that each chapters should be generated based on the premise given above.
    First start by giving the book introduction, chapter introductions and then each chapter. It should also have a proper ending.
    The book should have prologue and epilogue.
    """
    # config = GenerationConfig(
    #     temperature=temperature,
    #     candidate_count=1,
    #     max_output_tokens=max_output_tokens,
    # )

    config_llm = {
        "temperature": 0.8,
        "max_output_tokens": 2048,
    }

    image_prompt = f"""Generate a character based on following parameters: character_name: {character_name} \n
        character_type: {character_type} \n
        character_persona: {character_persona} \n
        character_location: {character_location} \n
        story_premise: {",".join(story_premise)} \n"""


    generate_t2t = st.button("Generate my story", key="generate_t2t")
    if generate_t2t and prompt:
        # st.write(prompt)
        with st.spinner("Generating your story using Gemini..."):
            first_tab1, first_tab2 = st.tabs(["Story", "Prompt" ])
            with first_tab1:
                
                response = get_gemini_pro_text_response(
                    text_model_pro,
                    prompt,
                    generation_config=config_llm,
                )
                if response:
                    st.write("Your story:")
                    st.write(response)
                    story_data.story = response

            with first_tab2:
                st.text(prompt)
         

    generate_t2i = st.button("Generate an illustrative image of the main character", key="generate_t2ti")
    if generate_t2i and image_prompt and story_data.story:
        
        with st.spinner("Generating your storyboard using imagen..."):
            st.write("Your story:")
            st.write(story_data.story)

            first_img_tab1, first_img_tab2 = st.tabs(["Image", "Prompt"])
            with first_img_tab1:
                import text2img as text2img
                images, generate_response = text2img.imagen_generate(image_prompt, image_model )
                
                if images:
                    st.write("Your storyboard:")
                    for image in images:
                        from PIL import Image as PIL_image

                        st.image(image, width=350, caption=f"Your generated character named {character_name}")
                        story_data.image_character = image
                        st.write(image)
            with first_img_tab2:
                st.text(image_prompt)

    generate_t3i = st.button("Generate my storyboard based on character and story", key="generate_t3i")
    if generate_t3i and story_data.story:

        story_parts = story_data.story.split("**")                
        with st.spinner("Generating your storyboard using imagen..."):
            st.write("Your story:")
            for story_part in story_parts:
                st.write(story_part)

                par_image_prompt = f"""Generate an ilustrative scene based on the context below of the character based on following parameters: 
                character_name: {character_name} \n
    character_type: {character_type} \n
    character_persona: {character_persona} \n
    character_location: {character_location} \n
    story_premise: {",".join(story_premise)} \n
    story_scene: {",".join(story_part)} \n
    """

                first_img_tab1, first_img_tab2 = st.tabs(["Image", "Prompt"])
                with first_img_tab1:
                    import text2img as text2img
                    images, generate_response = text2img.imagen_generate(image_prompt, image_model )
                    
                    if images:
                        #st.write("Your storyboard:")
                        for image in images:
                            from PIL import Image as PIL_image

                            st.image(image, width=350, caption=f"Your generated character named {character_name}")
                            story_data.image_character = image
                            st.write(image)



with tab3:
    st.write("Using Gemini Pro Vision - Multimodal model")
    tab_image, tab_image_to_text, tab_image_to_text_to_image = st.tabs(
        [
            "Using Imagen - Text 2 image" , "Using Imagen - image 2 text", "Using Imagen - image 2 text 2 image :-)",             
        ]
    )

    with tab_image:
        st.write("Using Imagen - Text 2 image")
        st.subheader("Generate image")

        image_prompt = "A real-estate luxuary living room"
        image_prompt = st.text_area('Enter your text here...', image_prompt, )

        generate_t2imagen = st.button("Generate an image", key="generate_t2imagen")
        if generate_t2imagen and image_prompt:
            
            with st.spinner("Generating an image using imagen..."):
                
                first_img_tab1, first_img_tab2 = st.tabs(["Image", "Prompt"])
                with first_img_tab1:
                    import text2img as text2img
                    images, generate_response = text2img.imagen_generate(image_prompt, image_model )
                    
                    if images:
                        for image in images:
                            st.image(image, width=350, caption=f"Your generated image")                       
                    else:
                        st.text("Error when generate image")
                        st.text(generate_response)    
                with first_img_tab2:
                    st.text(image_prompt)


    with tab_image_to_text:
        st.write("Using Imagen - image 2 text")
        st.subheader("image decription")

        image_2_describe_to_image = None
        uploaded_image2t2i_file = st.file_uploader("Choose an image file", type= ["jpg", "jpeg"])

        if uploaded_image2t2i_file is not None:
            # To read file as bytes:
            bytes_data = uploaded_image2t2i_file.getvalue()
            st.image(bytes_data, caption='Uploaded image')
            from PIL import Image as PIL_image
            #image_2_describe = PIL_image.Image.frombytes(bytes_data)
            #image_2_describe = Image.from_bytes(bytes_data)
            print("######################################")
            #print(image_2_describe) # = "image/jpeg"
            captions =  get_image_2_text_response(model=image2text_model, source_bytes=bytes_data)
            st.write("Caption from imagen:")
            st.write(captions)

            prompt = "Create a detailed prompt from the image"

            image_part = Part.from_data(bytes_data, mime_type="image/jpeg")
            response_gemini = get_gemini_pro_vision_response(
                model=multimodal_model_pro, prompt_list=[image_part, prompt]
            )
            st.write("Caption from gemini:")
            st.markdown(response_gemini)



        # describe_i2timagen = st.button("Describe an image", key="describe_i2timagen")
        # if describe_i2timagen and image_2_describe: # and image_prompt:
            
        #     with st.spinner("Generating a description of an image using imagen..."):
                
        #         first_img_tab1, first_img_tab2 = st.tabs(["Image", "Prompt"])
        #         with first_img_tab1:
        #             import text2img as text2img
        #             images, generate_response = text2img.imagen_generate(image_prompt, image_model )
                    
        #             if images:
        #                 for image in images:
        #                     st.image(image, width=350, caption=f"Your generated image")                       
        #             else:
        #                 st.text("Error when generate image")
        #                 st.text(generate_response)    
        #         with first_img_tab2:
        #             st.text(image_prompt)
    with tab_image_to_text_to_image:
        st.write("Using Imagen - image 2 text 2 image :-)")
        st.subheader("image description and generation")

        image_2_describe = None
        uploaded_image_to_text_to_image_file = st.file_uploader("Choose an image file", 
                                                                type= ["jpg", "jpeg"], 
                                                                key="uploaded_image_to_text_to_image_file")

        if uploaded_image_to_text_to_image_file is not None:
            # To read file as bytes:
            bytes_data = uploaded_image_to_text_to_image_file.getvalue()
            st.image(bytes_data, caption='Uploaded image')
            from PIL import Image as PIL_image
            #image_2_describe = PIL_image.Image.frombytes(bytes_data)
            #image_2_describe = Image.from_bytes(bytes_data)
            print("######################################")
            #print(image_2_describe) # = "image/jpeg"
            captions =  get_image_2_text_response(model=image2text_model, source_bytes=bytes_data)
            st.write("Caption from imagen:")            
            st.write(captions)

            image_part = Part.from_data(bytes_data, mime_type="image/jpeg")
            prompt = "Create a detailed prompt from the image"

            response_gemini = get_gemini_pro_vision_response(
                model=multimodal_model_pro, prompt_list=[image_part, prompt]
            )
            st.write("Caption from gemini:")
            st.markdown(response_gemini)

            image_prompt =  response_gemini + ", ".join(captions) 
            st.write(image_prompt)

            with st.spinner("Generating an image using imagen..."):
                
                first_img_tab1, first_img_tab2 = st.tabs(["Image", "Prompt"])
                with first_img_tab1:
                    import text2img as text2img
                    images, generate_response = text2img.imagen_generate(image_prompt, image_model )
                    
                    if images:
                        for image in images:
                            st.image(image, width=350, caption=f"Your generated image")                       
                    else:
                        st.text("Error when generate image")
                        st.text(generate_response)
                        
                with first_img_tab2:
                    st.text(image_prompt)

