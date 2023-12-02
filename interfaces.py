from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index import SimpleDirectoryReader
import os
import openai

def construct_openai_mm_llm(max_new_tokens=2000):
    #get the environment variable openai api token
    OPENAI_API_TOKEN = os.environ.get("OPENAI_API_TOKEN")

    openai_mm_llm = OpenAIMultiModal(
        model="gpt-4-vision-preview", api_key=OPENAI_API_TOKEN, max_new_tokens=max_new_tokens
    )

    return openai_mm_llm

def generate_image(prompt):
    client = openai.OpenAI()

    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1
    )
    return response.data[0].url



def construct_agent_for_prompt_evaluation(max_new_tokens=2000):
        OPENAI_API_TOKEN = os.environ.get("OPENAI_API_TOKEN")

        openai_evaluator_llm = OpenAIMultiModal(
            model="gpt-4-1106-preview", api_key=OPENAI_API_TOKEN, max_new_tokens=max_new_tokens
        )

def evaluate_prompt(openai_evaluator_llm, prompt):
    return openai_evaluator_llm.complete(
        prompt="Sei un esperto prompt engineer. Il tuo scopo è quello di valutare su una scala da 1 a 10 la qualità di un prompt."
    )

def read_all_images(path):
    #read all the images in the directory
    image_reader = SimpleDirectoryReader(
        input_dir=path
    )
    image_documents = image_reader.load_data()

    return image_documents

def inference_on_images(openai_mm_llm, image_documents):
    descriptions ={}
    for image_document in image_documents:
        prompt="Sei un esperto prompt engineer con un forte senso dell'estetica.\
                    Descrivi le immagini con un test che può essere dato in input ad un modello generativo di immagini come DallE3 o Stable Diffusion. La descrizione deve\
                    essere in italiano e deve essere in grado di generare immagini che siano coerenti con la descrizione stessa. La descrizione deve avere anche dettagli tecnici\
                    circa lo stile di realizzazione della nuova immagine generata.\
                    Lo stile dell'immagine da generare deve essere coerente con lo stile dell'immagine in input.\
                    Non fornire alcun tuo parere o alcuna osservazione oltre al prompt.\
                    Il prompt deve essere quanto più dettagliato possibile."
        descriptions[image_document.image_path] = openai_mm_llm.complete(
            prompt=prompt,
            image_documents=image_documents,  # Pass image_document as a keyword argument
            temperature=0.9
        ).text
    return descriptions
