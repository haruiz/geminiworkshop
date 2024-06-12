import typing
from pathlib import Path

import gradio as gr
import tenacity
import vertexai
from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from rich.console import Console
from vertexai.generative_models import (
    FunctionDeclaration,
    GenerationConfig,
    GenerativeModel,
    Tool,
)
from vertexai.preview.vision_models import ImageGenerationModel

from utils import get_image_bytes_base64

load_dotenv(find_dotenv())
console = Console()


class PICASSO:
    """
    The PICASSO class is a wrapper around the ImageGenerationModel and ChatGoogleGenerativeAI models.
    """

    def __init__(self, images_folder: typing.Union[str, Path] = "images"):
        """
        Initialize the PICASSO class with the required models.
        """
        self.images_folder = Path(images_folder)
        self.images_folder.mkdir(exist_ok=True)

        self.image_gen_model = ImageGenerationModel.from_pretrained(
            "imagegeneration@006"
        )
        self.poem_gen_model = ChatGoogleGenerativeAI(model="gemini-pro-vision")

    @tenacity.retry(wait=tenacity.wait_fixed(2), stop=tenacity.stop_after_attempt(3))
    def generate_images(
        self, prompt: str, number_of_images: typing.Union[int, float] = 4
    ):
        """
        Generates an image based on the provided prompt.

        Args:
            prompt (str): The text to generate the image from.
            number_of_images (int): The number of images to generate.

        Returns:
            str: The path to the generated image.
        """
        try:
            # it uses Imagen2 under the hood
            if prompt is None:
                raise ValueError("prompt cannot be None")

            if self.images_folder.exists():
                for image in self.images_folder.iterdir():
                    image.unlink()

            console.log(
                f"Generating {int(number_of_images)} images based on the prompt: {prompt}"
            )

            response = self.image_gen_model.generate_images(
                prompt=prompt,
                number_of_images=int(number_of_images),
                add_watermark=True,
            )

            for i, image in enumerate(response.images):
                image.save(str(self.images_folder / f"image_{i}.jpg"))
        except Exception as e:
            console.log(f"An error occurred: {e}")
            raise e

    @tenacity.retry(wait=tenacity.wait_fixed(2), stop=tenacity.stop_after_attempt(3))
    def write_poem(self, prompt: str) -> str:
        """
        generates a poem based on the provided prompt and images.

        Args:
            prompt (str): The prompt to use for generating the poem.

        Returns:
            str: The text generated from the image.
        """
        try:
            if prompt is None:
                raise ValueError("prompt cannot be None")

            images = list(Path(self.images_folder).iterdir())
            if len(images) == 0:
                raise ValueError("No images found in the images folder.")

            image_parts = [
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{get_image_bytes_base64(image_path)}",
                }
                for image_path in images
            ]
            text_part = {
                "type": "text",
                "text": "write a poem based on the prompt {prompt} and the images.",
            }
            messages = [
                SystemMessage(content="you are an artist!"),
                HumanMessage(content=[text_part] + image_parts),
            ]
            chain = (
                ChatPromptTemplate.from_messages(messages)
                | self.poem_gen_model
                | StrOutputParser()
            )
            ai_message = chain.invoke({"prompt": prompt})
            return ai_message
        except Exception as e:
            console.log(f"An error occurred: {e}")
            raise e

    def get_tools_declaration(self):
        """
        Returns the tools available for the PICASSO class.
        """
        generate_images_tool = FunctionDeclaration(
            name="generate_images",
            description="Generates an image based on the provided prompt.",
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "number_of_images": {
                        "type": "integer",
                        "description": "The number of images to generate as int.",
                    },
                },
                "required": ["prompt"],
            },
        )

        write_poem_tool = FunctionDeclaration(
            name="write_poem",
            description="Generates a poem based on the provided prompt and images.",
            parameters={
                "type": "object",
                "properties": {"prompt": {"type": "string"}},
            },
        )
        return [Tool(function_declarations=[generate_images_tool, write_poem_tool])]

    def chat(self):
        """
        Generates a poem based on the provided prompt.

        Args:
            prompt (str): The prompt to use for generating the poem.
        """
        try:
            chat_model = GenerativeModel(
                model_name="gemini-1.0-pro-001",
                generation_config=GenerationConfig(temperature=0),
                tools=self.get_tools_declaration(),
            )

            chat = chat_model.start_chat()

            def echo(history, message):
                """
                This function echoes the message back to the user.
                :param history:
                :param message:
                :return:
                """
                for x in message["files"]:
                    # (filepath, alt_text)
                    history.append(((x,), None))
                if message["text"] is not None and message["text"] != "":
                    # (text, alt_text)
                    history.append((message["text"], None))
                # we  return the history and a MultimodalTextbox component to render the chat elements
                return history, gr.MultimodalTextbox(value=None, interactive=False)

            def bot(history):
                """
                This function calls the bot to respond to the user message.
                :param history:
                :return:
                """
                message = history[-1][0]
                if isinstance(message, tuple):
                    history.append(("Not a valid message", None))
                    return history

                response = chat.send_message(message)
                message_part = response.candidates[0].content.parts[0]
                if message_part.function_call:
                    function_name = message_part.function_call.name
                    function_args = message_part.function_call.args
                    function_args = {
                        arg_name: arg_val for arg_name, arg_val in function_args.items()
                    }
                    console.print(
                        f"Calling function: {function_name} with args: {function_args}"
                    )
                    match function_name:
                        case "generate_images":
                            self.generate_images(**function_args)
                            images = list(self.images_folder.iterdir())
                            for image in images:
                                history.append(((str(image),), None))
                        case "write_poem":
                            poem = self.write_poem(**function_args)
                            history.append((poem, None))
                else:
                    history.append((message_part.text, None))
                return history

            CSS = """
            .contain { display: flex; flex-direction: column; }
            .gradio-container { height: 100vh !important; }
            #component-0 { height: 100%; }
            #chatbot { flex-grow: 1; overflow: auto;}
            """

            with gr.Blocks(css=CSS) as demo:
                chat_input = gr.MultimodalTextbox(
                    interactive=True,
                    file_types=["image"],
                    placeholder="Enter message or upload file...",
                    show_label=False,
                )
                # inputs: List of gradio.components to use as inputs
                # outputs: List of gradio.components to use as outputs
                chatbot = gr.Chatbot(elem_id="chatbot")
                chat_msg = chat_input.submit(
                    echo, [chatbot, chat_input], [chatbot, chat_input]
                )
                # when the chatbot responds, it will call the bot function
                bot_msg = chat_msg.then(bot, chatbot, chatbot, api_name="bot_response")
                # when the bot responds, it will render messages in the chatbot
                bot_msg.then(
                    lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input]
                )
            demo.queue()
            demo.launch()
        except Exception as e:
            console.log(f"An error occurred: {e}")
            raise e


if __name__ == "__main__":
    PROJECT_ID = "build-with-ai-project"
    LOCATION = "us-central1"
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    # imagen2 = ImageGenerationModel.from_pretrained("imagegeneration@006")
    # gemini = ChatGoogleGenerativeAI(model="gemini-pro-vision")
    #
    # prompt = "My lovely pomeranian playing in the beach, looking at the sunset, smiling as always."
    # images_folder = "images"
    #
    # images = generate_images(
    #     prompt=prompt, number_of_images=4, images_folder=images_folder
    # )
    #
    # text = write_poem(
    #     prompt=prompt,
    #     images_folder="images",
    # )

    app = PICASSO(images_folder="images")
    app.chat()
