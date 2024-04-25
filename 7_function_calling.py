from rich.console import Console
import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())

try:
    # Used to securely store your API key
    from google.colab import userdata

    # Or use `os.getenv('API_KEY')` to fetch an environment variable.
    GOOGLE_API_KEY = userdata.get("GOOGLE_API_KEY")
except ImportError:
    import os

    GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]

genai.configure(api_key=GOOGLE_API_KEY)

console = Console()


def basic_example():
    """
    Basic example of using function calling with Gemini
    :return:
    """

    # Create a model
    def multiply(a: float, b: float):
        """returns a * b."""
        return a * b

    def divide(a: float, b: float):
        """returns a / b."""
        return a / b

    model = genai.GenerativeModel(model_name="gemini-pro", tools=[multiply, divide])
    console.print(model)
    chat = model.start_chat(enable_automatic_function_calling=True)

    response = chat.send_message(
        "I have 57 cats, each owns 44 mittens, how many mittens is that in total?"
    )
    console.print(response.text)
    for content in chat.history:
        part = content.parts[0]
        print(content.role, "->", type(part).to_dict(part))
        print("-" * 80)


def advanced_example():
    """
    Advanced example of using function calling with Gemini
    :return:
    """
    import google.ai.generativelanguage as glm

    calculator = glm.Tool(
        function_declarations=[
            glm.FunctionDeclaration(
                name="multiply",
                description="Returns the product of two numbers.",
                parameters=glm.Schema(
                    type=glm.Type.OBJECT,
                    properties={
                        "a": glm.Schema(type=glm.Type.NUMBER),
                        "b": glm.Schema(type=glm.Type.NUMBER),
                    },
                    required=["a", "b"],
                ),
            )
        ]
    )
    print(glm.Tool(calculator))
    model = genai.GenerativeModel("gemini-pro", tools=calculator)
    chat = model.start_chat()
    response = chat.send_message(
        f"What's 234551 X 325552 ?",
    )
    console.print(response)


if __name__ == "__main__":
    advanced_example()
