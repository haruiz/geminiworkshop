import tenacity
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
import vertexai
from vertexai.preview import reasoning_engines

safety_settings = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

model_kwargs = {
    # temperature (float): The sampling temperature controls the degree of
    # randomness in token selection.
    "temperature": 0.28,
    # max_output_tokens (int): The token limit determines the maximum amount of
    # text output from one prompt.
    "max_output_tokens": 1000,
    # top_p (float): Tokens are selected from most probable to least until
    # the sum of their probabilities equals the top-p value.
    "top_p": 0.95,
    # top_k (int): The next token is selected from among the top-k most
    # probable tokens.
    "top_k": 40,
    # safety_settings (Dict[HarmCategory, HarmBlockThreshold]): The safety
    # settings to use for generating content.
    "safety_settings": safety_settings,
}


def sum(a: int, b: int):
    """Adds two numbers together.

    Args:
        a: The first number.
        b: The second number.

    Returns:
        int: The sum of the two numbers.
    """
    return a + b


def multiply(a: int, b: int):
    """Multiplies two numbers together.

    Args:
        a: The first number.
        b: The second number.

    Returns:
        int: The product of the two numbers.
    """
    return a * b


class Agent:
    """
    The Agent class is a wrapper around the reasoning engine that allows you to interact with it.
    """

    def __init__(self, model, tools, model_kwargs):
        self.model = model
        self.tools = tools
        self.agent = reasoning_engines.LangchainAgent(
            model=self.model,  # Required.
            tools=self.tools,  # Optional. List of functions to be used in the reasoning engine.)
            model_kwargs=model_kwargs,  # Optional.
        )

    @tenacity.retry(wait=tenacity.wait_fixed(2), stop=tenacity.stop_after_attempt(3))
    def query(self, input: str):
        """Get the response from the reasoning engine.

        Args:
            input: The input to the reasoning engine.

        Returns:
            str: The response from the reasoning engine.
        """
        response = self.agent.query(input=input)
        return response

    @tenacity.retry(wait=tenacity.wait_fixed(2), stop=tenacity.stop_after_attempt(3))
    def deploy(self, agent_name: str, requirements: list = None):
        """Deploy the reasoning engine.

        Returns:
            str: The resource name of the deployed reasoning engine.
        """
        agent_requirements = [
            "google-cloud-aiplatform[reasoningengine,langchain]",
        ]
        if requirements is None:
            agent_requirements.extend(requirements)

        remote_app = reasoning_engines.ReasoningEngine.create(
            reasoning_engines.LangchainAgent(
                model=MODEL,  # Required.
                tools=[
                    sum,
                    multiply,
                ],  # Optional. List of functions to be used in the reasoning engine.)
                model_kwargs=model_kwargs,  # Optional.
            ),
            requirements=agent_requirements,
            display_name=agent_name,
        )
        return remote_app.resource_name

    @tenacity.retry(wait=tenacity.wait_fixed(2), stop=tenacity.stop_after_attempt(3))
    def list(self):
        """List the deployed reasoning engines.

        Returns:
            list: The list of deployed reasoning engines.
        """
        apps = reasoning_engines.ReasoningEngine.list()
        return apps

    @tenacity.retry(wait=tenacity.wait_fixed(2), stop=tenacity.stop_after_attempt(3))
    def delete(self):
        """Delete the deployed reasoning engine.

        Args:
            resource_name: The resource name of the deployed reasoning engine.
        """
        apps = reasoning_engines.ReasoningEngine.list()
        for app in apps:
            remote_app = reasoning_engines.ReasoningEngine(app.resource_name)
            remote_app.delete()


if __name__ == "__main__":

    PROJECT_ID = "build-with-ai-project"
    LOCATION = "us-central1"
    STAGING_BUCKET = "gs://build-with-ai-project-vertexai"
    MODEL = "gemini-1.0-pro"

    vertexai.init(project=PROJECT_ID, location=LOCATION, staging_bucket=STAGING_BUCKET)

    agent = Agent(model=MODEL, tools=[sum, multiply], model_kwargs=model_kwargs)

    response = agent.query(
        input="You have 3 apples and 2 oranges. "
        "How many fruits do you have in total?. Multiply the results of the operation by 2."
    )
    print(response)
