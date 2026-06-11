from smolagents import CodeAgent, WebSearchTool, InferenceClientModel, OpenAIModel
import os
from dotenv import load_dotenv

load_dotenv(".env.local")


model = InferenceClientModel()



region = os.getenv("AWS_REGION", "us-east-1")

nemotron_super = OpenAIModel(
    model_id="nvidia.nemotron-super-3-120b",
    api_key=os.getenv("BEDROCK_API_KEY"),          # Bearer token from Bedrock console
    api_base=f"https://bedrock-mantle.{region}.api.aws/v1",
    temperature=0.7,                               # ChatOpenAI supports this directly
)

agent = CodeAgent(tools=[WebSearchTool()], model = nemotron_super, stream_outputs = True)

agent.run("How many hours would it take for a dog at full speed to run through Delhi to Mumbai")
