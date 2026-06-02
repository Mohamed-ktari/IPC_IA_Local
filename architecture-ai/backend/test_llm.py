# test_llm.py — run this directly to verify the LLM layer works
# Usage: cd backend && python test_llm.py
# Delete this file once the LLM layer is confirmed working

from app.llm.factory import get_llm
from app.llm.base import Message

def test_chat():
    print("Testing chat()...")
    llm = get_llm()
    messages = llm.build_messages(
        system_prompt="You are a helpful assistant for an architecture firm. Answer concisely.",
        user_message="What is a CCTP document in French construction? One sentence.",
    )
    response = llm.chat(messages, temperature=0.3, max_tokens=100)
    print(f"Provider : {response.provider}")
    print(f"Model    : {response.model}")
    print(f"Response : {response.content}")
    print(f"Tokens   : {response.prompt_tokens} prompt / {response.completion_tokens} completion")

def test_stream():
    print("\nTesting stream()...")
    llm = get_llm()
    messages = llm.build_messages(
        system_prompt="You are a helpful assistant. Answer concisely.",
        user_message="Say hello in French.",
    )
    print("Streamed: ", end="", flush=True)
    for token in llm.stream(messages, max_tokens=50):
        print(token, end="", flush=True)
    print()

##################### AGENT ############################
def test_base_agent():
    print("\nTesting BaseAgent...")

    # Quick concrete subclass just for testing
    from app.agents.base_agent import BaseAgent, AgentResponse

    class TestAgent(BaseAgent):
        agent_type = "analysis"
        description = "Test agent"

        def run(self, message: str) -> AgentResponse:
            return self.chat(message, max_tokens=80)

    agent = TestAgent()
    print(f"Prompt loaded: {'yes' if agent.system_prompt else 'no'}")

    response = agent.run("What is a diagnostic amiante? One sentence.")
    print(f"Agent type : {response.agent_type}")
    print(f"Duration   : {response.duration_seconds}s")
    print(f"Response   : {response.content}")



if __name__ == "__main__":
    test_chat()
    test_stream()
    test_base_agent()