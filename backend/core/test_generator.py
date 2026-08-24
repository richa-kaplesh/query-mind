from core.generator import Generator
from core.tools.pandas_sandbox_tool import PandasSandboxTool

CSV_PATH = "uploads/Titanic-Dataset.csv"

if __name__ == "__main__":
    tools = [PandasSandboxTool(CSV_PATH)]
    bot = Generator(tools=tools)

    # TEST 1: Normal query — LLM should NOT use any tool
    print("--- TEST 1: Normal Query ---")
    query1 = "What is the capital of France?"
    print(f"Q: {query1}")
    result1 = bot.generate_with_tools(query1)
    print(f"A: {result1['answer']}")
    print(f"Tool used: {result1['tool_used']}\n")

    # TEST 2: Titanic question — LLM should USE the pandas tool
    print("--- TEST 2: Titanic Dataset Query ---")
    query2 = f"Using the Titanic dataset at '{CSV_PATH}', what is the survival rate of passengers?"
    print(f"Q: {query2}")
    result2 = bot.generate_with_tools(query2)
    print(f"A: {result2['answer']}")
    print(f"Tool used: {result2['tool_used']}\n")
