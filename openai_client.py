import asyncio
import json
from contextlib import AsyncExitStack
from typing import Any, Dict, Optional, List

import nest_asyncio
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI

nest_asyncio.apply()

load_dotenv(".env")

class MCPOpenAIClient:
    """Client for interacting with OpenAI using MCP tools."""
    
    def __init__(self, model: str = "gpt-4o"):
        """
            Initialize the OpenAI MCP Client.
            
            Args:
                model (str, optional): The OpenAI model to use. Defaults to "gpt-4o".
        """
        
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.openai_client = AsyncOpenAI()
        self.model = model
        self.stdio: Optional[Any] = None
        self.write: Optional[Any] = None

    async def connect_to_server(self, server_script_path: str = "server.py"):
        """
            Connect to an MCP server.
            
            Args:
                server_script_path: The path to the server script.
        """
        
        # Server configuration
        server_params = StdioServerParameters(
            command = "python",
            args=[server_script_path]
        )
        
        # Connect to the server
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport        
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        # Initialize connection
        await self.session.initialize()
        
        # List available tools
        tools_result = await self.session.list_tools()
        print("Connected to server with tools:")
        for tool in tools_result.tools:
            print(f"- {tool.name}: {tool.description}")
            
    async def get_mcp_tools(self) -> List[Dict[str, Any]]:
        """
            Get available MCP tools.
            
            Returns:
                List[Dict[str, Any]]: A list of available MCP tools in OpenAI format.
        """
        
        tools_result = await self.session.list_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            }
            for tool in tools_result.tools
        ]
        
    async def process_query(self, query: str, max_iterations: int = 5) -> str:
        """
            Process a query using OpenAI and avaliable MCP tools and return the response.
            
            Args:
                query (str): The user query to process.
                
            Returns:
                str: The response from the OpenAI model.
        """
        
        # Get available tools
        tools = await self.get_mcp_tools()
        
        # system_prompt = """
        #     You are an AI assistant with access to various tools for data retrieval and analysis.
            
        #     TOOL USAGE GUIDELINES:
            
        #     1. **Data Schema Understanding**: 
        #     - Before querying data, always check available collections/tables using schema tools
        #     - Understand the structure of data before making specific queries
        #     - Look for tools like "get_collections", "describe_table", "get_schema" first
            
        #     2. **Query Planning**:
        #     - Break complex queries into steps: schema → data retrieval → analysis
        #     - For database queries, first understand what collections/tables exist
        #     - Then query the specific data you need
        #     - Finally, analyze and summarize the results
            
        #     3. **Tool Call Decision Making**:
        #     - Use tools when you need: external data, real-time information, database queries, file operations
        #     - Don't use tools for: general knowledge, calculations you can do directly, simple explanations
        #     - If unsure about data structure, check schema first
            
        #     4. **Error Handling**:
        #     - If a tool call fails, try alternative approaches
        #     - If data structure is unclear, use schema/discovery tools
        #     - Always explain what went wrong if tools fail
            
        #     5. **Response Quality**:
        #     - Analyze tool results carefully before responding
        #     - Provide context and insights, not just raw data
        #     - If results are empty, explain possible reasons
        #     - Summarize findings in a user-friendly way
            
        #     IMPORTANT: Always start with schema/discovery tools when dealing with databases or unknown data structures.
        # """
        messages = [
            {
                "role": "system", 
                    # 1. Always analyze the results carefully
                "content": """
                    You are an AI assistant with access to tools. 
                    You MUST respond with a tool_call if the answer requires external knowledge or tool usage. 
                    QUERY CONSTRUCTION RULES:
                        1. Always call the tool to retrieve schema information (such as collection field names or sample documents) for the target database and collection.
                        2. Carefully analyze the schema/tool results to identify valid field names and nested paths using dot notation.
                        3. Use only validated fields from the schema to build queries.
                        4. NEVER guess or invent nested field names without confirming them from schema data.
                        5. For case-insensitive searches, use regex: {"address.city": {"$regex": "bangalore", "$options": "i"}}
                    When you receive tool results:
                        1. Extract relevant information 
                        2. Provide a helpful summary to the user
                        3. If results are empty or error, explain what happened. 
                    When generating tool_calls, ENSURE the arguments are serialized as valid JSON.
                """
            },
            {
                "role": "user", 
                "content": query
            },
        ]
        
        iteration_count = 0
    
        # Initial OpenAI API iterative call
        while iteration_count < max_iterations:
            # Make OpenAI API call
            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            
            print(f"\n-------response (iteration {iteration_count})-------\n", response)
            
            # Get assistant's response
            assistant_message = response.choices[0].message
            
            # Add assistant message to conversation
            messages.append(assistant_message)
            
            # Check if there are tool calls
            if not assistant_message.tool_calls:
                # No more tool calls, return the final response
                return assistant_message.content
            
            # Process all tool calls in this iteration
            for tool_call in assistant_message.tool_calls:
                try:
                    result = await self.session.call_tool(
                        tool_call.function.name,
                        arguments=json.loads(tool_call.function.arguments)
                    )
                    content = ""
                    for text_content in result.content:
                        content += text_content.text + "\n"
                    # Add tool response to conversation
                    messages.append({
                        "role": "tool", 
                        "tool_call_id": tool_call.id, 
                        "content": content
                    })
                    
                except Exception as e:
                    # Handle tool execution errors gracefully
                    print(f"Error executing tool {tool_call.function.name}: {e}")
                    messages.append({
                        "role": "tool", 
                        "tool_call_id": tool_call.id, 
                        "content": f"Error executing tool: {str(e)}"
                    })
            
            # print(f"\n--------messages after tools (iteration {iteration_count})-------\n", messages)
            
            iteration_count += 1
        
        # If we've reached max iterations, make one final call without tools
        print(f"Reached max iterations ({max_iterations}), making final call without tools")
        
        final_response = await self.openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            tool_choice="none"  # Force no tool calls
        )
        
        return final_response.choices[0].message.content
    
    async def chat_loop(self):
        """
            Starts an interactive chat loop with the user.
            
            Waits for user input, processes the query using the OpenAI model,
            and prints out the response. If the user types 'quit', the loop exits.
            
            This function is an infinite loop and should be called inside an async context.
        """
        
        print("\nMCP OpenAI Client Started! Type your queries or 'quit' to exit.")
        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == 'quit':
                    print("\nGoodbye!")
                    break
                response = await self.process_query(query)
                print("\nAnswer: " + response)
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
    
    async def cleanup(self):
        """Clean up resources."""
        await self.exit_stack.aclose()
        
        
async def main():
    """"Main entrypoint for the client."""
    
    client = MCPOpenAIClient()
    try:
        await client.connect_to_server()
        await client.chat_loop()
    finally:
        await client.cleanup()
    
if __name__ == "__main__":
    asyncio.run(main())