# running into a few issues here, base model is not able to understand the tools and the user queries for some reason, looking into it - have to try pro
# issue-1: suddenly asks for sample documents even though there is a tool provided for it
# issue-2: asks for collection name even though there is a tool provided to get all collection names and for it to make its best judgement
import asyncio
import json
from contextlib import AsyncExitStack
from typing import Any, Dict, Optional, List

import nest_asyncio
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from google import genai
from google.genai import types

nest_asyncio.apply()

load_dotenv(".env")

class MCPGoogleClient:
    """Client for interacting with Gemini using MCP tools."""
    
    def __init__(self, model: str = "gemini-2.0-flash"):
        """
            Initialize the Google MCP Client.
            
            Args:
                model (str, optional): The Google model to use. Defaults to "gemini-2.0-pro".
        """
        
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.google_client = genai.Client()
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
            
    # async def get_mcp_tools(self) -> List[types.Tool]:
    #     """
    #         Get available MCP tools.
            
    #         Returns:
    #             List[types.Tool]: A list of available MCP tools in Google format.
    #     """
        
    #     tools_result = await self.session.list_tools()
    #     return [types.Tool(
    #         function_declarations=[
    #             {
    #                 "name": tool.name,
    #                 "description": tool.description,
    #                 "parameters": {
    #                     k: v for k, v in tool.inputSchema.items() 
    #                     if k not in {"additional_properties", "$schema"}
    #                 },
    #             }
    #             for tool in tools_result.tools
    #         ]
    #     )]
    
    async def get_mcp_tools(self) -> List[types.Tool]:
        """
            Get available MCP tools.
            
            Returns:
                List[Dict[str, Any]]: A list of available MCP tools in Gemini format.
        """
        
        def clean_schema_recursive(schema):
            """Recursively clean schema for Google AI compatibility"""
            if isinstance(schema, dict):
                cleaned = {}
                
                for key, value in schema.items():
                    # Skip unsupported fields
                    if key in {"additional_properties", "additionalProperties", "examples"}:
                        continue
                    
                    # Handle any_of by taking the first non-null type
                    if key == "any_of" or key == "anyOf":
                        for option in value:
                            if isinstance(option, dict) and option.get("type") != "null":
                                return clean_schema_recursive(option)
                        continue
                    
                    # Recursively clean nested objects
                    cleaned[key] = clean_schema_recursive(value)
                
                return cleaned
            
            elif isinstance(schema, list):
                return [clean_schema_recursive(item) for item in schema]
            
            return schema
        
        tools_result = await self.session.list_tools()
        
        tools = []
        for tool in tools_result.tools:
            # Clean the entire input schema recursively
            cleaned_schema = clean_schema_recursive(tool.inputSchema)
            
            # Keep only the fields Google AI supports at the top level
            parameters = {
                k: v for k, v in cleaned_schema.items() 
                if k in {"type", "properties", "required", "description", "title", "default", "enum"}
            }
            
            tools.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters,
            })
        
        return [types.Tool(
            function_declarations=tools
        )]
        
    async def process_query(self, query: str, max_iterations: int = 5) -> str:
        """
            Process a query using Gemini and avaliable MCP tools and return the response.
            
            Args:
                query (str): The user query to process.
                
            Returns:
                str: The response from the Gemini model.
        """
        
        # Get available tools
        tools = await self.get_mcp_tools()
        system_instruction = """
            You are an AI assistant with access to tools. 
            You MUST respond with a function_call if the answer requires external knowledge or tool usage. 
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
            When generating function_calls, ENSURE the arguments are serialized as valid JSON.
        """
        contents = [
            types.Content(
                role="user", 
                parts=[types.Part(text=query)]
            ),
        ]
        iteration_count = 0
        # print("\n-------query-------\n", query)
        # Initial Gemini API iterative call
        while iteration_count < max_iterations:
            # Make Gemini API call
            response = await self.google_client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0,
                    tools=tools,
                )
            )
            
            
            # Get assistant's response
            assistant_message = response.candidates[0].content
            print(f"\n-------response (iteration {iteration_count})-------\n", assistant_message)
            
            flag = False
            # Check if there are tool calls
            for part in assistant_message.parts:
                if part.function_call:
                    flag = True
                    break
            if not flag:
                # No more tool calls, return the final response
                return assistant_message.parts[0].text            
            contents.append(assistant_message)
            
            fc_parts_response: List[types.Part] = []
            # Process all tool calls in this iteration
            for part in assistant_message.parts:
                if part.function_call:
                    fc = part.function_call
                    try:
                        fc_response = await self.session.call_tool(
                            fc.name,
                            arguments=json.loads(fc.args) if type(fc.args) != dict else fc.args
                        )            
                        fc_response_content = ""
                        for text_content in fc_response.content:
                            fc_response_content += text_content.text + "\n"
                            
                        if fc_response.isError:
                            part_response = {"error": fc_response_content}
                        else:
                            part_response = {"result": fc_response_content}
                        
                    except Exception as e:
                        # Handle tool execution errors gracefully
                        print(f"Error executing tool {fc.name}: {e}")
                        part_response = {"error": f"Error executing tool: {str(e)}"}
                        
                    # Add tool response to all final fc response
                    fc_parts_response.append(types.Part.from_function_response(name=fc.name, response=part_response))
            
            # Add tool response to final response
            if fc_parts_response:
                contents.append(types.Content(role="function", parts=fc_parts_response))
            # print(f"\n--------messages after tools (iteration {iteration_count})-------\n", messages)
            
            iteration_count += 1
        
        # If we've reached max iterations, make one final call without tools
        print(f"Reached max iterations ({max_iterations}), making final call without tools")
        
        final_response = await self.google_client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            # tool_choice="none"  # Force no tool calls
        )
        
        return final_response.choices[0].message.content
    
    async def chat_loop(self):
        """
            Starts an interactive chat loop with the user.
            
            Waits for user input, processes the query using the Gemini model,
            and prints out the response. If the user types 'quit', the loop exits.
            
            This function is an infinite loop and should be called inside an async context.
        """
        
        print("\nMCP Gemini Client Started! Type your queries or 'quit' to exit.")
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
    
    client = MCPGoogleClient()
    try:
        await client.connect_to_server()
        await client.chat_loop()
    finally:
        await client.cleanup()
    
if __name__ == "__main__":
    asyncio.run(main())
