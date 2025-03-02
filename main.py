from typing import Any, Dict, List, Optional
import httpx
from mcp.server.fastmcp import FastMCP
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("api-tester")

# Constants
USER_AGENT = "api-tester/1.0"

# Read the OpenAI API key from environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Print a message about API key status at startup (without revealing the key)
if OPENAI_API_KEY:
    print("Using OpenAI API key from environment variable")
else:
    print("WARNING: No OpenAI API key found. OpenAI tools will not work. Set OPENAI_API_KEY environment variable.")


async def make_request(url: str, method: str, headers: Optional[Dict[str, str]] = None, 
                      params: Optional[Dict[str, str]] = None, 
                      json_body: Optional[Dict[str, Any]] = None,
                      timeout: float = 30.0) -> Dict[str, Any]:
    """Make an HTTP request with proper error handling."""
    if headers is None:
        headers = {}
    
    # Set default User-Agent if not provided
    if "User-Agent" not in headers:
        headers["User-Agent"] = USER_AGENT
    
    async with httpx.AsyncClient() as client:
        try:
            # Prepare kwargs based on the HTTP method
            kwargs = {
                "headers": headers,
                "timeout": timeout
            }
            
            # Add params for all methods
            if params:
                kwargs["params"] = params
                
            # Add JSON body only for methods that support it
            if json_body and method.lower() in ["post", "put", "patch"]:
                kwargs["json"] = json_body
            
            # Make the request with appropriate arguments
            response = await getattr(client, method.lower())(url, **kwargs)
            
            # Try to parse as JSON first
            try:
                json_response = response.json()
                return {
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "body": json_response,
                    "is_json": True
                }
            except Exception:
                # If not JSON, return as text
                return {
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "body": response.text,
                    "is_json": False
                }
                
        except Exception as e:
            return {
                "error": str(e),
                "status_code": None,
                "headers": None,
                "body": None,
                "is_json": False
            }


@mcp.tool()
async def get_request(url: str, headers: Optional[Dict[str, str]] = None, 
                     params: Optional[Dict[str, str]] = None) -> str:
    """Make a GET request to the specified URL.
    
    Args:
        url: The URL to send the request to
        headers: Optional dictionary of HTTP headers
        params: Optional dictionary of query parameters
    """
    result = await make_request(url, "GET", headers, params)
    
    if "error" in result:
        return f"Error: {result['error']}"
    
    # Format the response
    formatted_response = f"Status Code: {result['status_code']}\n\n"
    
    # Add headers (limited to important ones)
    important_headers = ["content-type", "content-length", "server", "date"]
    header_section = "Headers:\n"
    for key, value in result["headers"].items():
        if key.lower() in important_headers:
            header_section += f"  {key}: {value}\n"
    formatted_response += header_section + "\n"
    
    # Add body
    if result["is_json"]:
        formatted_response += "Body (JSON):\n"
        formatted_response += json.dumps(result["body"], indent=2)
    else:
        formatted_response += "Body:\n"
        # Limit text length to avoid overwhelming responses
        body_text = result["body"]
        if len(body_text) > 1000:
            body_text = body_text[:1000] + "... (truncated)"
        formatted_response += body_text
    
    return formatted_response


@mcp.tool()
async def post_request(url: str, json_body: Dict[str, Any], 
                      headers: Optional[Dict[str, str]] = None) -> str:
    """Make a POST request with a JSON body to the specified URL.
    
    Args:
        url: The URL to send the request to
        json_body: Dictionary to be sent as JSON in the request body
        headers: Optional dictionary of HTTP headers
    """
    result = await make_request(url, "POST", headers, json_body=json_body)
    
    if "error" in result:
        return f"Error: {result['error']}"
    
    # Format the response
    formatted_response = f"Status Code: {result['status_code']}\n\n"
    
    # Add headers (limited to important ones)
    important_headers = ["content-type", "content-length", "server", "date"]
    header_section = "Headers:\n"
    for key, value in result["headers"].items():
        if key.lower() in important_headers:
            header_section += f"  {key}: {value}\n"
    formatted_response += header_section + "\n"
    
    # Add body
    if result["is_json"]:
        formatted_response += "Body (JSON):\n"
        formatted_response += json.dumps(result["body"], indent=2)
    else:
        formatted_response += "Body:\n"
        # Limit text length to avoid overwhelming responses
        body_text = result["body"]
        if len(body_text) > 1000:
            body_text = body_text[:1000] + "... (truncated)"
        formatted_response += body_text
    
    return formatted_response


@mcp.tool()
async def put_request(url: str, json_body: Dict[str, Any], 
                     headers: Optional[Dict[str, str]] = None) -> str:
    """Make a PUT request with a JSON body to the specified URL.
    
    Args:
        url: The URL to send the request to
        json_body: Dictionary to be sent as JSON in the request body
        headers: Optional dictionary of HTTP headers
    """
    result = await make_request(url, "PUT", headers, json_body=json_body)
    
    if "error" in result:
        return f"Error: {result['error']}"
    
    # Format the response similar to other methods
    formatted_response = f"Status Code: {result['status_code']}\n\n"
    
    important_headers = ["content-type", "content-length", "server", "date"]
    header_section = "Headers:\n"
    for key, value in result["headers"].items():
        if key.lower() in important_headers:
            header_section += f"  {key}: {value}\n"
    formatted_response += header_section + "\n"
    
    if result["is_json"]:
        formatted_response += "Body (JSON):\n"
        formatted_response += json.dumps(result["body"], indent=2)
    else:
        formatted_response += "Body:\n"
        body_text = result["body"]
        if len(body_text) > 1000:
            body_text = body_text[:1000] + "... (truncated)"
        formatted_response += body_text
    
    return formatted_response


@mcp.tool()
async def delete_request(url: str, headers: Optional[Dict[str, str]] = None) -> str:
    """Make a DELETE request to the specified URL.
    
    Args:
        url: The URL to send the request to
        headers: Optional dictionary of HTTP headers
    """
    result = await make_request(url, "DELETE", headers)
    
    if "error" in result:
        return f"Error: {result['error']}"
    
    # Format the response
    formatted_response = f"Status Code: {result['status_code']}\n\n"
    
    important_headers = ["content-type", "content-length", "server", "date"]
    header_section = "Headers:\n"
    for key, value in result["headers"].items():
        if key.lower() in important_headers:
            header_section += f"  {key}: {value}\n"
    formatted_response += header_section + "\n"
    
    if result["is_json"]:
        formatted_response += "Body (JSON):\n"
        formatted_response += json.dumps(result["body"], indent=2)
    else:
        formatted_response += "Body:\n"
        body_text = result["body"]
        if len(body_text) > 1000:
            body_text = body_text[:1000] + "... (truncated)"
        formatted_response += body_text
    
    return formatted_response


@mcp.tool()
async def openai_chat_completion(prompt: str, system_message: Optional[str] = None, 
                                model: str = "gpt-3.5-turbo", temperature: float = 0.7) -> str:
    """Make a request to OpenAI's chat completion API.
    
    Args:
        prompt: The user message to send to the model
        system_message: Optional system message to set the behavior of the assistant
        model: The OpenAI model to use (default: gpt-3.5-turbo)
        temperature: Controls randomness (0-1, lower is more deterministic)
    """
    url = "https://api.openai.com/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    
    messages.append({"role": "user", "content": prompt})
    
    json_body = {
        "model": model,
        "messages": messages,
        "temperature": temperature
    }
    
    result = await make_request(url, "POST", headers, json_body=json_body)
    
    if "error" in result:
        return f"Error calling OpenAI API: {result['error']}"
    
    if result["is_json"]:
        try:
            response_content = result["body"]["choices"][0]["message"]["content"]
            tokens_used = result["body"].get("usage", {})
            
            formatted_response = f"Response from {model}:\n\n{response_content}\n\n"
            formatted_response += f"Tokens used: {tokens_used.get('total_tokens', 'unknown')}"
            
            return formatted_response
        except (KeyError, IndexError) as e:
            return f"Error parsing OpenAI response: {str(e)}\nRaw response: {json.dumps(result['body'], indent=2)}"
    else:
        return f"Unexpected response format from OpenAI API: {result['body']}"


@mcp.tool()
async def openai_image_generation(prompt: str, size: str = "1024x1024", n: int = 1) -> str:
    """Generate images using OpenAI's DALL-E model.
    
    Args:
        prompt: The description of the image to generate
        size: Image size (256x256, 512x512, or 1024x1024)
        n: Number of images to generate (1-10)
    """
    url = "https://api.openai.com/v1/images/generations"
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    json_body = {
        "prompt": prompt,
        "n": min(n, 10),  # Limit to 10 images max
        "size": size
    }
    
    result = await make_request(url, "POST", headers, json_body=json_body)
    
    if "error" in result:
        return f"Error calling OpenAI Image API: {result['error']}"
    
    if result["is_json"]:
        try:
            image_urls = [data["url"] for data in result["body"]["data"]]
            formatted_response = f"Generated {len(image_urls)} image(s):\n\n"
            for i, url in enumerate(image_urls, 1):
                formatted_response += f"Image {i}: {url}\n\n"
            return formatted_response
        except (KeyError, IndexError) as e:
            return f"Error parsing OpenAI response: {str(e)}\nRaw response: {json.dumps(result['body'], indent=2)}"
    else:
        return f"Unexpected response format from OpenAI API: {result['body']}"


def main():
    print("API Tester MCP Server starting...")
    # Initialize and run the server
    mcp.run(transport='stdio')


if __name__ == "__main__":
    main()
