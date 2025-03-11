from typing import Any, Optional
import asyncio
import httpx
import base64
import os
from pathlib import Path
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio

TOGETHER_AI_BASE = "https://api.together.xyz/v1/images/generations"
API_KEY = os.getenv("TOGETHER_AI_API_KEY")
DEFAULT_MODEL = "black-forest-labs/FLUX.1-schnell"

# Create a folder for saving images if it doesn't exist
TMP_FOLDER = "/tmp/generated_images"
os.makedirs(TMP_FOLDER, exist_ok=True)

server = Server("image-gen")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """
    List available tools.
    Each tool specifies its arguments using JSON Schema validation.
    """
    return [
        types.Tool(
            name="generate_image",
            description="Generate an image based on the text prompt, model, and optional dimensions",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The text prompt for image generation",
                    },
                    "model": {
                        "type": "string",
                        "description": "The exact model name as it appears in Together AI. If incorrect, it will fallback to the default model (black-forest-labs/FLUX.1-schnell).",
                    },
                    "width": {
                        "type": "number",
                        "description": "Optional width for the image",
                    },
                    "height": {
                        "type": "number",
                        "description": "Optional height for the image",
                    },
                    "filename": {
                        "type": "string",
                        "description": "Optional filename to save the image (without extension)",
                    },
                },
                "required": ["prompt", "model"],
            },
        )
    ]


async def make_together_request(
    client: httpx.AsyncClient,
    prompt: str,
    model: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> dict[str, Any]:
    """Make a request to the Together API with error handling and fallback for incorrect model."""
    request_body = {"model": model, "prompt": prompt, "response_format": "b64_json"}
    headers = {"Authorization": f"Bearer {API_KEY}"}

    if width is not None:
        request_body["width"] = width
    if height is not None:
        request_body["height"] = height

    async def send_request(body: dict) -> (int, dict):
        response = await client.post(TOGETHER_AI_BASE, headers=headers, json=body)
        try:
            data = response.json()
        except Exception:
            data = {}
        return response.status_code, data

    # First request with user-provided model
    status, data = await send_request(request_body)

    # Check if the request failed due to an invalid model error
    if status != 200 and "error" in data:
        error_info = data["error"]
        error_msg = error_info.get("message", "").lower()
        error_code = error_info.get("code", "").lower()
        if (
            "model" in error_msg and "not available" in error_msg
        ) or error_code == "model_not_available":
            # Fallback to the default model
            request_body["model"] = DEFAULT_MODEL
            status, data = await send_request(request_body)
            if status != 200 or "error" in data:
                return {
                    "error": f"Fallback API error: {data.get('error', 'Unknown error')} (HTTP {status})"
                }
            return data
        else:
            return {"error": f"Together API error: {data.get('error')}"}
    elif status != 200:
        return {"error": f"HTTP error {status}"}

    return data


def save_image_to_file(b64_image: str, filename: str = None) -> str:
    """
    Save the base64 encoded image to a file in the /tmp folder.
    Returns the full path to the saved image.
    """
    # Generate a filename based on timestamp if not provided
    if not filename:
        import time
        filename = f"image_{int(time.time())}"
    
    # Ensure filename doesn't have an extension
    filename = Path(filename).stem
    
    # Create the full file path
    file_path = os.path.join(TMP_FOLDER, f"{filename}.jpg")
    
    # Decode the base64 image and save to file
    try:
        image_data = base64.b64decode(b64_image)
        with open(file_path, "wb") as f:
            f.write(image_data)
        return file_path
    except Exception as e:
        print(f"Error saving image: {e}")
        return None


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Handle tool execution requests.
    Tools can generate images and notify clients of changes.
    """
    if not arguments:
        return [
            types.TextContent(type="text", text="Missing arguments for the request")
        ]

    if name == "generate_image":
        prompt = arguments.get("prompt")
        model = arguments.get("model")
        width = arguments.get("width")
        height = arguments.get("height")
        filename = arguments.get("filename")

        if not prompt or not model:
            return [
                types.TextContent(type="text", text="Missing prompt or model parameter")
            ]

        async with httpx.AsyncClient() as client:
            response_data = await make_together_request(
                client=client,
                prompt=prompt,
                model=model,  # User-provided model (or fallback will be used)
                width=width,
                height=height,
            )

            if "error" in response_data:
                return [types.TextContent(type="text", text=response_data["error"])]

            try:
                b64_image = response_data["data"][0]["b64_json"]
                
                # Save the image to a file
                saved_path = save_image_to_file(b64_image, filename)
                
                # Include the saved file path in the response
                if saved_path:
                    response_text = f"Image saved to: {saved_path}"
                else:
                    response_text = "Failed to save image to file."
                
                return [
                    # Return the image content for display
                    types.ImageContent(
                        type="image", data=b64_image, mimeType="image/jpeg"
                    ),
                    # Also return text with the file path
                    types.TextContent(type="text", text=response_text)
                ]
            except (KeyError, IndexError) as e:
                return [
                    types.TextContent(
                        type="text", text=f"Failed to parse API response: {e}"
                    )
                ]


async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="image-gen",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())