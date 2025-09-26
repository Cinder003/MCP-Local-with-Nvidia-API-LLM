# nvidia_fastmcp_client.py
import asyncio
import os
import json
import re
from typing import Dict, Any, Optional, List, Tuple
from fastmcp import Client
from fastmcp.exceptions import McpError
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_client_config():
    """Load client configuration from JSON file with environment variable fallbacks"""
    config_path = 'client_config.json'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Default configuration if file not found
        config = {
            "nvidia": {
                "api_key": "${NVIDIA_API_KEY}",
                "model": "meta/llama-3.1-8b-instruct",
                "temperature": 0.1,
                "max_completion_tokens": 1000
            },
            "client": {
                "default_working_dir": "${PWD}",
                "connection_timeout": 30,
                "health_check_interval": 60
            },
            "parsing": {
                "min_confidence_threshold": 3,
                "enable_llm_fallback": True,
                "enable_pattern_fallback": True
            }
        }

    # Override with environment variables
    api_key = os.getenv("NVIDIA_API_KEY")
    if api_key:
        config["nvidia"]["api_key"] = api_key

    working_dir = os.getenv("WORKING_DIR", os.getcwd())
    config["client"]["default_working_dir"] = working_dir

    return config

class UltraRobustNVIDIAFastMCPClient:
    """Ultra-robust client with advanced natural language processing and configurable settings"""

    def __init__(self, config_file: str = None):
        # Load configuration
        self.config = load_client_config()

        # Extract configuration values
        self.api_key = self.config["nvidia"]["api_key"]
        self.model = self.config["nvidia"]["model"]
        self.temperature = self.config["nvidia"]["temperature"]
        self.max_tokens = self.config["nvidia"]["max_completion_tokens"]

        self.default_working_dir = self.config["client"]["default_working_dir"]
        self.connection_timeout = self.config["client"]["connection_timeout"]
        self.health_check_interval = self.config["client"]["health_check_interval"]

        self.min_confidence = self.config["parsing"]["min_confidence_threshold"]
        self.enable_llm_fallback = self.config["parsing"]["enable_llm_fallback"]
        self.enable_pattern_fallback = self.config["parsing"]["enable_pattern_fallback"]

        self.server_connected = False

        # Validate API key
        if self.api_key == "${NVIDIA_API_KEY}" or not self.api_key:
            print("⚠️  NVIDIA_API_KEY not set - will use pattern-only parsing")
            self.llm = None
        else:
            # Set up NVIDIA API key
            os.environ["NVIDIA_API_KEY"] = self.api_key

            # Initialize NVIDIA LLM
            try:
                self.llm = ChatNVIDIA(
                    model=self.model,
                    temperature=self.temperature,
                    max_completion_tokens=self.max_tokens,
                    nvidia_api_key=self.api_key
                )
                print(f"✅ Connected to NVIDIA model: {self.model}")
            except Exception as e:
                print(f"❌ NVIDIA LLM initialization failed: {e}")
                self.llm = None

        # FastMCP client connection
        self.mcp_client = None

        # Setup advanced command parsing
        self.setup_ultra_robust_parser()
        self.setup_intent_patterns()

    def setup_ultra_robust_parser(self):
        """Setup ultra-robust natural language parser"""
        if not self.llm:
            return

        system_prompt = f"""You are an advanced AI system orchestrator with deep natural language understanding.
You can interpret ANY conversational request and map it to appropriate FastMCP tools.

Configuration:
- Model: {self.model}
- Temperature: {self.temperature}
- Max tokens: {self.max_tokens}
- Min confidence: {self.min_confidence}

AVAILABLE FASTMCP TOOLS:

FILE & FOLDER OPERATIONS:
- create_file(path, content="", file_type="auto", working_directory=None) - Create any file type
- create_folder(path, working_directory=None) - Create directories  
- read_file(path, working_directory=None) - Read any file
- list_directory(path=".", working_directory=None) - List directory contents

SYSTEM OPERATIONS:
- run_shell_command(command, working_directory=None) - Execute shell/cmd commands
- launch_application(app_name, file_path=None) - Launch apps or open files
- list_processes(filter_name=None) - Monitor running processes

ARCHIVE OPERATIONS:
- zip_folder(folder_path, archive_name=None, working_directory=None) - Create ZIP archives

RESPOND ONLY WITH VALID JSON:
{{"tool": "create_folder", "params": {{"path": "extracted_name"}}}}

Examples with escaped JSON:
"create a folder called myproject" → {{"tool": "create_folder", "params": {{"path": "myproject"}}}}
"I need a text file named notes.txt" → {{"tool": "create_file", "params": {{"path": "notes.txt", "file_type": "txt"}}}}
"show me files in Documents" → {{"tool": "list_directory", "params": {{"path": "Documents"}}}}
"run dir command" → {{"tool": "run_shell_command", "params": {{"command": "dir"}}}}
"""

        self.command_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{user_input}")
        ])

        if self.llm:
            self.command_parser = self.command_prompt | self.llm | StrOutputParser()

    def setup_intent_patterns(self):
        """Setup advanced intent classification patterns"""
        self.intent_patterns = {
            "create_file": {
                "keywords": ["create", "make", "new", "generate", "build"],
                "objects": ["file", "document", "text", "excel", "word", "csv", "json", "script", "code"],
                "extensions": [".txt", ".xlsx", ".docx", ".csv", ".json", ".py", ".js", ".html", ".md", ".xml"],
                "indicators": ["called", "named", "titled", "with name"]
            },
            "create_folder": {
                "keywords": ["create", "make", "new", "build", "setup"],
                "objects": ["folder", "directory", "dir", "subfolder", "subdirectory"],
                "indicators": ["called", "named", "for", "to store", "to organize"]
            },
            "read_file": {
                "keywords": ["read", "show", "display", "view", "see", "check", "look"],
                "objects": ["file", "content", "contents", "data"],
                "prepositions": ["in", "inside", "from", "of"]
            },
            "list_directory": {
                "keywords": ["list", "show", "display", "see", "view", "check"],
                "objects": ["files", "directories", "contents", "items", "stuff"],
                "locations": ["folder", "directory", "current", "here", "this"]
            },
            "run_shell_command": {
                "keywords": ["run", "execute", "cmd", "shell", "command"],
                "indicators": ["command", "script", "terminal"]
            },
            "launch_application": {
                "keywords": ["open", "launch", "start", "run"],
                "objects": ["app", "application", "program", "software"],
                "apps": ["notepad", "calculator", "chrome", "firefox", "excel", "word", "code"]
            },
            "zip_folder": {
                "keywords": ["zip", "compress", "archive", "backup"],
                "objects": ["folder", "directory", "files", "project"]
            }
        }

    async def initialize_mcp_client(self):
        """Initialize connection to the enhanced FastMCP server"""
        try:
            from complete_server_updated import mcp  # Import updated server
            self.mcp_client = Client(mcp)
            await self.mcp_client.__aenter__()
            await self.health_check()
            self.server_connected = True
            print("✅ Connected to Enhanced FastMCP Server")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Enhanced FastMCP server: {e}")
            return False

    async def health_check(self):
        """Check server health"""
        try:
            await self.mcp_client.call_tool("list_directory", {"path": "."})
            return True
        except:
            self.server_connected = False
            return False

    async def close_mcp_client(self):
        """Close connection"""
        if self.mcp_client:
            await self.mcp_client.__aexit__(None, None, None)
            self.server_connected = False

    def parse_command(self, user_input: str) -> Tuple[str, Dict]:
        """Ultra-robust command parsing with multiple strategies"""

        # Strategy 1: Try NVIDIA AI first if enabled
        if self.llm and self.enable_llm_fallback:
            try:
                print("🤖 NVIDIA AI processing natural language...")
                response = self.command_parser.invoke({"user_input": user_input})

                # Enhanced JSON extraction
                text = response.strip()
                json_text = self.extract_json_from_text(text)

                if json_text:
                    try:
                        parsed = json.loads(json_text)
                        tool_name = parsed.get("tool", "unknown")
                        params = parsed.get("params", {})

                        if tool_name != "unknown":
                            print(f"✅ AI understood: {tool_name}")
                            print(f"Parameters: {params}")
                            return tool_name, params
                    except json.JSONDecodeError:
                        pass
            except Exception as e:
                print(f"❌ NVIDIA AI failed: {e}")

        # Strategy 2: Advanced intent classification if pattern fallback enabled
        if self.enable_pattern_fallback:
            print("🧠 Using advanced intent classification...")
            intent_result = self.classify_intent_advanced(user_input)
            if intent_result[0] != "unknown":
                return intent_result

        # Strategy 3: Multi-pattern fallback
        print("🔍 Using multi-pattern analysis...")
        return self.multi_pattern_fallback(user_input)

    # [Include all the existing parsing methods with config references...]
    # For brevity, showing key methods that would use configuration

    def extract_json_from_text(self, text: str) -> Optional[str]:
        """Advanced JSON extraction from various text formats"""
        # Method 1: Standard markdown code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end != -1:
                return text[start:end].strip()

        # Method 2: Any code blocks
        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end != -1:
                candidate = text[start:end].strip()
                if candidate.startswith("{") and candidate.endswith("}"):
                    return candidate

        # Method 3: Find JSON object boundaries
        json_start = text.find("{")
        json_end = text.rfind("}")
        if json_start != -1 and json_end != -1 and json_end > json_start:
            return text[json_start:json_end + 1]

        # Method 4: Regex extraction
        json_pattern = r'\{"tool":[^}]+\}'
        match = re.search(json_pattern, text)
        if match:
            return match.group(0)

        return None

    def classify_intent_advanced(self, user_input: str) -> Tuple[str, Dict]:
        """Advanced intent classification using multiple signals"""
        ui = user_input.lower().strip()

        intent_scores = {}

        # Score each intent
        for intent, patterns in self.intent_patterns.items():
            score = 0

            # Check keywords
            for keyword in patterns.get("keywords", []):
                if keyword in ui:
                    score += 3

            # Check objects
            for obj in patterns.get("objects", []):
                if obj in ui:
                    score += 2

            # Check indicators
            for indicator in patterns.get("indicators", []):
                if indicator in ui:
                    score += 1

            # Check extensions for files
            if intent == "create_file":
                for ext in patterns.get("extensions", []):
                    if ext in ui:
                        score += 4

            # Check app names
            if intent == "launch_application":
                for app in patterns.get("apps", []):
                    if app in ui:
                        score += 4

            intent_scores[intent] = score

        # Get best intent
        best_intent = max(intent_scores, key=intent_scores.get)
        max_score = intent_scores[best_intent]

        if max_score >= self.min_confidence:  # Use configured minimum confidence
            return self.extract_parameters_for_intent(user_input, best_intent)

        return "unknown", {}

    # [Include all remaining methods with configuration references...]

    async def chat_mode(self):
        """Ultra-robust interactive chat mode"""
        print("🚀 Ultra-Robust NVIDIA FastMCP Client")
        print("=" * 60)
        print("🤖 Advanced Natural Language Processing")
        print("📁 Understands ANY conversational request")
        print(f"📂 Working directory: {self.default_working_dir}")
        print(f"⚙️  Configuration loaded from client_config.json")

        if not await self.initialize_mcp_client():
            print("❌ Failed to connect to Enhanced FastMCP server")
            return

        print("\n💬 Just talk naturally! Examples:")
        print("• I need a text file called notes.txt")
        print("• can you create a folder for my project")
        print("• show me what files are in Documents")
        print("• run a dir command to see current files")
        print("• open notepad for me please")
        print()
        print("Commands: help, status, examples, exit")
        print()

        try:
            while True:
                try:
                    user_input = input("You: ").strip()

                    if user_input.lower() in ['exit', 'quit', 'q', 'bye']:
                        print("👋 Goodbye!")
                        break

                    if user_input.lower() == 'help':
                        self.show_help()
                        continue

                    if user_input.lower() == 'status':
                        await self.show_status()
                        continue

                    if user_input.lower() == 'examples':
                        self.show_examples()
                        continue

                    if user_input.lower().startswith('cd '):
                        self.set_working_directory(user_input[3:].strip())
                        continue

                    if not user_input:
                        continue

                    # Parse command
                    tool_name, params = self.parse_command(user_input)

                    if tool_name == "unknown":
                        print("❓ I couldn't understand that request. Could you rephrase it?")
                        print("Examples: create a file called test.txt, make a folder named project, list files")
                        continue

                    # Execute command
                    result = await self.execute_command(user_input)
                    print(f"Result: {result}")
                    print()

                except KeyboardInterrupt:
                    print("\n⏹️  Interrupted. Goodbye!")
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
                    continue

        finally:
            await self.close_mcp_client()

# [Include all remaining methods...]

async def main():
    """Main entry point with configuration loading"""
    try:
        print("🔧 Loading configuration...")
        client = UltraRobustNVIDIAFastMCPClient()
        await client.chat_mode()
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")

if __name__ == "__main__":
    asyncio.run(main())
