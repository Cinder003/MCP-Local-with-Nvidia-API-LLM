

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

class UltraRobustNVIDIAFastMCPClient:
    """Ultra-robust client with Query Classification + LLM Knowledge + MCP Actions"""

    def __init__(self, api_key: str, model: str = "meta/llama-3.1-8b-instruct", default_working_dir: str = None):
        self.default_working_dir = default_working_dir or os.getcwd()
        self.server_connected = False
        
        # Set up NVIDIA API key
        os.environ["NVIDIA_API_KEY"] = api_key
        
        # Initialize NVIDIA LLM
        try:
            self.llm = ChatNVIDIA(
                model=model,
                temperature=0.1,
                max_completion_tokens=1000,
                nvidia_api_key=api_key
            )
            print(f"✅ Connected to NVIDIA model: {model}")
        except Exception as e:
            print(f"❌ NVIDIA LLM initialization failed: {e}")
            self.llm = None
        
        # FastMCP client connection
        self.mcp_client = None
        
        # Setup all components
        self._setup_query_classifier()
        self._setup_knowledge_llm()
        self._setup_ultra_robust_parser()
        self._setup_intent_patterns()

    def _setup_query_classifier(self):
        """Setup query classifier to route between LLM knowledge and MCP actions"""
        if not self.llm:
            return
            
        classification_prompt = """You are a query classifier. Classify the user's query into exactly one category:

CATEGORIES:
- "knowledge": User is asking for information, explanations, definitions, concepts, or general knowledge
- "action": User wants to perform a specific task, operation, or action (like creating files, running commands)
- "hybrid": User wants both information AND to perform an action

KNOWLEDGE Examples:
"What is machine learning?"
"Explain how Python works"
"Tell me about climate change"
"How does photosynthesis work?"
"What are the benefits of exercise?"
"Define artificial intelligence"
"How do neural networks learn?"
"What is the difference between AI and ML?"

ACTION Examples:
"Create a file called report.txt"
"Run the dir command"
"Launch notepad"
"Make a folder called project"
"Zip my documents folder"
"Show me files in Documents"
"Open calculator"
"List all processes"

HYBRID Examples:
"Explain machine learning and create a demo script"
"Tell me about Python and then create a hello.py file"
"What is data analysis and create a sample CSV"
"Describe file compression and zip my folder"

Query: {user_input}

Respond with ONLY the category name: knowledge, action, or hybrid"""

        self.classifier_prompt = ChatPromptTemplate.from_messages([
            ("system", classification_prompt),
            ("human", "{user_input}")
        ])
        
        self.query_classifier = self.classifier_prompt | self.llm | StrOutputParser()

    def _setup_knowledge_llm(self):
        """Setup dedicated LLM for knowledge-based queries"""
        if not self.llm:
            return
            
        knowledge_prompt = """You are a knowledgeable AI assistant. Provide comprehensive, accurate, and helpful information on any topic the user asks about.

Guidelines:
- Give detailed explanations with clear examples
- Break down complex concepts into understandable parts
- Provide practical context and real-world applications
- Be conversational and engaging
- Use analogies and examples to clarify difficult concepts
- Structure your response with clear sections if the topic is complex
- Include relevant facts, statistics, or research when applicable
- If uncertain about specific details, acknowledge limitations

User Question: {user_input}

Detailed Response:"""

        self.knowledge_prompt = ChatPromptTemplate.from_messages([
            ("system", knowledge_prompt),
            ("human", "{user_input}")
        ])
        
        self.knowledge_llm = self.knowledge_prompt | self.llm | StrOutputParser()

    def _setup_ultra_robust_parser(self):
        """Setup ultra-robust natural language parser for MCP actions"""
        if not self.llm:
            return

        system_prompt = """You are an advanced AI system orchestrator with deep natural language understanding. You can interpret ANY conversational request and map it to appropriate FastMCP tools.

AVAILABLE FASTMCP TOOLS:
📄 FILE & FOLDER OPERATIONS:
- create_file(path, content="", file_type="auto", working_directory=None) - Create any file type
- create_folder(path, working_directory=None) - Create directories
- read_file(path, working_directory=None) - Read any file  
- list_directory(path=".", working_directory=None) - List directory contents

🖥️ SYSTEM OPERATIONS:
- run_shell_command(command, working_directory=None) - Execute shell/cmd commands
- launch_application(app_name, file_path=None) - Launch apps or open files
- list_processes(filter_name=None) - Monitor running processes

📦 ARCHIVE OPERATIONS:
- zip_folder(folder_path, archive_name=None, working_directory=None) - Create ZIP archives

PARAMETER EXTRACTION INTELLIGENCE:
- Extract file/folder names from anywhere in the sentence
- Handle quoted names: "create folder 'my project'"
- Handle extensions: "make file report.xlsx" 
- Handle content: "create file with some text content"
- Handle paths: "list files in Documents/work"
- Handle commands: "run dir /w" or "execute ls -la"

RESPOND ONLY WITH VALID JSON:
{{"tool": "create_folder", "params": {{"path": "extracted_name"}}}}

Examples with escaped JSON:
"create a folder called my_project" → {{"tool": "create_folder", "params": {{"path": "my_project"}}}}
"I need a text file named notes.txt" → {{"tool": "create_file", "params": {{"path": "notes.txt", "file_type": "txt"}}}}
"show me files in Documents" → {{"tool": "list_directory", "params": {{"path": "Documents"}}}}
"run dir command" → {{"tool": "run_shell_command", "params": {{"command": "dir"}}}}"""

        self.command_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{user_input}")
        ])
        
        if self.llm:
            self.command_parser = self.command_prompt | self.llm | StrOutputParser()

    def _setup_intent_patterns(self):
        """Setup advanced intent classification patterns for fallback"""
        self.intent_patterns = {
            'create_file': {
                'keywords': ['create', 'make', 'new', 'generate', 'build'],
                'objects': ['file', 'document', 'text', 'excel', 'word', 'csv', 'json', 'script', 'code'],
                'extensions': ['.txt', '.xlsx', '.docx', '.csv', '.json', '.py', '.js', '.html', '.md', '.xml'],
                'indicators': ['called', 'named', 'titled', 'with name']
            },
            'create_folder': {
                'keywords': ['create', 'make', 'new', 'build', 'setup'],
                'objects': ['folder', 'directory', 'dir', 'subfolder', 'subdirectory'],
                'indicators': ['called', 'named', 'for', 'to store', 'to organize']
            },
            'read_file': {
                'keywords': ['read', 'show', 'display', 'view', 'see', 'check', 'look'],
                'objects': ['file', 'content', 'contents', 'data'],
                'prepositions': ['in', 'inside', 'from', 'of']
            },
            'list_directory': {
                'keywords': ['list', 'show', 'display', 'see', 'view', 'check'],
                'objects': ['files', 'directories', 'contents', 'items', 'stuff'],
                'locations': ['folder', 'directory', 'current', 'here', 'this']
            },
            'run_shell_command': {
                'keywords': ['run', 'execute', 'cmd', 'shell', 'command'],
                'indicators': ['command', 'script', 'terminal']
            },
            'launch_application': {
                'keywords': ['open', 'launch', 'start', 'run'],
                'objects': ['app', 'application', 'program', 'software'],
                'apps': ['notepad', 'calculator', 'chrome', 'firefox', 'excel', 'word', 'code']
            },
            'zip_folder': {
                'keywords': ['zip', 'compress', 'archive', 'backup'],
                'objects': ['folder', 'directory', 'files', 'project']
            }
        }

    async def initialize_mcp_client(self):
        """Initialize connection to the enhanced FastMCP server"""
        try:
            from complete_server import mcp
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

    async def classify_query(self, user_input: str) -> str:
        """Classify query into knowledge, action, or hybrid"""
        if not self.llm:
            return self._fallback_classification(user_input)
            
        try:
            print("🧠 Classifying query intent...")
            classification = await self.query_classifier.ainvoke({"user_input": user_input})
            classification = classification.strip().lower()
            
            # Validate classification
            valid_categories = ["knowledge", "action", "hybrid"]
            if classification not in valid_categories:
                classification = self._fallback_classification(user_input)
            
            print(f"📋 Classification: {classification}")
            return classification
            
        except Exception as e:
            print(f"⚠️ Classification failed: {e}, using fallback")
            return self._fallback_classification(user_input)

    def _fallback_classification(self, user_input: str) -> str:
        """Fallback classification using pattern matching"""
        ui = user_input.lower()
        
        # Check for action keywords
        action_keywords = ['create', 'make', 'run', 'execute', 'launch', 'open', 'list', 'show', 'zip', 'compress']
        
        # Check for knowledge keywords  
        knowledge_keywords = ['what', 'how', 'why', 'explain', 'tell me', 'define', 'describe', 'who', 'when', 'where']
        
        action_score = sum(1 for keyword in action_keywords if keyword in ui)
        knowledge_score = sum(1 for keyword in knowledge_keywords if keyword in ui)
        
        if action_score > 0 and knowledge_score > 0:
            return "hybrid"
        elif action_score > 0:
            return "action"
        elif knowledge_score > 0:
            return "knowledge"
        else:
            return "action"  # Default to action

    async def knowledge_workflow(self, user_input: str) -> str:
        """Handle knowledge-based queries with LLM"""
        try:
            print("📚 Processing knowledge query with LLM...")
            response = await self.knowledge_llm.ainvoke({"user_input": user_input})
            return f"🧠 **Knowledge Response:**\n{response}"
        except Exception as e:
            return f"❌ Knowledge processing failed: {str(e)}"

    async def action_workflow(self, user_input: str) -> str:
        """Handle action-based queries with MCP"""
        if not self.server_connected:
            return "❌ Not connected to FastMCP server."
        
        if not await self.health_check():
            return "❌ FastMCP server connection lost."
        
        # Parse command using existing MCP logic
        tool_name, params = self.parse_command(user_input)
        
        if tool_name == "unknown":
            return f"❌ I couldn't understand that request. Could you rephrase it?\n" \
                   f"Examples: 'create a file called test.txt', 'make a folder named project', 'list files'"
        
        # Add working directory
        if "working_directory" not in params and tool_name in ["create_file", "read_file", "create_folder", "list_directory", "zip_folder", "run_shell_command"]:
            params["working_directory"] = self.default_working_dir
        
        try:
            print(f"⚡ Executing {tool_name} with parameters...")
            result = await self.mcp_client.call_tool(tool_name, params)
            return f"⚡ **Action Result:**\n{result}"
            
        except McpError as e:
            return f"❌ FastMCP Error: {str(e)}"
        except Exception as e:
            if "connection" in str(e).lower():
                self.server_connected = False
                return "❌ Lost connection to FastMCP server."
            return f"❌ Error: {str(e)}"

    async def hybrid_workflow(self, user_input: str) -> str:
        """Handle hybrid queries with both LLM knowledge and MCP actions"""
        print("🔄 Processing hybrid query...")
        
        # Get knowledge response
        knowledge_response = await self.knowledge_workflow(user_input)
        
        # Execute action
        action_response = await self.action_workflow(user_input)
        
        # Combine responses
        return f"{knowledge_response}\n\n{action_response}"

    async def process_query(self, user_input: str) -> str:
        """Main query processing with classification and routing"""
        
        # Step 1: Classify the query
        classification = await self.classify_query(user_input)
        
        # Step 2: Route based on classification
        if classification == "knowledge":
            return await self.knowledge_workflow(user_input)
        elif classification == "action":
            return await self.action_workflow(user_input)
        else:  # hybrid
            return await self.hybrid_workflow(user_input)

    def parse_command(self, user_input: str) -> Tuple[str, Dict]:
        """Ultra-robust command parsing with multiple strategies (for MCP actions)"""
        
        # Strategy 1: Try NVIDIA AI first
        if self.llm:
            try:
                print("🤖 NVIDIA AI processing action command...")
                
                response = self.command_parser.invoke({"user_input": user_input})
                
                # Enhanced JSON extraction
                text = response.strip()
                
                # Multiple JSON extraction methods
                json_text = self._extract_json_from_text(text)
                
                if json_text:
                    try:
                        parsed = json.loads(json_text)
                        tool_name = parsed.get("tool", "unknown")
                        params = parsed.get("params", {})
                        
                        if tool_name != "unknown":
                            print(f"✅ AI understood: {tool_name}")
                            print(f"📊 Parameters: {params}")
                            return tool_name, params
                    except json.JSONDecodeError:
                        pass
            except Exception as e:
                print(f"⚠️ NVIDIA AI failed: {e}")
        
        # Strategy 2: Advanced intent classification
        print("🧠 Using advanced intent classification...")
        intent_result = self._classify_intent_advanced(user_input)
        if intent_result[0] != "unknown":
            return intent_result
        
        # Strategy 3: Multi-pattern fallback
        print("🔍 Using multi-pattern analysis...")
        return self._multi_pattern_fallback(user_input)

    # [Keep all your existing methods for parameter extraction]
    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """Advanced JSON extraction from various text formats - FIXED SYNTAX"""
        
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
                if candidate.startswith('{') and candidate.endswith('}'):
                    return candidate
        
        # Method 3: Find JSON object boundaries
        json_start = text.find('{')
        json_end = text.rfind('}')
        
        if json_start != -1 and json_end != -1 and json_end > json_start:
            return text[json_start:json_end + 1]
        
        # Method 4: Regex extraction
        json_pattern = r'\{[^{}]*"tool"[^{}]*"params"[^{}]*\}'
        match = re.search(json_pattern, text)
        if match:
            return match.group(0)
        
        return None

    def _classify_intent_advanced(self, user_input: str) -> Tuple[str, Dict]:
        """Advanced intent classification using multiple signals"""
        ui = user_input.lower().strip()
        
        # Score each intent
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            
            # Check keywords
            for keyword in patterns.get('keywords', []):
                if keyword in ui:
                    score += 3
            
            # Check objects
            for obj in patterns.get('objects', []):
                if obj in ui:
                    score += 2
            
            # Check indicators
            for indicator in patterns.get('indicators', []):
                if indicator in ui:
                    score += 1
            
            # Check extensions for files
            if intent == 'create_file':
                for ext in patterns.get('extensions', []):
                    if ext in ui:
                        score += 4
            
            # Check app names
            if intent == 'launch_application':
                for app in patterns.get('apps', []):
                    if app in ui:
                        score += 4
            
            intent_scores[intent] = score
        
        # Get best intent
        best_intent = max(intent_scores, key=intent_scores.get)
        max_score = intent_scores[best_intent]
        
        if max_score >= 3:  # Minimum confidence threshold
            return self._extract_parameters_for_intent(user_input, best_intent)
        
        return "unknown", {}

    def _extract_parameters_for_intent(self, user_input: str, intent: str) -> Tuple[str, Dict]:
        """Extract parameters for specific intent"""
        if intent == "create_file":
            return self._extract_file_creation_params(user_input)
        elif intent == "create_folder":
            return self._extract_folder_creation_params(user_input)
        elif intent == "read_file":
            return self._extract_file_reading_params(user_input)
        elif intent == "list_directory":
            return self._extract_directory_listing_params(user_input)
        elif intent == "run_shell_command":
            return self._extract_shell_command_params(user_input)
        elif intent == "launch_application":
            return self._extract_app_launch_params(user_input)
        elif intent == "zip_folder":
            return self._extract_zip_params(user_input)
        
        return "unknown", {}

    # [Keep all your existing parameter extraction methods]
    def _extract_file_creation_params(self, user_input: str) -> Tuple[str, Dict]:
        """Extract file creation parameters"""
        ui = user_input.lower()
        
        # Extract filename
        file_name = self._extract_name_with_multiple_methods(user_input, ['called', 'named', 'titled'])
        
        # If no name found, look for extensions
        if file_name in ['default', 'file']:
            extensions = re.findall(r'\b\w+\.(txt|xlsx|docx|csv|json|py|js|html|md)\b', ui)
            if extensions:
                file_name = extensions[0]
        
        # Extract content
        content = ""
        content_patterns = ['with content', 'containing', 'that says', 'with text']
        for pattern in content_patterns:
            if pattern in ui:
                content = user_input.split(pattern, 1)[1].strip()
                break
        
        # Determine file type
        file_type = "auto"
        if '.' in file_name:
            ext = file_name.split('.')[-1].lower()
            if ext in ['txt', 'xlsx', 'docx', 'csv', 'json', 'py', 'js', 'html', 'md']:
                file_type = ext
        
        params = {"path": file_name}
        if content:
            params["content"] = content
        if file_type != "auto":
            params["file_type"] = file_type
        
        return "create_file", params

    def _extract_folder_creation_params(self, user_input: str) -> Tuple[str, Dict]:
        """Extract folder creation parameters"""
        folder_name = self._extract_name_with_multiple_methods(user_input, ['called', 'named', 'for', 'titled'])
        return "create_folder", {"path": folder_name}

    def _extract_file_reading_params(self, user_input: str) -> Tuple[str, Dict]:
        """Extract file reading parameters"""
        file_name = self._extract_filename_from_text(user_input)
        return "read_file", {"path": file_name}

    def _extract_directory_listing_params(self, user_input: str) -> Tuple[str, Dict]:
        """Extract directory listing parameters"""
        ui = user_input.lower()
        path = "."
        
        # Look for path indicators
        path_patterns = [' in ', ' inside ', ' from ', ' of ']
        for pattern in path_patterns:
            if pattern in ui:
                path_part = ui.split(pattern, 1)[1].strip()
                path = path_part.split()[0] if path_part.split() else "."
                break
        
        return "list_directory", {"path": path}

    def _extract_shell_command_params(self, user_input: str) -> Tuple[str, Dict]:
        """Extract shell command parameters"""
        ui = user_input.lower()
        
        # Remove common prefixes
        command = user_input
        prefixes = ['run', 'execute', 'cmd', 'shell', 'command']
        for prefix in prefixes:
            if ui.startswith(prefix):
                command = user_input[len(prefix):].strip()
                break
        
        # Handle "run the X command" pattern
        if 'the ' in command and ' command' in command:
            command = command.replace('the ', '').replace(' command', '')
        
        return "run_shell_command", {"command": command}

    def _extract_app_launch_params(self, user_input: str) -> Tuple[str, Dict]:
        """Extract application launch parameters"""
        ui = user_input.lower()
        
        # Common applications
        apps = ['notepad', 'calculator', 'chrome', 'firefox', 'excel', 'word', 'code', 'cmd', 'powershell']
        
        app_name = "notepad"  # default
        for app in apps:
            if app in ui:
                app_name = app
                break
        
        # If no known app, take last word
        if app_name == "notepad" and ui.split():
            app_name = ui.split()[-1]
        
        return "launch_application", {"app_name": app_name}

    def _extract_zip_params(self, user_input: str) -> Tuple[str, Dict]:
        """Extract zip operation parameters"""
        ui = user_input.lower()
        
        # Extract folder name
        folder_path = "."
        words = ui.split()
        
        # Look for folder names
        for word in words:
            if word not in ['zip', 'compress', 'archive', 'backup', 'folder', 'directory', 'the', 'my']:
                folder_path = word
                break
        
        return "zip_folder", {"folder_path": folder_path}

    def _extract_name_with_multiple_methods(self, text: str, keywords: List[str]) -> str:
        """Extract names using multiple methods"""
        
        # Method 1: After keywords
        for keyword in keywords:
            if keyword in text.lower():
                parts = text.lower().split(keyword, 1)
                if len(parts) > 1:
                    after_keyword = parts[1].strip()
                    words = after_keyword.split()
                    if words:
                        return words[0].replace('"', '').replace("'", '')
        
        # Method 2: Quoted strings
        quoted = re.findall(r'"([^"]*)"', text) or re.findall(r"'([^']*)'", text)
        if quoted:
            return quoted[0]
        
        # Method 3: Extensions
        extensions = re.findall(r'\b(\w+\.\w+)\b', text)
        if extensions:
            return extensions[0]
        
        # Method 4: Last meaningful word
        words = text.split()
        meaningful_words = [w for w in words if w.lower() not in ['create', 'make', 'new', 'folder', 'file', 'called', 'named', 'a', 'the']]
        if meaningful_words:
            return meaningful_words[-1].replace('"', '').replace("'", '')
        
        return "default"

    def _extract_filename_from_text(self, text: str) -> str:
        """Extract filename from text using multiple strategies"""
        
        # Look for extensions
        extensions = re.findall(r'\b(\w+\.\w+)\b', text)
        if extensions:
            return extensions[0]
        
        # Look after common words
        keywords = ['file', 'in', 'from', 'of', 'inside']
        for keyword in keywords:
            if keyword in text.lower():
                parts = text.lower().split(keyword, 1)
                if len(parts) > 1:
                    words = parts[1].strip().split()
                    if words:
                        return words[0]
        
        # Last word as fallback
        words = text.split()
        return words[-1] if words else "file.txt"

    def _multi_pattern_fallback(self, user_input: str) -> Tuple[str, Dict]:
        """Multi-pattern fallback analysis"""
        ui = user_input.lower().strip()
        
        # Comprehensive patterns
        patterns = [
            # File creation patterns
            (r'\b(create|make|new|generate|build)\b.*\b(file|document|text|excel|word|csv|json|script)', 'create_file'),
            
            # Folder creation patterns  
            (r'\b(create|make|new|build|setup)\b.*\b(folder|directory|dir|subfolder)', 'create_folder'),
            
            # File reading patterns
            (r'\b(read|show|display|view|see|check|look)\b.*\b(file|content|contents|data)', 'read_file'),
            
            # Directory listing patterns
            (r'\b(list|show|display|see|view)\b.*\b(files|directories|contents|items)', 'list_directory'),
            
            # Shell command patterns
            (r'\b(run|execute|cmd|shell|command)\b', 'run_shell_command'),
            
            # App launch patterns
            (r'\b(open|launch|start)\b.*\b(app|application|program|notepad|calculator|chrome)', 'launch_application'),
            
            # Zip patterns
            (r'\b(zip|compress|archive|backup)\b', 'zip_folder'),
        ]
        
        for pattern, intent in patterns:
            if re.search(pattern, ui):
                return self._extract_parameters_for_intent(user_input, intent)
        
        return "unknown", {}

    def set_working_directory(self, directory: str):
        """Set working directory"""
        if os.path.exists(directory):
            self.default_working_dir = os.path.abspath(directory)
            print(f"📁 Working directory: {self.default_working_dir}")
        else:
            print(f"❌ Directory not found: {directory}")

    async def chat_mode(self):
        """Enhanced chat mode with query classification"""
        print("🚀 Enhanced NVIDIA FastMCP Client with Query Classification")
        print("=" * 70)
        print("🧠 Query Classification: Knowledge vs Action vs Hybrid")
        print("📚 Knowledge queries → Direct LLM responses")
        print("⚡ Action queries → MCP tool execution")
        print("🔄 Hybrid queries → Both knowledge and actions")
        print(f"📁 Working directory: {self.default_working_dir}")
        print()
        
        if not await self.initialize_mcp_client():
            print("❌ Failed to connect to FastMCP server")
            print("💡 Make sure to run: python complete_server.py")
            return
        
        print("💡 Try these examples:")
        print("   📚 Knowledge: 'What is machine learning?'")
        print("   📚 Knowledge: 'Explain how Python works'")
        print("   ⚡ Action: 'Create a file called report.txt'")
        print("   ⚡ Action: 'Show me files in this directory'")
        print("   🔄 Hybrid: 'Explain Python and create hello.py'")
        print()
        print("Commands: 'help', 'status', 'examples', 'exit'")
        print()
        
        try:
            while True:
                try:
                    user_input = input("You: ").strip()
                    
                    if user_input.lower() in ['exit', 'quit', 'q', 'bye']:
                        print("👋 Goodbye!")
                        break
                    
                    if user_input.lower() == 'help':
                        self._show_help()
                        continue
                    
                    if user_input.lower() == 'status':
                        await self._show_status()
                        continue
                    
                    if user_input.lower() == 'examples':
                        self._show_examples()
                        continue
                    
                    if user_input.lower().startswith('cd '):
                        self.set_working_directory(user_input[3:].strip())
                        continue
                    
                    if not user_input:
                        continue
                    
                    # Process query with classification and routing
                    result = await self.process_query(user_input)
                    
                    print()
                    print("🔍 **Response:**")
                    print(result)
                    print()
                    
                except KeyboardInterrupt:
                    print("\n👋 Interrupted. Goodbye!")
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
                    continue
        
        finally:
            await self.close_mcp_client()

    def _show_help(self):
        """Show comprehensive help with classification examples"""
        print()
        print("📋 Enhanced Query Classification System:")
        print()
        print("🧠 **KNOWLEDGE QUERIES** (Answered by LLM):")
        print("   • 'What is artificial intelligence?'")
        print("   • 'How does machine learning work?'")
        print("   • 'Explain the benefits of Python programming'")
        print("   • 'Tell me about climate change'")
        print("   • 'Define quantum computing'")
        print()
        print("⚡ **ACTION QUERIES** (Executed via MCP):")
        print("   • 'Create a file called notes.txt'")
        print("   • 'Make a folder called my_project'")
        print("   • 'Show me files in Documents'")
        print("   • 'Run the dir command'")
        print("   • 'Launch calculator'")
        print()
        print("🔄 **HYBRID QUERIES** (Both Knowledge + Action):")
        print("   • 'Explain machine learning and create demo.py'")
        print("   • 'Tell me about data analysis and create sample.csv'")
        print("   • 'What is file compression and zip my folder'")
        print()

    async def _show_status(self):
        """Show enhanced system status"""
        print()
        print("📊 **Enhanced System Status:**")
        print(f"   🔗 FastMCP Server: {'✅ Connected' if self.server_connected else '❌ Disconnected'}")
        print(f"   🤖 NVIDIA LLM: {'✅ Available' if self.llm else '❌ Not Available'}")
        print(f"   📁 Working Directory: {self.default_working_dir}")
        print(f"   🧠 Query Classifier: {'✅ Active' if self.llm else '❌ Pattern-based fallback'}")
        print(f"   📚 Knowledge LLM: {'✅ Active' if self.llm else '❌ Not Available'}")
        
        if self.server_connected:
            health = await self.health_check()
            print(f"   💚 Server Health: {'✅ Good' if health else '❌ Poor'}")
        print()

    def _show_examples(self):
        """Show practical examples with classification"""
        print()
        print("🎯 **Practical Usage Examples:**")
        print()
        print("📚 **Knowledge Examples:**")
        print("   'What is the difference between AI and ML?'")
        print("   'How do neural networks learn?'")
        print("   'Explain the Python programming language'")
        print()
        print("⚡ **Action Examples:**")
        print("   'Create a Python script called calculator.py'")
        print("   'Make a folder for my data science project'")
        print("   'Show me all text files in this directory'")
        print()
        print("🔄 **Hybrid Examples:**")
        print("   'Explain REST APIs and create api_demo.py'")
        print("   'Tell me about data visualization and create chart.py'")
        print("   'What is automation and create a batch script'")
        print()

async def main():
    """Main entry point"""
    API_KEY = "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"
    
    try:
        print("🔧 Initializing Enhanced Query Classifier Client...")
        client = UltraRobustNVIDIAFastMCPClient(api_key=API_KEY)
        await client.chat_mode()
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")

if __name__ == "__main__":
    asyncio.run(main())
