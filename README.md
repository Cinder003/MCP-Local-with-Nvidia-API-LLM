# MCP-Local-with-Nvidia-API-LLM
FastMCP server for system orchestration: create/read files (TXT, CSV, XLSX, DOCX, JSON), manage folders, run shell commands, launch apps, list processes, and zip folders with safety guards; plus an NVIDIA LLM client with intelligent query classifier mapping natural language to tool calls, resilient fallbacks, parameter extraction, and file previews built-in.

# Enhanced FastMCP System Documentation

## Overview

The Enhanced FastMCP System is a comprehensive, AI-powered file and system orchestration platform that combines the power of NVIDIA's language models with FastMCP's robust server-client architecture. This system provides natural language interfaces for complex file operations, system management, and data processing tasks with intelligent query classification.

## System Architecture

### Core Components

1. **Enhanced FastMCP Server** (`complete_server.py`)
   - Comprehensive system orchestration capabilities
   - Universal file creation and management
   - System operations and process management
   - Configurable security and access controls

2. **Ultra-Robust NVIDIA Client with Query Classifier** (`complete_client.py`)
   - **Intelligent Query Classification**: Auto-routes queries as Knowledge, Action, or Hybrid
   - Natural language processing with NVIDIA AI models
   - Advanced intent classification with triple-fallback parsing
   - Multi-strategy command parsing with resilient error handling

3. **Configuration System**
   - Server configuration (`server_config.json`)
   - Client configuration (`client_config.json`)
   - Environment variable support
   - Runtime parameter adjustment

## Key Features

### 🧠 Intelligent Query Classification System

#### Query Categories
- **Knowledge Queries**: Informational requests answered directly by LLM
  - Examples: "What is machine learning?", "Explain Python programming"
- **Action Queries**: System operations executed via MCP tools
  - Examples: "Create file report.txt", "Run dir command", "Launch notepad"
- **Hybrid Queries**: Combined knowledge + action responses
  - Examples: "Explain ML and create demo.py", "What is file compression and zip my folder"

#### Classification Workflow
```python
# Enhanced client flow with classification
User Query → Query Classifier → Route Decision
                     ↓
    Knowledge ← [LLM Response] → Action → [MCP Execution]
                     ↓
                  Hybrid → [Both LLM + MCP]
```

#### Classification Methods
1. **Primary**: LLM-based classification with prompt engineering
2. **Secondary**: Pattern-based fallback classification
3. **Tertiary**: Keyword scoring with confidence thresholds

### 🔧 Universal File Operations

#### File Creation Support
- **Text Files**: `.txt`, `.md`, `.log`, `.ini`, `.cfg`, `.conf`
- **Programming Files**: `.py`, `.js`, `.html`, `.css`, `.json`, `.xml`, `.yaml`, `.sql`
- **Office Documents**: `.xlsx`, `.xls`, `.docx`, `.doc`, `.rtf`
- **Data Files**: `.csv`, `.tsv`
- **Scripts**: `.bat`, `.sh`, `.ps1`

#### Advanced File Management
```python
# Examples of supported operations
create_file(path, content="", file_type="auto", working_directory=None)
create_folder(path, working_directory=None)
read_file(path, working_directory=None)
list_directory(path=".", working_directory=None)
copy_file(source, destination, working_directory=None)
move_file(source, destination, working_directory=None)
delete_file(path, working_directory=None)
search_files(pattern, directory=".", recursive=True)
```

### 🖥️ System Operations

#### Shell Command Execution
- Cross-platform shell command support (Windows CMD, PowerShell, Linux Bash)
- Configurable timeout settings
- Output capture and formatting
- Working directory context

#### Process Management
- List running processes with filtering
- Process monitoring and statistics
- System resource information
- Application launching capabilities

#### Archive Operations
- ZIP file creation with folder structure preservation
- Archive extraction with proper permissions
- Batch compression operations
- Integrity verification

### 🤖 Enhanced Natural Language Interface

#### Query Classification Features
```python
# New client components
_setup_query_classifier()      # LLM-based query routing
_setup_knowledge_llm()         # Dedicated knowledge responses
classify_query()               # Primary classification method
knowledge_workflow()           # LLM-only workflow
action_workflow()              # MCP-only workflow
hybrid_workflow()              # Combined workflow
process_query()                # Main routing entry point
```

#### NVIDIA AI Integration
- **Supported Models**: Meta Llama 3.1, Mistral, CodeLlama, and more
- **Advanced Parsing**: Intent classification with confidence scoring
- **Context Awareness**: Working directory and session state management
- **Fallback Strategies**: Multiple parsing approaches for robustness

#### Enhanced Classification Patterns
```json
{
  "query_classification": {
    "knowledge": {
      "keywords": ["what", "how", "why", "explain", "tell me", "define"],
      "confidence_threshold": 3
    },
    "action": {
      "keywords": ["create", "make", "run", "execute", "launch", "show"],
      "confidence_threshold": 3
    },
    "hybrid": {
      "patterns": ["explain X and create Y", "tell me about X and make Y"],
      "confidence_threshold": 5
    }
  }
}
```

### 🔒 Security & Configuration

#### Access Control
- System-wide access control with configurable restrictions
- Path validation and sanitization
- Restricted directory protection
- File size limitations

#### Enhanced Configuration
```json
{
  "server": {
    "max_file_size": 104857600,
    "system_wide_access": true,
    "restricted_paths": ["C:\Windows\System32", "/bin", "/sbin"]
  },
  "nvidia": {
    "model": "meta/llama-3.1-8b-instruct",
    "temperature": 0.1,
    "max_completion_tokens": 1000
  },
  "classification": {
    "enable_query_classification": true,
    "fallback_to_action": true,
    "min_confidence_threshold": 3
  }
}
```

## Usage Examples

### Enhanced Natural Language Commands

#### Knowledge Queries (LLM Response Only)
```
User: "What is machine learning and how does it work?"
Classification: Knowledge
Response: Comprehensive LLM explanation about ML concepts, algorithms, applications

User: "Explain the difference between Python and JavaScript"
Classification: Knowledge
Response: Detailed comparison of both programming languages
```

#### Action Queries (MCP Execution Only)
```
User: "Create a text file called notes.txt with some project ideas"
Classification: Action
Response: File created via MCP server with specified content

User: "Show me all files in the Documents folder"
Classification: Action
Response: Directory listing via MCP list_directory tool
```

#### Hybrid Queries (Both LLM + MCP)
```
User: "Explain Python programming and create hello.py"
Classification: Hybrid
Response: 
🧠 Knowledge Response: [Detailed Python explanation]
⚡ Action Result: [hello.py file created successfully]

User: "Tell me about data analysis and create sample.csv"
Classification: Hybrid
Response:
🧠 Knowledge Response: [Data analysis concepts and methods]
⚡ Action Result: [sample.csv created with headers]
```

## Enhanced Client Architecture Changes

### New Query Processing Flow

#### Before (Action-Only Client)
```
User Query → Action Parser → MCP Tool → Result
```

#### After (Classification-Enhanced Client)
```
User Query → Query Classifier → Branch Decision
                    ↓
Knowledge: LLM Response → Direct Answer
Action: MCP Tools → Tool Execution Result  
Hybrid: Both Workflows → Combined Response
```

### New Client Components

#### 1. Query Classification System
```python
async def classify_query(self, user_input: str) -> str:
    """Classify query into knowledge, action, or hybrid"""
    # LLM-based classification with fallback
    
def _fallback_classification(self, user_input: str) -> str:
    """Pattern-based classification backup"""
    # Keyword scoring and confidence thresholds
```

#### 2. Workflow Routing
```python
async def knowledge_workflow(self, user_input: str) -> str:
    """Handle pure knowledge queries with LLM"""
    
async def action_workflow(self, user_input: str) -> str:
    """Handle action queries with MCP tools"""
    
async def hybrid_workflow(self, user_input: str) -> str:
    """Handle queries needing both knowledge and actions"""
```

#### 3. Enhanced Configuration
```python
def _setup_query_classifier(self):
    """Setup LLM-based query classification"""
    
def _setup_knowledge_llm(self):
    """Setup dedicated knowledge response system"""
```

### Classification Accuracy Metrics

#### Confidence Scoring
- **High Confidence** (Score 5+): Direct routing
- **Medium Confidence** (Score 3-4): Routing with validation
- **Low Confidence** (Score <3): Fallback to pattern matching

#### Fallback Strategy
1. **Primary**: NVIDIA LLM classification
2. **Secondary**: Pattern-based keyword matching
3. **Tertiary**: Default to action workflow

## Technical Specifications

### Enhanced Dependencies

#### Server Dependencies (Unchanged)
```python
# Core Framework
fastmcp>=1.0.0
# [Previous dependencies remain the same]
```

#### Enhanced Client Dependencies
```python
# NVIDIA AI Integration (Enhanced)
langchain-nvidia-ai-endpoints>=0.1.0
langchain-core>=0.2.0

# FastMCP Client
fastmcp>=1.0.0

# New: Enhanced natural language processing
# Additional utilities for classification
asyncio
re (for pattern matching)
typing (for enhanced type hints)
```

### Performance Characteristics

- **Classification Time**: < 200ms per query
- **Knowledge Response**: 1-3 seconds (LLM dependent)
- **Action Execution**: < 500ms for simple operations
- **Hybrid Operations**: Combined timing of both workflows
- **Fallback Activation**: < 50ms for pattern matching

### Enhanced Error Handling

#### Four-Level Error Recovery
1. **Primary**: NVIDIA AI query classification
2. **Secondary**: NVIDIA AI action parsing  
3. **Tertiary**: Pattern-based intent classification
4. **Quaternary**: Rule-based command parsing with user clarification

## Advanced Configuration

### Classification Tuning

```python
# Adjust classification sensitivity
"classification_config": {
    "knowledge_keywords": ["what", "how", "why", "explain", "define", "tell me"],
    "action_keywords": ["create", "make", "run", "execute", "launch", "show"],
    "hybrid_patterns": ["explain.*and.*create", "tell.*about.*and.*make"],
    "confidence_thresholds": {
        "knowledge": 3,
        "action": 3, 
        "hybrid": 5
    }
}
```

### Workflow Customization

```python
# Custom workflow behaviors
"workflow_config": {
    "knowledge_workflow": {
        "max_response_length": 1000,
        "include_examples": true,
        "format_markdown": true
    },
    "action_workflow": {
        "confirm_destructive_actions": true,
        "show_execution_steps": true
    },
    "hybrid_workflow": {
        "knowledge_first": true,
        "combine_responses": true
    }
}
```

## Version History

- **v1.0**: Initial release with basic FastMCP integration
- **v1.1**: Added NVIDIA AI natural language processing
- **v1.2**: Enhanced configuration system and error handling
- **v1.3**: Advanced intent classification and multi-strategy parsing
- **v1.4**: Comprehensive file type support and system operations
- **v1.5**: **NEW - Intelligent Query Classification System**
  - Added query classifier for Knowledge/Action/Hybrid routing
  - Implemented dedicated knowledge workflow with LLM responses
  - Enhanced hybrid workflow combining both LLM knowledge and MCP actions
  - Added classification confidence scoring and fallback mechanisms

***

*This documentation covers the Enhanced FastMCP System with Intelligent Query Classification - a powerful, AI-driven platform that seamlessly combines natural language understanding with robust system operations through intelligent query routing.*
