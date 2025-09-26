# MCP-Local-with-Nvidia-API-LLM
FastMCP server for system orchestration: create/read files (TXT, CSV, XLSX, DOCX, JSON), manage folders, run shell commands, launch apps, list processes, and zip folders with safety guards; plus an NVIDIA LLM client that maps natural language to tool calls with resilient fallbacks and parameter extraction and file previews built-in.

# Enhanced FastMCP System Documentation

## Overview

The Enhanced FastMCP System is a comprehensive, AI-powered file and system orchestration platform that combines the power of NVIDIA's language models with FastMCP's robust server-client architecture. This system provides natural language interfaces for complex file operations, system management, and data processing tasks.

## System Architecture

### Core Components

1. **Enhanced FastMCP Server** (`complete_server.py`)
   - Comprehensive system orchestration capabilities
   - Universal file creation and management
   - System operations and process management
   - Configurable security and access controls

2. **Ultra-Robust NVIDIA Client** (`complete_client.py`)
   - Natural language processing with NVIDIA AI models
   - Advanced intent classification
   - Multi-strategy command parsing
   - Robust error handling and fallback mechanisms

3. **Configuration System**
   - Server configuration (`server_config.json`)
   - Client configuration (`client_config.json`)
   - Environment variable support
   - Runtime parameter adjustment

## Key Features

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

### 🤖 Natural Language Interface

#### NVIDIA AI Integration
- **Supported Models**: Meta Llama 3.1, Mistral, CodeLlama, and more
- **Advanced Parsing**: Intent classification with confidence scoring
- **Context Awareness**: Working directory and session state management
- **Fallback Strategies**: Multiple parsing approaches for robustness

#### Intent Classification Patterns
```json
{
  "create_file": {
    "keywords": ["create", "make", "new", "generate", "build"],
    "objects": ["file", "document", "text", "excel", "word"],
    "confidence_threshold": 3
  },
  "system_operations": {
    "keywords": ["run", "execute", "launch", "start"],
    "objects": ["command", "app", "program"],
    "confidence_threshold": 2
  }
}
```

### 🔒 Security & Configuration

#### Access Control
- System-wide access control with configurable restrictions
- Path validation and sanitization
- Restricted directory protection
- File size limitations

#### Configurable Parameters
```json
{
  "server": {
    "max_file_size": 104857600,
    "system_wide_access": true,
    "restricted_paths": ["C:\\Windows\\System32", "/bin", "/sbin"]
  },
  "nvidia": {
    "model": "meta/llama-3.1-8b-instruct",
    "temperature": 0.1,
    "max_completion_tokens": 1000
  }
}
```

## Usage Examples

### Natural Language Commands

#### File Creation
```
User: "I need a text file called notes.txt with some project ideas"
System: Creates notes.txt with specified content

User: "can you make an excel spreadsheet for my budget tracking"
System: Creates budget.xlsx with sample headers and structure

User: "create a python script named data_processor.py"
System: Creates data_processor.py with basic Python template
```

#### System Operations
```
User: "show me all files in the Documents folder"
System: Lists directory contents with file details

User: "run a dir command to see current files"
System: Executes shell command and displays results

User: "open notepad please"
System: Launches Notepad application
```

#### Data Processing
```
User: "zip up my project folder into a backup archive"
System: Creates compressed archive with proper structure

User: "convert this CSV data to Excel format"
System: Processes and converts file format
```

## Technical Specifications

### Dependencies

#### Server Dependencies
```python
# Core Framework
fastmcp>=1.0.0
fastmcp.exceptions

# File Processing
openpyxl>=3.1.0
python-docx>=0.8.11
pandas>=1.5.0

# System Operations
psutil>=5.9.0
pathlib
subprocess
sqlite3

# Configuration
python-dotenv>=1.0.0
```

#### Client Dependencies
```python
# NVIDIA AI Integration
langchain-nvidia-ai-endpoints>=0.1.0
langchain-core>=0.2.0

# FastMCP Client
fastmcp>=1.0.0

# Utilities
python-dotenv>=1.0.0
asyncio
```

### Performance Characteristics

- **File Size Limit**: Configurable (default 100MB)
- **Command Timeout**: Configurable (default 30 seconds)
- **Parsing Strategies**: 3-tier fallback system
- **Memory Usage**: Optimized for large file operations
- **Response Time**: < 500ms for simple operations

### Error Handling

#### Multi-Level Error Recovery
1. **Primary**: NVIDIA AI natural language processing
2. **Secondary**: Pattern-based intent classification
3. **Tertiary**: Rule-based command parsing
4. **Fallback**: Interactive clarification prompts

#### Robust Exception Management
- Network connectivity issues
- API rate limiting
- File system permissions
- Invalid user input
- Resource constraints

## Deployment & Setup

### Environment Configuration

1. **NVIDIA API Setup**
   ```bash
   export NVIDIA_API_KEY="your_api_key_here"
   ```

2. **Working Directory**
   ```bash
   export WORKING_DIR="/path/to/workspace"
   ```

3. **Server Configuration**
   ```bash
   export SYSTEM_WIDE_ACCESS="true"
   export MAX_FILE_SIZE="104857600"
   export LOG_LEVEL="INFO"
   ```

### Installation Steps

1. **Install Dependencies**
   ```bash
   pip install fastmcp langchain-nvidia-ai-endpoints openpyxl python-docx pandas psutil python-dotenv
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your NVIDIA API key
   ```

3. **Start Server**
   ```bash
   python complete_server.py
   ```

4. **Run Client**
   ```bash
   python complete_client.py
   ```

## Advanced Configuration

### Custom Intent Patterns

Add new command patterns by extending the `intent_patterns` dictionary:

```python
"custom_operation": {
    "keywords": ["custom", "special", "unique"],
    "objects": ["task", "operation", "process"],
    "indicators": ["with", "using", "via"],
    "confidence_threshold": 2
}
```

### Server Extensions

Extend server capabilities by adding new tools:

```python
@mcp.tool
def custom_operation(param1: str, param2: int = 10) -> str:
    """Custom operation with configurable parameters"""
    # Implementation here
    return "Operation completed successfully"
```

### Client Customization

Modify parsing strategies by adjusting configuration:

```json
{
  "parsing": {
    "min_confidence_threshold": 3,
    "enable_llm_fallback": true,
    "enable_pattern_fallback": true,
    "custom_patterns": ["pattern1", "pattern2"]
  }
}
```

## Best Practices

### Security Considerations
- Always validate user input paths
- Implement proper access controls
- Use environment variables for sensitive data
- Regular security updates for dependencies

### Performance Optimization
- Configure appropriate file size limits
- Use working directory contexts
- Implement caching for frequently accessed files
- Monitor system resource usage

### Maintainability
- Keep configuration files updated
- Regular backup of important data
- Document custom extensions
- Version control for configuration changes

## Troubleshooting

### Common Issues

1. **NVIDIA API Connection**
   - Verify API key validity
   - Check network connectivity
   - Review rate limiting

2. **File Permission Errors**
   - Check directory permissions
   - Verify restricted path settings
   - Run with appropriate privileges

3. **Parsing Failures**
   - Review intent classification patterns
   - Check minimum confidence thresholds
   - Verify fallback mechanisms

### Debug Mode

Enable detailed logging for troubleshooting:

```json
{
  "logging": {
    "level": "DEBUG",
    "file": "debug.log",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  }
}
```

## Version History

- **v1.0**: Initial release with basic FastMCP integration
- **v1.1**: Added NVIDIA AI natural language processing
- **v1.2**: Enhanced configuration system and error handling
- **v1.3**: Advanced intent classification and multi-strategy parsing
- **v1.4**: Comprehensive file type support and system operations

## License

This project is licensed under the MIT License. See LICENSE file for details.

---

*This documentation covers the Enhanced FastMCP System - a powerful, AI-driven file and system orchestration platform combining natural language processing with robust system operations.*
