# Graphiti MCP Server

Graphiti is a framework for building and querying temporally-aware knowledge graphs, specifically tailored for AI agents
operating in dynamic environments. Unlike traditional retrieval-augmented generation (RAG) methods, Graphiti
continuously integrates user interactions, structured and unstructured enterprise data, and external information into a
coherent, queryable graph. The framework supports incremental data updates, efficient retrieval, and precise historical
queries without requiring complete graph recomputation, making it suitable for developing interactive, context-aware AI
applications.

This is an experimental Model Context Protocol (MCP) server implementation for Graphiti. The MCP server exposes
Graphiti's key functionality through the MCP protocol, allowing AI assistants to interact with Graphiti's knowledge
graph capabilities.

## Features

The Graphiti MCP server provides comprehensive knowledge graph capabilities:

- **Episode Management**: Add, retrieve, and delete episodes (text, messages, or JSON data)
- **Entity Management**: Search and manage entity nodes and relationships in the knowledge graph
- **Search Capabilities**: Search for facts (edges) and node summaries using semantic and hybrid search
- **Group Management**: Organize and manage groups of related data with group_id filtering
- **Graph Maintenance**: Clear the graph and rebuild indices
- **Graph Database Support**: FalkorDB backend for graph storage
- **LLM Provider**: Google Gemini for inference, embeddings, and reranking
- **Rich Entity Types**: Built-in entity types including Preferences, Requirements, Procedures, Locations, Events, Organizations, Documents, and more for structured knowledge extraction
- **HTTP Transport**: Default HTTP transport with MCP endpoint at `/mcp/` for broad client compatibility
- **Queue-based Processing**: Asynchronous episode processing with configurable concurrency limits

## Quick Start

### Clone the Graphiti GitHub repo

```bash
git clone https://github.com/getzep/graphiti.git
```

or

```bash
gh repo clone getzep/graphiti
```

### For Claude Desktop and other `stdio` only clients

1. Note the full path to this directory.

```
cd graphiti && pwd
```

2. Install the [Graphiti prerequisites](#prerequisites).

3. Configure Claude, Cursor, or other MCP client to use [Graphiti with a `stdio` transport](#integrating-with-mcp-clients). See the client documentation on where to find their MCP configuration files.

### For Cursor and other HTTP-enabled clients

1. Change directory to the `mcp_server` directory

`cd graphiti/mcp_server`

2. Start the combined FalkorDB + MCP server using Docker Compose (recommended)

```bash
docker compose up
```

This starts both FalkorDB and the MCP server in a single container.

4. Point your MCP client to `http://localhost:8000/mcp/`

## Installation

### Prerequisites

1. Docker and Docker Compose (for the default FalkorDB setup)
2. Google Gemini API key for LLM operations
3. (Optional) Python 3.10+ if running the MCP server standalone with an external FalkorDB instance

### Setup

1. Clone the repository and navigate to the mcp_server directory
2. Use `uv` to create a virtual environment and install dependencies:

```bash
# Install uv if you don't have it already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment and install dependencies in one step
uv sync
```

## Configuration

The server can be configured using a `config.yaml` file, environment variables, or command-line arguments (in order of precedence).

### Default Configuration

The MCP server comes with sensible defaults:
- **Transport**: HTTP (accessible at `http://localhost:8000/mcp/`)
- **Database**: FalkorDB (combined in single container with MCP server)
- **LLM**: Google Gemini with model gemini-2.0-flash
- **Embedder**: Google Gemini text-embedding-004
- **Reranker**: Google Gemini gemini-flash-latest

### Database Configuration

#### FalkorDB (Default)

FalkorDB is a Redis-based graph database that comes bundled with the MCP server in a single Docker container. This is the default and recommended setup.

```yaml
database:
  provider: "falkordb"  # Default
  providers:
    falkordb:
      uri: "redis://localhost:6379"
      password: ""  # Optional
      database: "default_db"  # Optional
```

### Configuration File (config.yaml)

The server uses Google Gemini for LLM inference, embeddings, and reranking. Edit `config.yaml` to configure:

```yaml
server:
  transport: "http"  # Default. Options: stdio, http

llm:
  provider: "gemini"
  model: "gemini-2.0-flash"  # Default model

database:
  provider: "falkordb"  # Default
```

### Entity Types

Graphiti MCP Server includes built-in entity types for structured knowledge extraction. These entity types are always enabled and configured via the `entity_types` section in your `config.yaml`:

**Available Entity Types:**

- **Preference**: User preferences, choices, opinions, or selections (prioritized for user-specific information)
- **Requirement**: Specific needs, features, or functionality that must be fulfilled
- **Procedure**: Standard operating procedures and sequential instructions
- **Location**: Physical or virtual places where activities occur
- **Event**: Time-bound activities, occurrences, or experiences
- **Organization**: Companies, institutions, groups, or formal entities
- **Document**: Information content in various forms (books, articles, reports, videos, etc.)
- **Topic**: Subject of conversation, interest, or knowledge domain (used as a fallback)
- **Object**: Physical items, tools, devices, or possessions (used as a fallback)

These entity types are defined in `config.yaml` and can be customized by modifying the descriptions:

```yaml
graphiti:
  entity_types:
    - name: "Preference"
      description: "User preferences, choices, opinions, or selections"
    - name: "Requirement"
      description: "Specific needs, features, or functionality"
    # ... additional entity types
```

The MCP server automatically uses these entity types during episode ingestion to extract and structure information from conversations and documents.

### Environment Variables

The `config.yaml` file supports environment variable expansion using `${VAR_NAME}` or `${VAR_NAME:default}` syntax. Key variables:

- `GOOGLE_API_KEY`: Google Gemini API key (required)
- `FALKORDB_URI`: URI for FalkorDB (default: `redis://localhost:6379`)
- `FALKORDB_PASSWORD`: FalkorDB password (optional)
- `SEMAPHORE_LIMIT`: Episode processing concurrency. See [Concurrency and LLM Provider 429 Rate Limit Errors](#concurrency-and-llm-provider-429-rate-limit-errors)

You can set these variables in a `.env` file in the project directory.

## Running the Server

### Default Setup (FalkorDB Combined Container)

To run the Graphiti MCP server with the default FalkorDB setup:

```bash
docker compose up
```

This starts a single container with:
- HTTP transport on `http://localhost:8000/mcp/`
- FalkorDB graph database on `localhost:6379`
- FalkorDB web UI on `http://localhost:3000`
- Google Gemini for LLM inference, embeddings, and reranking

### Running with FalkorDB

#### Option 1: Using Docker Compose

```bash
# This starts both FalkorDB (Redis-based) and the MCP server
docker compose up
```

#### Option 2: Direct Execution with Existing FalkorDB

```bash
# Set environment variables
export FALKORDB_URI="redis://localhost:6379"
export FALKORDB_PASSWORD=""  # If password protected

# Run with FalkorDB
uv run graphiti_mcp_server.py
```

Or use the FalkorDB configuration file:

```bash
uv run graphiti_mcp_server.py --config config/config-docker-falkordb.yaml
```

### Available Command-Line Arguments

- `--config`: Path to YAML configuration file (default: config.yaml)
- `--model`: Model name to use with Google Gemini (default: gemini-2.0-flash)
- `--temperature`: Temperature setting for the LLM (0.0-2.0)
- `--transport`: Choose the transport method (http or stdio, default: http)
- `--group-id`: Set a namespace for the graph (optional). If not provided, defaults to "main"
- `--destroy-graph`: If set, destroys all Graphiti graphs on startup

### Concurrency and LLM Provider 429 Rate Limit Errors

Graphiti's ingestion pipelines are designed for high concurrency, controlled by the `SEMAPHORE_LIMIT` environment variable. This setting determines how many episodes can be processed simultaneously. Since each episode involves multiple LLM calls (entity extraction, deduplication, summarization), the actual number of concurrent LLM requests will be several times higher.

**Default:** `SEMAPHORE_LIMIT=10` (suitable for Google Gemini free tier)

#### Tuning Guidelines

**Google Gemini:**
- Free tier: 15 RPM → `SEMAPHORE_LIMIT=2-5`
- Paid tier: 1,000+ RPM → `SEMAPHORE_LIMIT=10-30`
- Check your quota at [Google AI Studio](https://aistudio.google.com/)

#### Symptoms

- **Too high**: 429 rate limit errors, increased API costs from parallel processing
- **Too low**: Slow episode throughput, underutilized API quota

#### Monitoring

- Watch logs for `429` rate limit errors
- Monitor episode processing times in server logs
- Check your LLM provider's dashboard for actual request rates
- Track token usage and costs

Set this in your `.env` file:
```bash
SEMAPHORE_LIMIT=10  # Adjust based on your LLM provider tier
```

### Docker Deployment

The Graphiti MCP server can be deployed using Docker with your choice of database backend. The Dockerfile uses `uv` for package management, ensuring consistent dependency installation.

A pre-built Graphiti MCP container is available at: `zepai/knowledge-graph-mcp`

> [!TIP]
> For deploying on Dokploy (an open-source hosting platform), see the [Dokploy Deployment Guide](./DOKPLOY_DEPLOYMENT.md) for detailed instructions.

#### Environment Configuration

Before running Docker Compose, configure your API keys using a `.env` file (recommended):

1. **Create a .env file in the mcp_server directory**:
   ```bash
   cd graphiti/mcp_server
   cp .env.example .env
   ```

2. **Edit the .env file** to set your API key:
   ```bash
   # Required - Google Gemini API key
   GOOGLE_API_KEY=your_google_api_key_here
   ```

**Important**: The `.env` file must be in the `mcp_server/` directory (the parent of the `docker/` subdirectory).

#### Running with Docker Compose

**All commands must be run from the `mcp_server` directory** to ensure the `.env` file is loaded correctly:

```bash
cd graphiti/mcp_server
```

##### FalkorDB Combined Container (Default)

Single container with both FalkorDB and MCP server - simplest option:

```bash
docker compose up
```

Default FalkorDB setup:
- Redis port: `6379`
- Web UI: `http://localhost:3000`
- Connection: `redis://falkordb:6379`

#### Accessing the MCP Server

Once running, the MCP server is available at:
- **HTTP endpoint**: `http://localhost:8000/mcp/`
- **Health check**: `http://localhost:8000/health`

#### Running Docker Compose from a Different Directory

If you run Docker Compose from the `docker/` subdirectory instead of `mcp_server/`, you'll need to modify the `.env` file path in the compose file:

```yaml
# Change this line in the docker-compose file:
env_file:
  - path: ../.env    # When running from mcp_server/

# To this:
env_file:
  - path: .env       # When running from mcp_server/docker/
```

However, **running from the `mcp_server/` directory is recommended** to avoid confusion.

## Integrating with MCP Clients

### VS Code / GitHub Copilot

VS Code with GitHub Copilot Chat extension supports MCP servers. Add to your VS Code settings (`.vscode/mcp.json` or global settings):

```json
{
  "mcpServers": {
    "graphiti": {
      "uri": "http://localhost:8000/mcp/",
      "transport": {
        "type": "http"
      }
    }
  }
}
```

### Other MCP Clients

To use the Graphiti MCP server with other MCP-compatible clients, configure it to connect to the server:

> [!IMPORTANT]
> You will need the Python package manager, `uv` installed. Please refer to the [`uv` install instructions](https://docs.astral.sh/uv/getting-started/installation/).
>
> Ensure that you set the full path to the `uv` binary and your Graphiti project folder.

```json
{
  "mcpServers": {
    "graphiti-memory": {
      "transport": "stdio",
      "command": "/Users/<user>/.local/bin/uv",
      "args": [
        "run",
        "--isolated",
        "--directory",
        "/Users/<user>>/dev/zep/graphiti/mcp_server",
        "--project",
        ".",
        "graphiti_mcp_server.py",
        "--transport",
        "stdio"
      ],
      "env": {
        "FALKORDB_URI": "redis://localhost:6379",
        "GOOGLE_API_KEY": "your-google-api-key",
        "MODEL_NAME": "gemini-2.0-flash"
      }
    }
  }
}
```

For HTTP transport (default), you can use this configuration:

```json
{
  "mcpServers": {
    "graphiti-memory": {
      "transport": "http",
      "url": "http://localhost:8000/mcp/"
    }
  }
}
```

## Available Tools

The Graphiti MCP server exposes the following tools:

- `add_episode`: Add an episode to the knowledge graph (supports text, JSON, and message formats)
- `search_nodes`: Search the knowledge graph for relevant node summaries
- `search_facts`: Search the knowledge graph for relevant facts (edges between entities)
- `delete_entity_edge`: Delete an entity edge from the knowledge graph
- `delete_episode`: Delete an episode from the knowledge graph
- `get_entity_edge`: Get an entity edge by its UUID
- `get_episodes`: Get the most recent episodes for a specific group
- `clear_graph`: Clear all data from the knowledge graph and rebuild indices
- `get_status`: Get the status of the Graphiti MCP server and FalkorDB connection

## Working with JSON Data

The Graphiti MCP server can process structured JSON data through the `add_episode` tool with `source="json"`. This
allows you to automatically extract entities and relationships from structured data:

```

add_episode(
name="Customer Profile",
episode_body="{\"company\": {\"name\": \"Acme Technologies\"}, \"products\": [{\"id\": \"P001\", \"name\": \"CloudSync\"}, {\"id\": \"P002\", \"name\": \"DataMiner\"}]}",
source="json",
source_description="CRM data"
)

```

## Integrating with the Cursor IDE

To integrate the Graphiti MCP Server with the Cursor IDE, follow these steps:

1. Run the Graphiti MCP server using the default HTTP transport:

```bash
uv run graphiti_mcp_server.py --group-id <your_group_id>
```

Hint: specify a `group_id` to namespace graph data. If you do not specify a `group_id`, the server will use "main" as the group_id.

or

```bash
docker compose up
```

2. Configure Cursor to connect to the Graphiti MCP server.

```json
{
  "mcpServers": {
    "graphiti-memory": {
      "url": "http://localhost:8000/mcp/"
    }
  }
}
```

3. Add the Graphiti rules to Cursor's User Rules. See [cursor_rules.md](cursor_rules.md) for details.

4. Kick off an agent session in Cursor.

The integration enables AI assistants in Cursor to maintain persistent memory through Graphiti's knowledge graph
capabilities.

## Integrating with Claude Desktop (Docker MCP Server)

The Graphiti MCP Server uses HTTP transport (at endpoint `/mcp/`). Claude Desktop does not natively support HTTP transport, so you'll need to use a gateway like `mcp-remote`.

1.  **Run the Graphiti MCP server**:

    ```bash
    docker compose up
    # Or run directly with uv:
    uv run graphiti_mcp_server.py
    ```

2.  **(Optional) Install `mcp-remote` globally**:
    If you prefer to have `mcp-remote` installed globally, or if you encounter issues with `npx` fetching the package, you can install it globally. Otherwise, `npx` (used in the next step) will handle it for you.

    ```bash
    npm install -g mcp-remote
    ```

3.  **Configure Claude Desktop**:
    Open your Claude Desktop configuration file (usually `claude_desktop_config.json`) and add or modify the `mcpServers` section as follows:

    ```json
    {
      "mcpServers": {
        "graphiti-memory": {
          // You can choose a different name if you prefer
          "command": "npx", // Or the full path to mcp-remote if npx is not in your PATH
          "args": [
            "mcp-remote",
            "http://localhost:8000/mcp/" // The Graphiti server's HTTP endpoint
          ]
        }
      }
    }
    ```

    If you already have an `mcpServers` entry, add `graphiti-memory` (or your chosen name) as a new key within it.

4.  **Restart Claude Desktop** for the changes to take effect.

## Requirements

- Python 3.10 or higher
- Google Gemini API key (for LLM operations, embeddings, and reranking)
- MCP-compatible client
- Docker and Docker Compose (for the default FalkorDB combined container)

## Telemetry

The Graphiti MCP server uses the Graphiti core library, which includes anonymous telemetry collection. When you initialize the Graphiti MCP server, anonymous usage statistics are collected to help improve the framework.

### What's Collected

- Anonymous identifier and system information (OS, Python version)
- Graphiti version and configuration choices (LLM provider, database backend, embedder type)
- **No personal data, API keys, or actual graph content is ever collected**

### How to Disable

To disable telemetry in the MCP server, set the environment variable:

```bash
export GRAPHITI_TELEMETRY_ENABLED=false
```

Or add it to your `.env` file:

```
GRAPHITI_TELEMETRY_ENABLED=false
```

For complete details about what's collected and why, see the [Telemetry section in the main Graphiti README](../README.md#telemetry).

## License

This project is licensed under the same license as the parent Graphiti project.
