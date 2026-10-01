# Sidekick - AI Personal Co-Worker

An intelligent AI assistant powered by LangGraph that uses a multi-agent architecture (Planner, Worker, Evaluator) to complete tasks with built-in quality control and user authentication.

## 🌟 Features

- **Multi-Agent System**: Planner → Worker → Evaluator architecture with iterative refinement
- **Smart Planning**: Asks up to 3 clarifying questions before starting work
- **Quality Control**: Evaluator ensures success criteria are met before completion
- **User Authentication**: Multi-user support with isolated conversation sessions
- **Persistent Memory**: SQLite-backed conversation history per user
- **Rich Tool Set**: Web browsing, search, Wikipedia, Python execution, file operations, notifications

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Input                           │
│          (Message + Optional Success Criteria)              │
└─────────────────────┬───────────────────────────────────────┘
                      ↓
              ┌───────────────┐
              │    PLANNER    │ ← Clarifies ambiguous requests
              └───────┬───────┘
                      ↓
              [Need Clarification?]
                   ↙     ↘
              YES          NO
               ↓            ↓
          Ask User    ┌──────────┐
               ↓      │  WORKER  │ ← Executes task with tools
          [Answer]    └─────┬────┘
               ↓            ↓
               └───→  ┌──────────┐
                      │  TOOLS   │ ← Browser, Search, Code, Files
                      └─────┬────┘
                            ↓
                      ┌───────────┐
                      │ EVALUATOR │ ← Checks success criteria
                      └─────┬─────┘
                            ↓
                   [Criteria Met?]
                      ↙         ↘
                  YES             NO
                   ↓               ↓
              Return Result    Back to Worker
                                (with feedback)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- UV package manager
- OpenAI API key

### Installation

1. **Navigate to project directory**:

```bash
cd Sidekick_hopeogbons
```

2. **Install dependencies with UV**:

```bash
uv pip install gradio langgraph langchain-openai langchain-community \
    langchain-experimental playwright aiosqlite python-dotenv \
    pydantic requests
```

3. **Install Playwright browsers**:

```bash
python -m playwright install chromium
```

4. **Create `.env` file**:

```bash
cat > .env << EOF
GROQ_API_KEY=your_GROQ_API_KEy_here
SERPER_API_KEY=your_google_serper_key  # Optional
PUSHOVER_TOKEN=your_pushover_token      # Optional
PUSHOVER_USER=your_pushover_user        # Optional
EOF
```

5. **Run the application**:

```bash
python app.py
```

6. **Access the UI**:
   Open your browser to `http://127.0.0.1:7860`

## 🔐 Authentication

### Default Users

| Username | Password | Purpose    |
| -------- | -------- | ---------- |
| test     | test     | Demo user  |
| admin    | admin    | Admin user |

### Adding New Users

Edit `app.py` and modify the `authenticate()` function:

```python
def authenticate(username: str, password: str) -> bool:
    valid_users = {
        "alice": "password1",
        "bob": "password2",
        "admin": "admin123",
        "yourname": "yourpassword",  # Add here
    }
    return valid_users.get(username) == password
```

⚠️ **Security Note**: This is a simple demo authentication. For production, use proper authentication with hashed passwords and secure session management.

## 🛠️ Available Tools

| Tool                   | Description                    | Requires             |
| ---------------------- | ------------------------------ | -------------------- |
| **Playwright Browser** | Navigate websites, scrape data | Chromium installed   |
| **Web Search**         | Google search via Serper API   | SERPER_API_KEY       |
| **Wikipedia**          | Query Wikipedia directly       | Built-in             |
| **Python REPL**        | Execute Python code            | Built-in             |
| **File Management**    | Read/write files in sandbox    | Built-in             |
| **Push Notifications** | Send Pushover notifications    | PUSHOVER credentials |

## 📖 Usage Examples

### Example 1: Research Task

**User Input**:

```
Message: Research the latest AI developments in 2024
Success Criteria: Include at least 3 major breakthroughs
```

**Flow**:

1. Planner checks if request is clear
2. Worker uses web search and browsing tools
3. Evaluator verifies 3+ breakthroughs are included
4. Returns results or requests more work

### Example 2: Data Analysis

**User Input**:

```
Message: Calculate statistics for numbers 1 to 100
Success Criteria: Show mean, median, and standard deviation
```

**Flow**:

1. Planner verifies what statistics are needed
2. Worker uses Python REPL tool
3. Evaluator checks all three statistics are present
4. Returns complete analysis

### Example 3: Ambiguous Request

**User Input**:

```
Message: Get me information
Success Criteria: Detailed and accurate
```

**Flow**:

1. Planner asks: "What specific information are you looking for?"
2. User clarifies: "Weather forecast for New York"
3. Worker searches and retrieves weather data
4. Evaluator checks accuracy and detail
5. Returns weather forecast

## 📁 Project Structure

```
Sidekick_hopeogbons/
├── app.py                 # Gradio UI and authentication
├── sidekick.py           # LangGraph agents (Planner, Worker, Evaluator)
├── sidekick_tools.py     # Tool definitions
├── .env                  # Environment variables (create this)
├── memory.db             # SQLite database (auto-created)
├── memory.db-wal         # SQLite WAL file (auto-created)
├── memory.db-shm         # SQLite shared memory (auto-created)
└── sandbox/              # File operations directory (auto-created)
```

## ⚙️ Configuration

### Changing AI Models

Edit `sidekick.py` (lines 73-77):

```python
worker_llm = ChatOpenAI(model="gpt-4o")           # Change to gpt-4o
evaluator_llm = ChatOpenAI(model="gpt-4o-mini")   # Keep lightweight
planner_llm = ChatOpenAI(model="gpt-4o-mini")     # Keep lightweight
```

### Adjusting Planner Questions

Edit `sidekick.py` (line 82):

```python
max_questions = 5  # Increase from 3 to 5
```

### Database Location

Default: `memory.db` in the same directory as `sidekick.py`

To change, edit `sidekick.py` (line 68):

```python
self.memory_context = AsyncSqliteSaver.from_conn_string("path/to/your/database.db")
```

## 🔧 Troubleshooting

### Issue: Playwright Browser Not Found

**Solution**:

```bash
python -m playwright install chromium
```

### Issue: Database Threading Errors

**Solution**: Delete database files and restart

```bash
rm memory.db memory.db-wal memory.db-shm
python app.py
```

### Issue: OpenAI API Errors

**Solution**: Check your `.env` file has valid `GROQ_API_KEY`

### Issue: Web Search Not Working

**Solution**: Add `SERPER_API_KEY` to `.env` or tools will skip search operations

## 🧪 Testing Multi-User Sessions

1. Open the app in Chrome and login as `alice`
2. Start a conversation
3. Open the app in an Incognito window and login as `bob`
4. Notice Bob has a separate conversation history
5. Both users' sessions are persisted independently in the database

## 🔒 Security Best Practices

For production deployment:

1. **Authentication**:

   - Use OAuth2, JWT, or proper authentication framework
   - Hash passwords with bcrypt/argon2
   - Never store plaintext passwords

2. **API Keys**:

   - Use environment variables or secret management service
   - Rotate keys regularly
   - Limit API key permissions

3. **Database**:

   - Use proper database with access controls
   - Regular backups
   - Encrypt sensitive data

4. **Network**:

   - Deploy with HTTPS/TLS
   - Use proper CORS settings
   - Rate limiting

5. **File Operations**:
   - Strictly limit sandbox directory
   - Validate all file paths
   - Scan uploads for malware

## 🧑‍💻 Development

### Adding a New Tool

1. **Define the tool** in `sidekick_tools.py`:

```python
def my_custom_tool(input: str) -> str:
    """Description of what this tool does"""
    # Your implementation
    result = process(input)
    return result
```

2. **Register the tool** in `other_tools()`:

```python
async def other_tools():
    # ... existing tools ...

    custom_tool = Tool(
        name="my_custom_tool",
        func=my_custom_tool,
        description="When to use this tool"
    )

    return file_tools + [push_tool, tool_search, python_repl, wiki_tool, custom_tool]
```

### Customizing Agent Behavior

Edit system prompts in `sidekick.py`:

- **Planner**: Lines 84-98
- **Worker**: Lines 152-176
- **Evaluator**: Lines 226-246

## 📊 How It Works

### State Management

The system uses LangGraph's state management with these fields:

```python
class State(TypedDict):
    messages: List[Any]              # Conversation history
    success_criteria: str            # User's success criteria
    feedback_on_work: Optional[str]  # Evaluator feedback
    success_criteria_met: bool       # Task completion flag
    user_input_needed: bool          # Needs clarification flag
    clarifying_questions_asked: int  # Counter for planner
    planning_complete: bool          # Planning phase done
```

### Conversation Flow

1. **User sends message** → Stored in `messages`
2. **Planner analyzes** → May increment `clarifying_questions_asked`
3. **Worker executes** → Uses tools, adds to `messages`
4. **Evaluator judges** → Sets `success_criteria_met` or provides `feedback_on_work`
5. **Loop continues** → Until success or user input needed

### Database Schema

Each user's conversations are stored with:

- `thread_id`: Username
- `checkpoint_id`: Unique per conversation turn
- `state`: Serialized conversation state

## 📝 License

MIT License

Copyright (c) 2025 Sidekick Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## 🤝 Contributing

We are open to collaborating and welcome contributions from the community! Whether you're fixing bugs, adding new features, improving documentation, or sharing ideas, your contributions are greatly appreciated.

### How to Contribute

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** with clear, commented code
3. **Test thoroughly** to ensure nothing breaks
4. **Update documentation** if you're adding new features
5. **Submit a pull request** with a clear description of your changes

### Areas We'd Love Help With

- 🛠️ Adding new tools and integrations
- 🎨 Improving the UI/UX
- 📚 Enhancing documentation and examples
- 🐛 Bug fixes and performance improvements
- 🔒 Security enhancements
- 🌍 Internationalization/localization
- ✅ Writing tests

### Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the community
- Show empathy towards other community members

We believe in the power of collaboration and look forward to building something amazing together!

## 📧 Support

For issues, questions, or feature requests, please [add contact/issue information].

---

**Powered by LangGraph** 🦜🕸️ | **Built with OpenAI GPT-4** 🤖
