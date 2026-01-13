import sys
sys.path.append("..")
sys.path.append(".")
sys.path.append("src/")
from src.training.utils import process_mm_messages
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/share_data/data1/models/Qwen/Qwen3-8B", trust_remote_code=True)

messages = [
    {"role":"user","content":"test"},
    {"role":"assistant","content":"test","tool_calls":[
                {
            "id": "chatcmpl-tool-1c515355c3a14f879e3eef0a63473afd",
            "function": {
                "arguments": "{\"code\": \"table = {\\n    'a': ['a', 'b', 'c', 'b', 'd'],\\n    'b': ['b', 'c', 'a', 'e', 'c'],\\n    'c': ['c', 'a', 'b', 'b', 'a'],\\n    'd': ['b', 'e', 'b', 'e', 'd'],\\n    'e': ['d', 'b', 'a', 'd', 'c']\\n}\\n\\nS = ['a', 'b', 'c', 'd', 'e']\\nnon_commutative_pairs = []\\n\\nfor i in range(len(S)):\\n    for j in range(i + 1, len(S)):\\n        x = S[i]\\n        y = S[j]\\n        if table[x][j] != table[y][i]:\\n            non_commutative_pairs.append((x, y))\\n\\ninvolved_elements = set()\\nfor pair in non_commutative_pairs:\\n    involved_elements.add(pair[0])\\n    involved_elements.add(pair[1])\\n\\nresult = sorted(involved_elements)\\nprint(result)\"}",
                "name": "execute_code"
            },
            "type": "function",
            "status": "success"
            }
    ], "reasoning_content":"this is a reasoning content!"    },
    {
        "role": "tool",
        "tool_call_id": "chatcmpl-tool-1c515355c3a14f879e3eef0a63473afd",
        "name": "execute_code",
        "content": "{status: \"success\", output: \"['b', 'e']\\n\", file_path: \"/app/code_storage/code_41a88e23.py\", generated_filename: \"code_41a88e23.py\"}"
    },
      {"role":"assistant","content":"test","tool_calls":[
                {
            "id": "chatcmpl-tool-1c515355c3a14f879e3eef0a63473afd",
            "function": {
                "arguments": "{\"code\": \"table = {\\n    'a': ['a', 'b', 'c', 'b', 'd'],\\n    'b': ['b', 'c', 'a', 'e', 'c'],\\n    'c': ['c', 'a', 'b', 'b', 'a'],\\n    'd': ['b', 'e', 'b', 'e', 'd'],\\n    'e': ['d', 'b', 'a', 'd', 'c']\\n}\\n\\nS = ['a', 'b', 'c', 'd', 'e']\\nnon_commutative_pairs = []\\n\\nfor i in range(len(S)):\\n    for j in range(i + 1, len(S)):\\n        x = S[i]\\n        y = S[j]\\n        if table[x][j] != table[y][i]:\\n            non_commutative_pairs.append((x, y))\\n\\ninvolved_elements = set()\\nfor pair in non_commutative_pairs:\\n    involved_elements.add(pair[0])\\n    involved_elements.add(pair[1])\\n\\nresult = sorted(involved_elements)\\nprint(result)\"}",
                "name": "execute_code"
            },
            "type": "function",
            "status": "success"
            }
    ], "reasoning_content":"this is second reasoning content!"    },
]

tools = [
  {
    "type": "function",
    "function": {
      "name": "execute_code",
      "description": "Execute Python code in the conda environment. For short code snippets only. For longer code, use initialize_code_file and append_to_code_file instead.",
      "parameters": {
        "type": "object",
        "properties": {
          "code": {
            "type": "string",
            "description": "Python code to execute"
          },
          "filename": {
            "type": "string",
            "description": "Optional: Name of the file to save the code (default: generated UUID)"
          }
        },
        "required": [
          "code"
        ]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "initialize_code_file",
      "description": "Create a new Python file with initial content. Use this as the first step for longer code that may exceed token limits. Follow with append_to_code_file for additional code.",
      "parameters": {
        "type": "object",
        "properties": {
          "content": {
            "type": "string",
            "description": "Initial content to write to the file"
          },
          "filename": {
            "type": "string",
            "description": "Optional: Name of the file (default: generated UUID)"
          }
        },
        "required": [
          "content"
        ]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "append_to_code_file",
      "description": "Append content to an existing Python code file. Use this to add more code to a file created with initialize_code_file, allowing you to build up larger code bases in parts.",
      "parameters": {
        "type": "object",
        "properties": {
          "file_path": {
            "type": "string",
            "description": "Full path to the file"
          },
          "content": {
            "type": "string",
            "description": "Content to append to the file"
          }
        },
        "required": [
          "file_path",
          "content"
        ]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "execute_code_file",
      "description": "Execute an existing Python file. Use this as the final step after building up code with initialize_code_file and append_to_code_file.",
      "parameters": {
        "type": "object",
        "properties": {
          "file_path": {
            "type": "string",
            "description": "Full path to the Python file to execute"
          }
        },
        "required": [
          "file_path"
        ]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "read_code_file",
      "description": "Read the content of an existing Python code file. Use this to verify the current state of a file before appending more content or executing it.",
      "parameters": {
        "type": "object",
        "properties": {
          "file_path": {
            "type": "string",
            "description": "Full path to the file to read"
          }
        },
        "required": [
          "file_path"
        ]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "install_dependencies",
      "description": "Install Python dependencies in the conda environment",
      "parameters": {
        "type": "object",
        "properties": {
          "packages": {
            "type": "array",
            "items": {
              "type": "string"
            },
            "description": "List of packages to install"
          }
        },
        "required": [
          "packages"
        ]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "check_installed_packages",
      "description": "Check if packages are installed in the conda environment",
      "parameters": {
        "type": "object",
        "properties": {
          "packages": {
            "type": "array",
            "items": {
              "type": "string"
            },
            "description": "List of packages to check"
          }
        },
        "required": [
          "packages"
        ]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "configure_environment",
      "description": "Change the environment configuration settings",
      "parameters": {
        "type": "object",
        "properties": {
          "type": {
            "type": "string",
            "enum": [
              "conda",
              "venv",
              "venv-uv"
            ],
            "description": "Type of Python environment"
          },
          "conda_name": {
            "type": "string",
            "description": "Name of the conda environment (required if type is 'conda')"
          },
          "venv_path": {
            "type": "string",
            "description": "Path to the virtualenv (required if type is 'venv')"
          },
          "uv_venv_path": {
            "type": "string",
            "description": "Path to the UV virtualenv (required if type is 'venv-uv')"
          }
        },
        "required": [
          "type"
        ]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_environment_config",
      "description": "Get the current environment configuration",
      "parameters": {
        "type": "object",
        "properties": {}
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "run_command",
      "description": "Run a command on this linux machine",
      "parameters": {
        "type": "object",
        "properties": {
          "command": {
            "type": "string",
            "description": "Command with args"
          },
          "workdir": {
            "type": "string",
            "description": "Optional, current working directory"
          },
          "stdin": {
            "type": "string",
            "description": "Optional, text to pipe into the command's STDIN. For example, pass a python script to python3. Or, pass text for a new file to the cat command to create it!"
          }
        },
        "required": [
          "command"
        ]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "read_file",
      "description": "Read the complete contents of a file from the file system. Handles various text encodings and provides detailed error messages if the file cannot be read. Use this tool when you need to examine the contents of a single file. Use the 'head' parameter to read only the first N lines of a file, or the 'tail' parameter to read only the last N lines of a file. Only works within allowed directories.",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {
            "type": "string"
          },
          "tail": {
            "type": "number",
            "description": "If provided, returns only the last N lines of the file"
          },
          "head": {
            "type": "number",
            "description": "If provided, returns only the first N lines of the file"
          }
        },
        "required": [
          "path"
        ],
        "additionalProperties": False,
        "$schema": "http://json-schema.org/draft-07/schema#"
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "read_multiple_files",
      "description": "Read the contents of multiple files simultaneously. This is more efficient than reading files one by one when you need to analyze or compare multiple files. Each file's content is returned with its path as a reference. Failed reads for individual files won't stop the entire operation. Only works within allowed directories.",
      "parameters": {
        "type": "object",
        "properties": {
          "paths": {
            "type": "array",
            "items": {
              "type": "string"
            }
          }
        },
        "required": [
          "paths"
        ],
        "additionalProperties": False,
        "$schema": "http://json-schema.org/draft-07/schema#"
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "write_file",
      "description": "Create a new file or completely overwrite an existing file with new content. Use with caution as it will overwrite existing files without warning. Handles text content with proper encoding. Only works within allowed directories.",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {
            "type": "string"
          },
          "content": {
            "type": "string"
          }
        },
        "required": [
          "path",
          "content"
        ],
        "additionalProperties": False,
        "$schema": "http://json-schema.org/draft-07/schema#"
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "edit_file",
      "description": "Make line-based edits to a text file. Each edit replaces exact line sequences with new content. Returns a git-style diff showing the changes made. Only works within allowed directories.",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {
            "type": "string"
          },
          "edits": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "oldText": {
                  "type": "string",
                  "description": "Text to search for - must match exactly"
                },
                "newText": {
                  "type": "string",
                  "description": "Text to replace with"
                }
              },
              "required": [
                "oldText",
                "newText"
              ],
              "additionalProperties": False
            }
          },
          "dryRun": {
            "type": "boolean",
            "default": False,
            "description": "Preview changes using git-style diff format"
          }
        },
        "required": [
          "path",
          "edits"
        ],
        "additionalProperties": False,
        "$schema": "http://json-schema.org/draft-07/schema#"
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "create_directory",
      "description": "Create a new directory or ensure a directory exists. Can create multiple nested directories in one operation. If the directory already exists, this operation will succeed silently. Perfect for setting up directory structures for projects or ensuring required paths exist. Only works within allowed directories.",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {
            "type": "string"
          }
        },
        "required": [
          "path"
        ],
        "additionalProperties": False,
        "$schema": "http://json-schema.org/draft-07/schema#"
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "list_directory",
      "description": "Get a detailed listing of all files and directories in a specified path. Results clearly distinguish between files and directories with [FILE] and [DIR] prefixes. This tool is essential for understanding directory structure and finding specific files within a directory. Only works within allowed directories.",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {
            "type": "string"
          }
        },
        "required": [
          "path"
        ],
        "additionalProperties": False,
        "$schema": "http://json-schema.org/draft-07/schema#"
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "list_directory_with_sizes",
      "description": "Get a detailed listing of all files and directories in a specified path, including sizes. Results clearly distinguish between files and directories with [FILE] and [DIR] prefixes. This tool is useful for understanding directory structure and finding specific files within a directory. Only works within allowed directories.",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {
            "type": "string"
          },
          "sortBy": {
            "type": "string",
            "enum": [
              "name",
              "size"
            ],
            "default": "name",
            "description": "Sort entries by name or size"
          }
        },
        "required": [
          "path"
        ],
        "additionalProperties": False,
        "$schema": "http://json-schema.org/draft-07/schema#"
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "directory_tree",
      "description": "Get a recursive tree view of files and directories as a JSON structure. Each entry includes 'name', 'type' (file/directory), and 'children' for directories. Files have no children array, while directories always have a children array (which may be empty). The output is formatted with 2-space indentation for readability. Only works within allowed directories.",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {
            "type": "string"
          }
        },
        "required": [
          "path"
        ],
        "additionalProperties": False,
        "$schema": "http://json-schema.org/draft-07/schema#"
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "move_file",
      "description": "Move or rename files and directories. Can move files between directories and rename them in a single operation. If the destination exists, the operation will fail. Works across different directories and can be used for simple renaming within the same directory. Both source and destination must be within allowed directories.",
      "parameters": {
        "type": "object",
        "properties": {
          "source": {
            "type": "string"
          },
          "destination": {
            "type": "string"
          }
        },
        "required": [
          "source",
          "destination"
        ],
        "additionalProperties": False,
        "$schema": "http://json-schema.org/draft-07/schema#"
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "search_files",
      "description": "Recursively search for files and directories matching a pattern. Searches through all subdirectories from the starting path. The search is case-insensitive and matches partial names. Returns full paths to all matching items. Great for finding files when you don't know their exact location. Only searches within allowed directories.",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {
            "type": "string"
          },
          "pattern": {
            "type": "string"
          },
          "excludePatterns": {
            "type": "array",
            "items": {
              "type": "string"
            },
            "default": []
          }
        },
        "required": [
          "path",
          "pattern"
        ],
        "additionalProperties": False,
        "$schema": "http://json-schema.org/draft-07/schema#"
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_file_info",
      "description": "Retrieve detailed metadata about a file or directory. Returns comprehensive information including size, creation time, last modified time, permissions, and type. This tool is perfect for understanding file characteristics without reading the actual content. Only works within allowed directories.",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {
            "type": "string"
          }
        },
        "required": [
          "path"
        ],
        "additionalProperties": False,
        "$schema": "http://json-schema.org/draft-07/schema#"
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "list_allowed_directories",
      "description": "Returns the list of directories that this server is allowed to access. Use this to understand which directories are available before trying to access files.",
      "parameters": {
        "type": "object",
        "properties": {},
        "required": []
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "search_web",
      "description": "执行网络搜索并返回结果。\n\nArgs:\n    query: 搜索查询词\n    num_results: 返回结果数量，默认为10\n    engine: 搜索引擎类型，可选值:\n        - \"auto\": 自动选择最佳可用搜索引擎（默认）\n        - \"google\": 优先使用Google搜索（需要API密钥）\n        - \"duckduckgo\": 优先使用DuckDuckGo搜索\n        - \"exa\": 优先使用Exa搜索（需要API密钥）\n        - \"bing\": 优先使用Bing搜索（需要API密钥）\n        - \"baidu\": 优先使用百度搜索（需要API密钥）\n        - \"wiki\": 优先使用本地Wiki搜索\n        - \"jina\": 使用Jina AI搜索（s.jina.ai）\n    use_jina: 是否使用Jina AI搜索增强结果，默认为True\n",
      "parameters": {
        "properties": {
          "query": {
            "title": "Query",
            "type": "string"
          },
          "num_results": {
            "default": 10,
            "title": "Num Results",
            "type": "integer"
          },
          "engine": {
            "default": "auto",
            "title": "Engine",
            "type": "string"
          },
          "use_jina": {
            "default": True,
            "title": "Use Jina",
            "type": "boolean"
          }
        },
        "required": [
          "query"
        ],
        "title": "search_webArguments",
        "type": "object"
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "fetch_url",
      "description": "抓取网页内容并返回文本。\n\nArgs:\n    url: 要抓取的网页URL\n    timeout: 超时时间，单位为秒，默认10秒\n    use_jina: 是否使用Jina Reader优化内容（适合LLM），默认为True\n    with_image_alt: 是否为图片生成替代文本描述，默认为False\n",
      "parameters": {
        "properties": {
          "url": {
            "title": "Url",
            "type": "string"
          },
          "timeout": {
            "default": 10,
            "title": "Timeout",
            "type": "integer"
          },
          "use_jina": {
            "default": True,
            "title": "Use Jina",
            "type": "boolean"
          },
          "with_image_alt": {
            "default": False,
            "title": "With Image Alt",
            "type": "boolean"
          }
        },
        "required": [
          "url"
        ],
        "title": "fetch_urlArguments",
        "type": "object"
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_available_engines",
      "description": "获取当前可用的搜索引擎列表和状态",
      "parameters": {
        "properties": {},
        "title": "get_available_enginesArguments",
        "type": "object"
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "manage_context",
      "description": "执行记忆管理、过程总结、任务规划，并动态管理context。\n   应该在任务开始、每个大步骤完成后、以及调用。不允许和其他工具同时调用。\n\nArgs:\n    memoryx_path(required): 历史文件路径前缀\n    plan_steps(required): 完整的plan，请先规划一个完整、清晰、尽可能少步骤的计划，并将plan的每个步骤填写到plan_steps中，并用[completed]、[partially completed]、[failed]来标注每个步骤的完成情况\n    next_step_goal(required): 下一个步骤目标\n    recommend_tools(required): 下一个步骤可能需要使用的工具名称列表\n    messages(required): 请填写None由外部注入\n    chosen_tools(required): 请填写None由外部注入\n\n    \n",
      "parameters": {
        "properties": {
          "plan_steps": {
            "description": "完整的plan，请先规划一个完整、清晰、尽可能少步骤的计划，并将plan的每个步骤填写到plan_steps中，并用[completed]、[partially completed]、[failed]来标注每个步骤的完成情况",
            "items": {
              "type": "string"
            },
            "title": "Plan Steps",
            "type": "array"
          },
          "next_step_goal": {
            "description": "下一个步骤需要实现的目标",
            "title": "Next Step Goal",
            "type": "string"
          },
          "recommend_tools": {
            "description": "下一个步骤可能需要使用的工具名称列表",
            "items": {
              "type": "string"
            },
            "title": "Recommend Tools",
            "type": "array"
          },
          "messages": {
            "anyOf": [
              {
                "items": {
                  "additionalProperties": True,
                  "type": "object"
                },
                "type": "array"
              },
              {
                "type": "null"
              }
            ],
            "description": "请填写None由外部注入",
            "title": "Messages"
          },
          "chosen_tools": {
            "anyOf": [
              {
                "items": {
                  "additionalProperties": True,
                  "type": "object"
                },
                "type": "array"
              },
              {
                "type": "null"
              }
            ],
            "description": "请填写None由外部注入",
            "title": "Chosen Tools"
          }
        },
        "required": [
          "plan_steps",
          "next_step_goal",
          "recommend_tools",
          "messages",
          "chosen_tools"
        ],
        "title": "manage_contextArguments",
        "type": "object"
      }
    }
  }
]

msg = process_mm_messages(messages,tokenizer,tools=tools)
print(msg)
import pdb;pdb.set_trace()