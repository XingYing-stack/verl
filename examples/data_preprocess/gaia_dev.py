"""
Preprocess the GAIA dev dataset to parquet format
"""

import argparse
import os
import re
import pandas as pd
import datasets

from copy import deepcopy


gaia_tools_content = [{'type': 'function', 'function': {'name': 'execute_code', 'description': 'Execute Python code in the conda environment. Packages installed in the environment: PyPDF2 geopy PyMuPDF docx2txt pdfminer.six rdkit python-chess stockfish yfinance CoolProp seaborn python-pptx python-docx pdfplumber geopandas biopython pubchempy googletrans pyshp selenium waybackpy networkx wbdata and their dependencies and other common python packages', 'parameters': {'type': 'object', 'properties': {'code': {'type': 'string', 'description': 'Python code to execute'}}, 'required': ['code']}}}, {'type': 'function', 'function': {'name': 'read_file', 'description': '\n    universal file processing tool that converts various formats to structured markdown.\n\n    ## Supported Formats\n    local_source:\n    - Office documents: Word, Excel, PowerPoint, PDF\n    - Images: AI OCR and AI description\n    - Media: audio transcription through google api, AI video description\n    - Archives: ZIP, RAR extraction\n    online_source:\n    - Online images\n    ', 'parameters': {'properties': {'uri': {'description': "The URI of the file to convert to markdown. Examples: 'file:///path/to/document.pdf'", 'title': 'Uri', 'type': 'string'}, 'purpose': {'description': 'The purpose of the file processing, what information you want to get from the file', 'title': 'Purpose', 'type': 'string'}, 'process_type': {'default': 'others', 'description': "The type of the file to process.'", 'enum': ['audio', 'image', 'video', 'others'], 'title': 'Process Type', 'type': 'string'}}, 'required': ['uri', 'purpose'], 'title': 'read_fileArguments', 'type': 'object'}}}, {'type': 'function', 'function': {'name': 'fetch_url', 'description': '\n    Fetch webpage(s) and online pdf(s) and return the content with AI summary.\n    Supports parallel processing of multiple (at most 3) URLs.\n    Uses Jina service for fetching content.\n    ', 'parameters': {'properties': {'url': {'description': 'The URL(s) of the webpage(s) and online pdf(s) to visit. Can be a single URL or a list of URLs.', 'items': {'type': 'string'}, 'title': 'Url', 'type': 'array'}, 'purpose': {'description': 'The purpose of the visit, what information you want to get from the webpage', 'title': 'Purpose', 'type': 'string'}}, 'required': ['url', 'purpose'], 'title': 'fetch_urlArguments', 'type': 'object'}}}, {'type': 'function', 'function': {'name': 'search', 'description': '\n    Google search supports parallel processing of multiple (at most 3) queries. \n    The tool retrieves the top 10 results for each query in parallel.\n    ', 'parameters': {'properties': {'query': {'description': 'Array of query strings. Include multiple complementary search queries in a single call.', 'items': {'type': 'string'}, 'title': 'Query', 'type': 'array'}}, 'required': ['query'], 'title': 'searchArguments', 'type': 'object'}}}]


FILES_DIR="/app/data/gaia_validation"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/workspace/fanshengda/verl/input_data/gaia_dev")
    parser.add_argument("--metadata_path", default="/workspace/fanshengda/AgentCPM-MCP/evaluation/benchmarks/gaia/dev.json")
    parser.add_argument("--llm_judge", action="store_true")
    parser.add_argument("--llm_judge_model", default="kimi-k2-0905-preview")
    parser.add_argument("--prompt_type", type=str, default="agentcpm", choices=["agentcpm", "tongyi"])

    args = parser.parse_args()


    if args.prompt_type == "agentcpm":
        print('using agentcpm prompt')
        MIXED_PROMPT = """# General Objective

        You accomplish a given task iteratively, breaking it down into clear steps and working through them methodically.

        ## Task Strategy

        1. **Analyze the user's request** to clarify the task objective, break it down into clear sub-goals, and arrange them in logical order.
        2. **If the task does not require tool use, think step by step and answer the user directly.**
        3. **If the task requires tool use, develop a concise step-by-step plan** (e.g., 1., 2., 3.), with each step corresponding to a specific sub-goal, obey tool-use guidelines to solve the task.

        ## Tool-Use Guidelines
        4. **Call only one tool per step**, prioritizing the tool that best advances the current sub-goal.
        5. **Tool Prioritization Rule: To access any online resource via a URL (like http:// or https://), including webpages and online PDFs, you must use the fetch_url tool. The read_file tool should only be used for local file URIs (e.g., file:///...).
        6. **After each tool call, stop responding immediately** and wait for user feedback or tool results. Do not assume results or continue analysis.
        7. **Extract and summarize key information from tool results** to inform the next step.
        8. **Adjust your plan promptly when new information or challenges arise**, ensuring all sub-goals are covered and nothing is missed.
        9. **For key conclusions, you must cross-validate using multiple tools or methods** to ensure the accuracy and consistency of the answer.
        10. **After you have verified the answer, output the final answer in the specified format**.

        ## Answer Format
        - **Answers should be direct and concise**, preferably using single words, numbers with commas and unit, or brief phrases.
        - **Strictly follow the format requirements**, wrapping the final answer in `<answer>
        </answer>` tags.

        **Your goal: Minimize unnecessary thinking, act decisively, continuously use tools to gather information, and cross-validate with multiple tools until you can confidently provide the most concise and accurate answer.**

        Where:
        - `tool_call_name` must be an exact match to one of the available tools
        - `tool_call_arguments` must be valid JSON that strictly follows the tool's Parameters Schema
        - Only one tool call is allowed per responses
        """

        gaia_system_prompt_content = MIXED_PROMPT.format(answer_schema="answer")

        MCP_USER_PROMPT_FOR_FILE = """Your task is to answer the user's question: {query}

        The filepath to the file you need in this task: "{task_dir}/{filename}" 
        """

        MCP_USER_PROMPT = """Your task is to answer the user's question: {query}."""
    elif args.prompt_type == 'tongyi':
        print('using tongyi prompt')
        gaia_system_prompt_content = "You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. For every request, synthesize information from credible, diverse sources to deliver a comprehensive, accurate, and objective response. When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final answer within <answer></answer> tags."

        MCP_USER_PROMPT_FOR_FILE = """{query} 
        The filepath to the file you need in this task: "{task_dir}/{filename}" """

        MCP_USER_PROMPT = """{query}"""

    data_source = "gaia_dev"

    if args.metadata_path.endswith(".json"):
        metadata = pd.read_json(args.metadata_path, lines=False)
    elif args.metadata_path.endswith(".jsonl"):
        metadata = pd.read_json(args.metadata_path, lines=True)


    # add a row to each data item that represents a unique id

    def process_fn(example, idx):
        question = example['Question']

        system_message = deepcopy(gaia_system_prompt_content)
        if 'file_name' in example and  example['file_name']:
            user_prompt = MCP_USER_PROMPT_FOR_FILE.format(query=question, task_dir=FILES_DIR, filename=example['file_name'])

        else:
            user_prompt = MCP_USER_PROMPT.format(query=question)
        solution = str(example.get("Final answer") or example.get("answer"))
        
        if args.llm_judge:
            print('using llm judge')

            reward_model = {
                "style": "llm",
                "model": args.llm_judge_model,
                "ground_truth": solution
            }
        else:
            print('using rule judge')

            reward_model = {
                "style": "rule",
                "ground_truth": solution
            }

        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                "role": "user",
                "content": user_prompt,
            }],
            "ability": "fact-reasoning",
            "reward_model": reward_model,
            "extra_info": {
                'split': 'dev',
                'index': idx,
                'task_id': example['task_id'],
                'Level': example['Level'],
                'question': question,
                'reward_model': reward_model,
                "need_tools_kwargs": True,
                # Ensure non-empty per-tool kwargs to avoid PyArrow struct<> write error
                # Ref: ArrowNotImplementedError: Cannot write struct type with no child field
                "tools_kwargs": {
                    item['function']['name']: {
                        "create_kwargs": {"ground_truth": solution}
                    }
                    for item in gaia_tools_content
                },
            }
        }
        return data


    train_dataset = pd.DataFrame([
        process_fn(row, idx) for idx, row in metadata.iterrows()
    ])


    local_dir = args.local_dir

    output_file_name = "dev_TextOnly_LLMJudge_docker25_tongyi.parquet"
    train_dataset.to_parquet(os.path.join(local_dir, output_file_name))
    print('path:', os.path.join(local_dir, output_file_name))
