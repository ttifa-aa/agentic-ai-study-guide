"""
Computer Science specific utilities for Track A1.
Includes code detection, algorithm analysis, and CS concept extraction.
"""

import re  # regular expressions for pattern matching in code and algorithm detection
from typing import Dict, List, Optional, Tuple, Set  # type hints for function signatures
from dataclasses import dataclass  # for creating structured data classes
from enum import Enum  # for enumerated types like algorithm categories

from config.settings import CODE_PATTERNS, COMPLEXITY_PATTERNS, CS_SUBJECTS
# import configuration constants from settings module


class AlgorithmType(str, Enum):
    """Enumeration of algorithm types for classification."""
    SORTING = "sorting"                      # algorithms that arrange data in order
    SEARCHING = "searching"                  # algorithms that find elements in data structures
    GRAPH = "graph"                          # algorithms operating on graph data structures
    TREE = "tree"                            # algorithms for tree traversal and manipulation
    DYNAMIC = "dynamic_programming"          # algorithms using memoization and optimal substructure
    GREEDY = "greedy"                        # algorithms making locally optimal choices
    DIVIDE_CONQUER = "divide_and_conquer"    # algorithms that break problems into subproblems
    BACKTRACKING = "backtracking"            # algorithms that explore possibilities and backtrack
    RECURSIVE = "recursive"                  # algorithms that call themselves
    ITERATIVE = "iterative"                  # algorithms using loops rather than recursion


@dataclass
class CodeBlock:
    """Represents a detected code block with metadata."""
    language: str          # detected programming language (python, java, cpp, etc.)
    code: str              # the actual code content
    start_line: int        # line number where code block starts in source document
    end_line: int          # line number where code block ends in source document
    context: str           # surrounding text context for better understanding


@dataclass
class AlgorithmInfo:
    """Information about a detected algorithm."""
    name: str                           # name of the algorithm
    algorithm_type: AlgorithmType       # category of algorithm
    complexity_time: str                # time complexity in big-o notation
    complexity_space: str               # space complexity in big-o notation
    description: str                    # brief description of what the algorithm does
    steps: List[str]                    # step-by-step breakdown of algorithm execution


def detect_code_language(text: str) -> Optional[str]:
    """
    Detect programming language from code snippet.
    analyzes syntax patterns to identify the programming language.
    
    Args:
        text (str): The text/code to analyze
    
    Returns:
        Optional[str]: Detected language or None if uncertain
    """
    text_sample = text[:500]  # check first 500 chars - enough to identify language
    
    language_scores: Dict[str, int] = {}  # dictionary to store confidence scores per language
    
    for lang, patterns in CODE_PATTERNS.items():  # check each supported language
        score = sum(1 for pattern in patterns if pattern in text_sample)
        # count how many patterns for this language appear in the text
        if score > 0:  # only consider languages with at least one pattern match
            language_scores[lang] = score  # store the score
    
    if language_scores:  # if any language patterns were detected
        # Return language with highest score (most pattern matches)
        return max(language_scores, key=language_scores.get)
        # max with key function returns language with highest confidence score
    
    return None  # return none if no language could be confidently identified


def extract_code_blocks(text: str) -> List[CodeBlock]:
    """
    Extract code blocks from text with language detection.
    identifies both markdown-style and indented code blocks.
    
    Args:
        text (str): The document text
    
    Returns:
        List[CodeBlock]: List of detected code blocks
    """
    code_blocks = []  # initialize empty list for detected code blocks
    
    # Pattern for code blocks (markdown style or indented)
    # note: patterns defined but not directly used - kept for reference
    # actual extraction uses line-by-line parsing for better accuracy
    patterns = [
        r'```(\w*)\n(.*?)```',           # Markdown code blocks with unix line endings
        r'```(\w*)\r?\n(.*?)```',        # Markdown with Windows line endings
        r'^\s{4,}(.+)$'                  # Indented code blocks (line by line)
    ]
    
    lines = text.split('\n')  # split text into lines for processing
    
    # Extract markdown-style code blocks using state machine approach
    in_code_block = False  # flag to track if we're inside a code block
    current_lang = ""      # language specified for current code block
    current_code = []      # lines of code being accumulated
    start_line = 0         # line number where code block starts
    
    for i, line in enumerate(lines):  # process each line with its index
        # Check for markdown code block start/end (triple backticks)
        if line.strip().startswith('```'):
            if not in_code_block:
                # Start of code block - we just encountered opening backticks
                in_code_block = True  # set flag indicating we're now inside a code block
                start_line = i  # record the starting line number
                # Extract language if specified (e.g., ```python)
                lang_match = re.match(r'```(\w+)', line.strip())
                # regex captures the language identifier after the backticks
                current_lang = lang_match.group(1) if lang_match else ""
                # if language specified, store it; otherwise empty string
                current_code = []  # initialize empty list for code lines
            else:
                # End of code block - we encountered closing backticks
                code_text = '\n'.join(current_code)  # join accumulated lines
                # detect language if not explicitly specified
                detected_lang = current_lang or detect_code_language(code_text) or "text"
                # use specified language, or detect it, fallback to "text"
                
                # Get surrounding context (2 lines before and after)
                context_start = max(0, start_line - 2)  # don't go below line 0
                context_end = min(len(lines), i + 3)    # don't exceed total lines
                context = '\n'.join(lines[context_start:context_end])
                # extract context lines for better understanding of code purpose
                
                code_blocks.append(CodeBlock(
                    language=detected_lang,     # detected or specified language
                    code=code_text,             # the actual code content
                    start_line=start_line,      # line where code block begins
                    end_line=i,                 # line where code block ends
                    context=context             # surrounding text context
                ))
                
                in_code_block = False  # reset flag
                current_lang = ""      # clear language
                current_code = []      # clear accumulated code
        
        elif in_code_block:
            # we're inside a code block, accumulate the code lines
            current_code.append(line)  # add current line to code block
    
    return code_blocks  # return list of all detected code blocks


def detect_algorithm_type(text: str) -> Optional[AlgorithmType]:
    """
    Detect algorithm type from description or code.
    uses keyword matching to identify the algorithm paradigm.
    
    Args:
        text (str): Algorithm description or code
    
    Returns:
        Optional[AlgorithmType]: Detected algorithm type
    """
    text_lower = text.lower()  # convert to lowercase for case-insensitive matching
    
    # mapping of algorithm types to their indicator keywords
    algorithm_indicators = {
        AlgorithmType.SORTING: ["sort", "order", "arrange", "bubble", "quick", "merge", "heap"],
        # sorting algorithms - rearrange data in a specific order
        
        AlgorithmType.SEARCHING: ["search", "find", "lookup", "binary", "linear", "locate"],
        # searching algorithms - find elements in collections
        
        AlgorithmType.GRAPH: ["graph", "vertex", "edge", "bfs", "dfs", "dijkstra", "path"],
        # graph algorithms - traverse or analyze graph structures
        
        AlgorithmType.TREE: ["tree", "binary", "bst", "traversal", "inorder", "preorder", "postorder"],
        # tree algorithms - operations on hierarchical tree structures
        
        AlgorithmType.DYNAMIC: ["dynamic", "memoization", "optimal substructure", "overlapping"],
        # dynamic programming - solving problems with overlapping subproblems
        
        AlgorithmType.GREEDY: ["greedy", "local optimal", "minimum spanning", "huffman"],
        # greedy algorithms - make locally optimal choices
        
        AlgorithmType.DIVIDE_CONQUER: ["divide", "conquer", "merge sort", "quick sort"],
        # divide and conquer - break problems into smaller subproblems
        
        AlgorithmType.BACKTRACKING: ["backtrack", "n-queens", "sudoku", "permutation"],
        # backtracking - explore possibilities and undo when needed
        
        AlgorithmType.RECURSIVE: ["recursive", "base case", "recurrence", "call stack"],
        # recursive algorithms - functions that call themselves
        
        AlgorithmType.ITERATIVE: ["iterative", "loop", "while", "for", "iteration"]
        # iterative algorithms - use loops instead of recursion
    }
    
    for algo_type, indicators in algorithm_indicators.items():  # check each algorithm type
        if any(ind in text_lower for ind in indicators):
            # if any indicator keyword is found in the text
            return algo_type  # return the matching algorithm type
    
    return None  # return none if no algorithm type could be identified


def analyze_algorithm_complexity(text: str) -> Tuple[str, str]:
    """
    Analyze algorithm complexity from description.
    extracts both time and space complexity in big-o notation.
    
    Args:
        text (str): Algorithm description
    
    Returns:
        Tuple[str, str]: (time_complexity, space_complexity)
    """
    text_lower = text.lower()  # convert to lowercase for case-insensitive matching
    
    time_complexity = "Not specified"   # default value if not found
    space_complexity = "Not specified"  # default value if not found
    
    # Detect time complexity using complexity patterns from config
    for complexity, patterns in COMPLEXITY_PATTERNS.items():
        if any(pattern in text_lower for pattern in patterns):
            # if any pattern for this complexity class is found
            time_complexity = complexity  # set the detected time complexity
            break  # stop after first match (prioritizes earlier entries)
    
    # Detect space complexity patterns - similar approach but space-specific patterns
    space_patterns = {
        "O(1)": ["constant space", "in-place", "no extra space"],
        # constant space - memory usage independent of input size
        
        "O(n)": ["linear space", "array of size n", "list of n"],
        # linear space - memory grows proportionally with input
        
        "O(n²)": ["matrix", "2d array", "n by n"],
        # quadratic space - common in 2d data structures
        
        "O(log n)": ["logarithmic space", "recursive stack"]
        # logarithmic space - typical for divide and conquer recursion depth
    }
    
    for complexity, patterns in space_patterns.items():
        if any(pattern in text_lower for pattern in patterns):
            # if space complexity pattern is found
            space_complexity = complexity  # set the detected space complexity
            break  # stop after first match
    
    return time_complexity, space_complexity  # return both complexities as tuple


def extract_algorithm_steps(text: str) -> List[str]:
    """
    Extract step-by-step algorithm description.
    identifies numbered lists, bullet points, and action sentences.
    
    Args:
        text (str): Algorithm description text
    
    Returns:
        List[str]: List of algorithm steps
    """
    steps = []  # initialize empty list for algorithm steps
    
    # Look for numbered steps (e.g., "1. Initialize array" or "1) Set counter")
    numbered_pattern = r'(?:^|\n)\s*(\d+)[\.\)]\s+(.+?)(?=\n\s*\d+[\.\)]|$)'
    # pattern matches lines starting with numbers followed by . or ) and captures the text
    numbered_matches = re.findall(numbered_pattern, text, re.MULTILINE | re.DOTALL)
    # findall with multiline and dotall flags to capture across lines
    
    if numbered_matches:
        steps = [match[1].strip() for match in numbered_matches]
        # extract just the step text (second capture group) and strip whitespace
    else:
        # Look for bullet points if no numbered steps found
        bullet_pattern = r'(?:^|\n)\s*[•\-\*]\s+(.+?)(?=\n\s*[•\-\*]|$)'
        # pattern matches bullet points starting with •, -, or *
        bullet_matches = re.findall(bullet_pattern, text, re.MULTILINE | re.DOTALL)
        
        if bullet_matches:
            steps = [match.strip() for match in bullet_matches]
            # extract the text after bullet and strip whitespace
        else:
            # Look for sentences with action verbs as fallback
            action_verbs = ["first", "then", "next", "finally", "initialize", 
                          "create", "set", "compute", "calculate", "return"]
            # words that often indicate sequential steps in algorithms
            sentences = text.split('.')  # split into sentences
            
            for sent in sentences:  # check each sentence
                sent_lower = sent.lower().strip()  # lowercase for matching
                if any(verb in sent_lower for verb in action_verbs):
                    # if sentence contains an action verb
                    steps.append(sent.strip())  # add sentence as a step
    
    return steps[:10]  # limit to 10 steps for conciseness


def identify_cs_subject(text: str) -> List[Tuple[str, float]]:
    """
    Identify relevant CS subjects from text content.
    uses keyword matching against predefined cs subject categories.
    
    Args:
        text (str): The document text
    
    Returns:
        List[Tuple[str, float]]: List of (subject, confidence) tuples
    """
    text_lower = text.lower()  # convert to lowercase for case-insensitive matching
    subject_scores = {}  # dictionary to store confidence scores per subject
    
    for subject, keywords in CS_SUBJECTS.items():  # check each cs subject
        score = sum(1 for keyword in keywords if keyword in text_lower)
        # count how many keywords for this subject appear in the text
        if score > 0:  # only consider subjects with at least one keyword match
            # Normalize score by total keywords for that subject
            confidence = score / len(keywords)  # proportion of keywords found
            subject_scores[subject] = confidence  # store confidence score
    
    # Sort by confidence descending and return as list of tuples
    return sorted(subject_scores.items(), key=lambda x: x[1], reverse=True)
    # items() returns (subject, confidence) pairs, sorted by confidence


def format_code_with_syntax(code: str, language: str) -> str:
    """
    Format code with basic syntax highlighting markers.
    wraps code in markdown code fences for proper display.
    
    Args:
        code (str): The code to format
        language (str): Programming language
    
    Returns:
        str: Formatted code string
    """
    # Basic syntax highlighting using markdown code fences
    return f"```{language}\n{code}\n```"
    # markdown triple backticks with language specifier for syntax highlighting


def generate_algorithm_explanation(algorithm_info: AlgorithmInfo) -> str:
    """
    Generate comprehensive algorithm explanation.
    creates formatted markdown explanation from algorithm information.
    
    Args:
        algorithm_info (AlgorithmInfo): Algorithm information
    
    Returns:
        str: Formatted algorithm explanation
    """
    # build explanation string with markdown formatting
    explanation = f"""
### Algorithm: {algorithm_info.name}

**Type:** {algorithm_info.algorithm_type.value.replace('_', ' ').title()}

**Time Complexity:** {algorithm_info.complexity_time}
**Space Complexity:** {algorithm_info.complexity_space}

**Description:**
{algorithm_info.description}

**Step-by-Step Breakdown:**
"""
    # replace underscores with spaces and title case for algorithm type display
    
    # add each step with numbering
    for i, step in enumerate(algorithm_info.steps, 1):  # start numbering at 1
        explanation += f"\n{i}. {step}"  # add numbered step
    
    # add key insights section with complexity analysis
    explanation += "\n\n**Key Insights:**\n"
    explanation += "- This algorithm uses a " + algorithm_info.algorithm_type.value + " approach\n"
    # mention the algorithmic paradigm used
    explanation += f"- Runtime performance is {algorithm_info.complexity_time}\n"
    # restate time complexity for emphasis
    
    return explanation  # return complete formatted explanation


def extract_data_structure_info(text: str) -> Dict[str, str]:
    """
    Extract information about data structures mentioned in text.
    identifies which data structures are discussed and returns descriptions.
    
    Args:
        text (str): The document text
    
    Returns:
        Dict[str, str]: Dictionary of data structure information
    """
    ds_info = {}  # dictionary to store data structure names and descriptions
    text_lower = text.lower()  # convert to lowercase for matching
    
    # mapping of data structures to their descriptions
    data_structures = {
        "array": "Linear collection of elements, indexed access O(1)",
        # arrays - contiguous memory with constant-time random access
        
        "linked_list": "Sequential elements with pointers, insertion O(1), access O(n)",
        # linked lists - dynamic size with efficient insertion/deletion
        
        "stack": "LIFO structure, push/pop operations O(1)",
        # stack - last-in-first-out with constant time operations
        
        "queue": "FIFO structure, enqueue/dequeue operations O(1)",
        # queue - first-in-first-out for sequential processing
        
        "hash_table": "Key-value pairs, average O(1) operations",
        # hash table - key-value mapping with near-constant access
        
        "binary_tree": "Hierarchical structure, O(log n) balanced operations",
        # binary tree - hierarchical data with logarithmic balanced operations
        
        "heap": "Complete binary tree, O(log n) insert/extract",
        # heap - priority queue implementation with logarithmic operations
        
        "graph": "Nodes and edges, various traversal algorithms"
        # graph - network structure with multiple traversal strategies
    }
    
    for ds, description in data_structures.items():  # check each data structure
        if ds.replace('_', ' ') in text_lower or ds in text_lower:
            # check both with underscore replaced by space and original form
            ds_info[ds] = description  # add to results if found
    
    return ds_info  # return dictionary of detected data structures