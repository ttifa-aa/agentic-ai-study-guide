"""
track A1: Computer Science Subject Guide.
specialized track for CS topics with code detection and algorithm analysis.
"""

import re  # regular expressions for code pattern matching
from typing import Dict, List, Optional, Any, Tuple  # type hints for function signatures
from datetime import datetime  # for timestamp handling

# local imports
from core.rag_chain import ChainMode
from tracks.base_track import BaseTrack, TrackFeatures  # base track class
from config.settings import TrackType, ContentType, CS_SUBJECTS  # cs-specific configuration
from utils.cs_utils import (  # cs utility functions
    detect_code_language,
    extract_code_blocks,
    detect_algorithm_type,
    analyze_algorithm_complexity,
    extract_algorithm_steps,
    identify_cs_subject,
    format_code_with_syntax,
    generate_algorithm_explanation,
    extract_data_structure_info,
    CodeBlock,
    AlgorithmInfo,
    AlgorithmType
)
from prompts.base_prompts import CS_SPECIFIC_PROMPT  # cs-specific prompt template


class TrackA1CS(BaseTrack):
    """
    Track A1: Computer Science Subject Guide.
    provides cs-specific content processing, code detection, and algorithm explanation.
    """
    
    def __init__(self):
        """Initialize the CS track with specialized features."""
        # call parent class initializer
        super().__init__()
        
        # set track type
        self.track_type = TrackType.TRACK_A1_CS
        
        # initialize track features
        self.features = self.get_features()
        
        # cache for detected code blocks and algorithms
        self.detected_code_blocks: List[CodeBlock] = []
        self.detected_algorithms: List[AlgorithmInfo] = []
        
        # track cs subjects identified in documents
        self.identified_subjects: Dict[str, float] = {}
    
    def get_features(self) -> TrackFeatures:
        """
        Get the features and capabilities of the CS track.
        
        Returns:
            TrackFeatures: CS track features
        """
        return TrackFeatures(
            name="Computer Science Subject Guide",
            description="""
            Specialized track for Computer Science topics with:
            - CS-specific content processing with code examples and algorithms
            - Programming language detection and syntax highlighting
            - Algorithm explanation with step-by-step breakdowns
            - Database, Networking, OS specific content understanding
            - Indian university CS curriculum patterns
            """,
            supported_content_types=[
                ContentType.LECTURE_NOTES.value,
                ContentType.TEXTBOOK.value,
                ContentType.LAB_MANUAL.value,
                ContentType.PAST_PAPER.value,
                ContentType.REFERENCE.value
            ],
            special_prompts={
                "cs_explain": CS_SPECIFIC_PROMPT,
                "algorithm_analysis": "Analyze the following algorithm...",
                "code_review": "Review this code snippet..."
            },
            analytics_enabled=True,
            progress_tracking=True,
            export_formats=["markdown", "pdf", "code_snippets"]
        )
    
    def get_specialized_prompt(self, prompt_type: str) -> str:
        """
        Get a cs-specific specialized prompt template.
        
        Args:
            prompt_type (str): Type of prompt needed (cs_explain, algorithm_analysis, etc.)
        
        Returns:
            str: Specialized prompt template
        """
        # return from features or generate dynamically
        if prompt_type in self.features.special_prompts:
            return self.features.special_prompts[prompt_type]
        
        # default to cs-specific prompt
        return CS_SPECIFIC_PROMPT
    
    def process_query(
        self,
        query: str,
        query_type: str = "general",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a user query with cs-specific handling.
        
        Args:
            query (str): User's query or question
            query_type (str): Type of query (explain, code, algorithm, etc.)
            **kwargs: Additional parameters
        
        Returns:
            Dict[str, Any]: Response with cs-specific metadata
        """
        # detect query category
        detected_category = self._detect_query_category(query)
        
        # process based on query category
        if detected_category == "code":
            return self._handle_code_query(query, **kwargs)
        elif detected_category == "algorithm":
            return self._handle_algorithm_query(query, **kwargs)
        elif detected_category == "data_structure":
            return self._handle_data_structure_query(query, **kwargs)
        elif detected_category == "complexity":
            return self._handle_complexity_query(query, **kwargs)
        else:
            return self._handle_general_cs_query(query, **kwargs)
    
    def _detect_query_category(self, query: str) -> str:
        """
        Detect the category of a cs query.
        
        Args:
            query (str): User's query
        
        Returns:
            str: Query category (code, algorithm, data_structure, complexity, general)
        """
        query_lower = query.lower()
        
        # check complexity first — it's more specific than "algorithm" and shares keywords
        # ("big o", "complexity") that would otherwise match the algorithm branch first
        complexity_keywords = ["time complexity", "space complexity", "o(n)", "o(log n)", "big o"]
        if any(keyword in query_lower for keyword in complexity_keywords):
            return "complexity"
        
        # check for code-related queries
        code_keywords = ["code", "implement", "write a program", "function", "class", "method"]
        if any(keyword in query_lower for keyword in code_keywords):
            return "code"
        
        # check for algorithm-related queries — "complexity" and "big o" deliberately excluded
        # here because they are handled by the complexity branch above
        algorithm_keywords = ["algorithm", "sort", "search", "traverse"]
        if any(keyword in query_lower for keyword in algorithm_keywords):
            return "algorithm"
        
        # check for data structure queries
        ds_keywords = ["array", "linked list", "stack", "queue", "tree", "graph", "hash"]
        if any(keyword in query_lower for keyword in ds_keywords):
            return "data_structure"
        
        return "general"  # default category
    
    def _handle_code_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Handle code-related queries with syntax highlighting and explanation.
        
        Args:
            query (str): Code-related query
            **kwargs: Additional parameters
        
        Returns:
            Dict[str, Any]: Response with code explanation
        """
        # use cs-specific prompt for code questions
        prompt = self.get_specialized_prompt("cs_explain")
        
        # get answer from rag chain
        answer = self.rag_manager.invoke(
            query,
            mode=ChainMode.STUDY
        )
        
        # detect code language from query or answer
        detected_language = detect_code_language(query) or detect_code_language(answer) or "text"
        
        # extract any code blocks from the answer
        code_blocks = extract_code_blocks(answer)
        
        # format code blocks with syntax highlighting
        formatted_answer = answer
        for code_block in code_blocks:
            formatted_block = format_code_with_syntax(code_block.code, code_block.language)
            formatted_answer = formatted_answer.replace(code_block.code, formatted_block)
        
        # prepare metadata
        metadata = {
            "query_category": "code",
            "detected_language": detected_language,
            "code_blocks_found": len(code_blocks),
            "cs_subjects": identify_cs_subject(query + answer)
        }
        
        return self.format_response(formatted_answer, metadata=metadata)
    
    def _handle_algorithm_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Handle algorithm-related queries with complexity analysis.
        
        Args:
            query (str): Algorithm-related query
            **kwargs: Additional parameters
        
        Returns:
            Dict[str, Any]: Response with algorithm analysis
        """
        # get answer from rag chain
        answer = self.rag_manager.invoke(
            query,
            mode=ChainMode.STUDY
        )
        
        # detect algorithm type
        algo_type = detect_algorithm_type(query + answer)
        
        # analyze complexity
        time_complexity, space_complexity = analyze_algorithm_complexity(query + answer)
        
        # extract algorithm steps
        steps = extract_algorithm_steps(answer)
        
        # create algorithm info object
        algorithm_info = AlgorithmInfo(
            name=self._extract_algorithm_name(query),
            algorithm_type=algo_type or AlgorithmType.ITERATIVE,
            complexity_time=time_complexity,
            complexity_space=space_complexity,
            description=answer[:200] + "...",  # first 200 chars as description
            steps=steps
        )
        
        # cache the detected algorithm (cap at 50 to avoid unbounded growth)
        if len(self.detected_algorithms) < 50:
            self.detected_algorithms.append(algorithm_info)
        
        # generate enhanced explanation
        enhanced_answer = generate_algorithm_explanation(algorithm_info)
        
        # prepare metadata
        metadata = {
            "query_category": "algorithm",
            "algorithm_type": algo_type.value if algo_type else "unknown",
            "time_complexity": time_complexity,
            "space_complexity": space_complexity,
            "steps_count": len(steps)
        }
        
        return self.format_response(enhanced_answer, metadata=metadata)
    
    def _handle_data_structure_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Handle data structure related queries.
        
        Args:
            query (str): Data structure query
            **kwargs: Additional parameters
        
        Returns:
            Dict[str, Any]: Response with data structure information
        """
        # get answer from rag chain
        answer = self.rag_manager.invoke(
            query,
            mode=ChainMode.STUDY
        )
        
        # extract data structure information
        ds_info = extract_data_structure_info(query + answer)
        
        # prepare metadata
        metadata = {
            "query_category": "data_structure",
            "data_structures_mentioned": list(ds_info.keys()),
            "cs_subjects": identify_cs_subject(query)
        }
        
        return self.format_response(answer, metadata=metadata)
    
    def _handle_complexity_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Handle complexity analysis queries.
        
        Args:
            query (str): Complexity query
            **kwargs: Additional parameters
        
        Returns:
            Dict[str, Any]: Response with complexity analysis
        """
        # get answer from rag chain
        answer = self.rag_manager.invoke(
            query,
            mode=ChainMode.STUDY
        )
        
        # analyze complexity
        time_complexity, space_complexity = analyze_algorithm_complexity(query + answer)
        
        # prepare metadata
        metadata = {
            "query_category": "complexity",
            "time_complexity": time_complexity,
            "space_complexity": space_complexity
        }
        
        return self.format_response(answer, metadata=metadata)
    
    def _handle_general_cs_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Handle general cs queries.
        
        Args:
            query (str): General cs query
            **kwargs: Additional parameters
        
        Returns:
            Dict[str, Any]: Response with cs context
        """
        # get answer from rag chain
        answer = self.rag_manager.invoke(
            query,
            mode=ChainMode.STUDY
        )
        
        # identify relevant cs subjects
        subjects = identify_cs_subject(query + answer)
        
        # prepare metadata
        metadata = {
            "query_category": "general",
            "cs_subjects": subjects,
            "primary_subject": subjects[0][0] if subjects else "Unknown"
        }
        
        return self.format_response(answer, metadata=metadata)
    
    def _extract_algorithm_name(self, query: str) -> str:
        """
        Extract algorithm name from query.
        
        Args:
            query (str): User query
        
        Returns:
            str: Extracted algorithm name
        """
        # common algorithm name patterns
        patterns = [
            r'(?:explain|what is|describe)\s+(\w+\s*(?:sort|search|algorithm))',
            r'(\w+\s*(?:sort|search))\s+algorithm',
            r'algorithm\s+(\w+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                return match.group(1).title()
        
        return "Unknown Algorithm"
    
    def analyze_document_for_cs_content(self, document_text: str) -> Dict[str, Any]:
        """
        Analyze a document for cs-specific content.
        
        Args:
            document_text (str): Document text to analyze
        
        Returns:
            Dict[str, Any]: Analysis results
        """
        analysis = {
            "code_blocks": [],
            "algorithms_detected": [],
            "data_structures": {},
            "cs_subjects": [],
            "complexity_mentions": []
        }
        
        # extract code blocks
        code_blocks = extract_code_blocks(document_text)
        analysis["code_blocks"] = [
            {
                "language": block.language,
                "line_count": len(block.code.split('\n')),
                "start_line": block.start_line
            }
            for block in code_blocks
        ]
        
        # identify cs subjects
        subjects = identify_cs_subject(document_text)
        analysis["cs_subjects"] = subjects
        
        # extract data structure information
        ds_info = extract_data_structure_info(document_text)
        analysis["data_structures"] = ds_info
        
        # cache detected code blocks
        self.detected_code_blocks.extend(code_blocks)
        
        # update identified subjects
        for subject, confidence in subjects:
            if subject not in self.identified_subjects:
                self.identified_subjects[subject] = confidence
            else:
                # average confidence for multiple mentions
                self.identified_subjects[subject] = (self.identified_subjects[subject] + confidence) / 2
        
        return analysis
    
    def get_cs_subject_summary(self) -> Dict[str, Any]:
        """
        Get summary of cs subjects identified in documents.
        
        Returns:
            Dict[str, Any]: Subject summary
        """
        return {
            "identified_subjects": self.identified_subjects,
            "total_subjects": len(self.identified_subjects),
            "primary_subject": max(self.identified_subjects.items(), key=lambda x: x[1])[0] if self.identified_subjects else None,
            "code_blocks_found": len(self.detected_code_blocks),
            "algorithms_detected": len(self.detected_algorithms)
        }
    
    def generate_code_snippet(
        self,
        description: str,
        language: str = "python"
    ) -> Dict[str, Any]:
        """
        Generate code snippet from description using llm.
        
        Args:
            description (str): Description of what code should do
            language (str): Target programming language
        
        Returns:
            Dict[str, Any]: Generated code with metadata
        """
        # create code generation prompt
        prompt = f"""
        Generate {language} code for the following description.
        Include comments explaining key steps.
        Format as a complete, runnable code snippet.
        
        Description: {description}
        
        {language.upper()} Code:
        """
        
        # use llm directly for code generation
        code = self.rag_manager.llm.invoke(prompt).content
        
        # format with syntax highlighting
        formatted_code = format_code_with_syntax(code, language)
        
        return {
            "code": code,
            "formatted": formatted_code,
            "language": language,
            "description": description
        }