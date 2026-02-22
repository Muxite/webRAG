"""
Test module loader and wrapper.
"""

import importlib.util
import logging
from pathlib import Path
from typing import Dict, Any, List

from agent.app.testing.validation import ValidationRunner, FunctionValidationCheck, LLMValidationCheck

_logger = logging.getLogger(__name__)


class IdeaTestModule:
    """
    Wrapper for test module with validation functions.
    """
    
    def __init__(self, module_path: Path):
        """
        Load test module from file.
        :param module_path: Path to test Python file.
        """
        self.path = module_path
        self.module = None
        self.metadata = {}
        self.validation_runner = ValidationRunner()
        self._load_module()
    
    def _load_module(self):
        """Load test module dynamically."""
        spec = importlib.util.spec_from_file_location("test_module", self.path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load test module: {self.path}")
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.metadata = self.module.get_test_metadata()
        
        validation_functions = self.module.get_validation_functions()
        for func in validation_functions:
            self.validation_runner.add_function_check(func)
        
        llm_func = self.module.get_llm_validation_function()
        if llm_func:
            self.validation_runner.add_llm_check(llm_func)
    
    def get_task_statement(self) -> str:
        """Get task statement from module."""
        return self.module.get_task_statement()
    
    def get_required_deliverables(self) -> List[str]:
        """Get required deliverables from module."""
        return self.module.get_required_deliverables()
    
    def get_success_criteria(self) -> List[str]:
        """Get success criteria from module."""
        return self.module.get_success_criteria()
    
    def get_validation_runner(self) -> ValidationRunner:
        """Get validation runner instance."""
        return self.validation_runner
