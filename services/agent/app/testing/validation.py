"""
Validation system for test results.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable

from agent.app.connector_llm import ConnectorLLM
from agent.app.idea_test_utils import call_validation_function

_logger = logging.getLogger(__name__)


class ValidationCheck(ABC):
    """
    Base class for validation checks.
    """
    
    @abstractmethod
    async def validate(self, result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run validation check.
        :param result: Test result.
        :param observability: Observability data.
        :return: Validation result dict.
        """
        pass
    
    def get_check_name(self) -> str:
        """
        Get check identifier.
        :return: Check name.
        """
        return self.__class__.__name__


class FunctionValidationCheck(ValidationCheck):
    """
    Wrapper for function-based validation checks.
    """
    
    def __init__(self, func: Callable, check_name: Optional[str] = None):
        """
        Initialize with validation function.
        :param func: Validation function.
        :param check_name: Optional check name override.
        """
        self.func = func
        self._check_name = check_name or func.__name__
    
    async def validate(self, result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run function-based validation.
        """
        try:
            return await call_validation_function(self.func, result, observability)
        except Exception as exc:
            _logger.error(f"Validation function {self._check_name} failed: {exc}")
            return {
                "check": self._check_name,
                "passed": False,
                "score": 0.0,
                "error": str(exc),
            }
    
    def get_check_name(self) -> str:
        return self._check_name


class LLMValidationCheck(ValidationCheck):
    """
    LLM-based validation check.
    """
    
    def __init__(self, func: Callable, check_name: str = "llm_validation"):
        """
        Initialize LLM validation.
        :param func: LLM validation function.
        :param check_name: Check name.
        """
        self.func = func
        self._check_name = check_name
    
    async def validate(
        self,
        result: Dict[str, Any],
        observability: Dict[str, Any],
        connector_llm: ConnectorLLM,
        validation_model: str,
    ) -> Dict[str, Any]:
        """
        Run LLM-based validation.
        :param result: Test result.
        :param observability: Observability data.
        :param connector_llm: LLM connector.
        :param validation_model: Model name for validation.
        :return: Validation result.
        """
        try:
            original_model = connector_llm.get_model()
            connector_llm.set_model(validation_model)
            try:
                return await call_validation_function(
                    self.func,
                    result,
                    observability,
                    connector_llm,
                    validation_model,
                )
            finally:
                connector_llm.set_model(original_model)
        except Exception as exc:
            _logger.error(f"LLM validation failed: {exc}")
            return {
                "check": self._check_name,
                "passed": False,
                "score": 0.0,
                "error": str(exc),
            }
    
    def get_check_name(self) -> str:
        return self._check_name


class ValidationRunner:
    """
    Runs validation checks and aggregates results.
    """
    
    def __init__(self, validation_model: str = "gpt-5-mini"):
        """
        Initialize validation runner.
        :param validation_model: Model to use for LLM validation.
        """
        self.validation_model = validation_model
        self.checks: List[ValidationCheck] = []
        self.llm_checks: List[LLMValidationCheck] = []
    
    def add_check(self, check: ValidationCheck):
        """
        Add validation check.
        :param check: Validation check instance.
        """
        if isinstance(check, LLMValidationCheck):
            self.llm_checks.append(check)
        else:
            self.checks.append(check)
    
    def add_function_check(self, func: Callable, check_name: Optional[str] = None):
        """
        Add function-based validation check.
        :param func: Validation function.
        :param check_name: Optional check name.
        """
        self.add_check(FunctionValidationCheck(func, check_name))
    
    def add_llm_check(self, func: Callable, check_name: str = "llm_validation"):
        """
        Add LLM-based validation check.
        :param func: LLM validation function.
        :param check_name: Check name.
        """
        self.add_check(LLMValidationCheck(func, check_name))
    
    async def run(
        self,
        result: Dict[str, Any],
        observability: Dict[str, Any],
        connector_llm: Optional[ConnectorLLM] = None,
    ) -> Dict[str, Any]:
        """
        Run all validation checks.
        :param result: Test result.
        :param observability: Observability data.
        :param connector_llm: LLM connector (required for LLM checks).
        :return: Complete validation results.
        """
        validation_results = {
            "grep_validations": [],
            "llm_validation": None,
            "overall_passed": False,
            "overall_score": 0.0,
        }
        
        scores = []
        
        for check in self.checks:
            check_result = await check.validate(result, observability)
            validation_results["grep_validations"].append(check_result)
            if "score" in check_result:
                scores.append(check_result["score"])
        
        if self.llm_checks and connector_llm:
            for llm_check in self.llm_checks:
                llm_result = await llm_check.validate(result, observability, connector_llm, self.validation_model)
                validation_results["llm_validation"] = llm_result
                if "score" in llm_result:
                    scores.append(llm_result["score"])
        
        if scores:
            validation_results["overall_score"] = sum(scores) / len(scores)
            validation_results["overall_passed"] = validation_results["overall_score"] >= 0.75
        
        passed_count = sum(1 for v in validation_results["grep_validations"] if v.get("passed", False))
        if validation_results["llm_validation"] and validation_results["llm_validation"].get("passed", False):
            passed_count += 1
        
        total_checks = len(validation_results["grep_validations"])
        if validation_results["llm_validation"]:
            total_checks += 1
        
        validation_results["checks_passed"] = passed_count
        validation_results["total_checks"] = total_checks
        validation_results["pass_rate"] = passed_count / total_checks if total_checks > 0 else 0.0
        
        return validation_results
