"""
Base module class for all GAJA Assistant plugins.
Provides common functionality and standardized interface.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class FunctionSchema:
    """Schema for plugin function definition."""
    name: str
    description: str
    parameters: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format expected by AI system."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }


class BaseModule(ABC):
    """Base class for all GAJA Assistant modules."""
    
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__.replace('Module', '').lower()
        self.logger = logging.getLogger(f"modules.{self.name}")
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the module. Called once during server startup."""
        if self._initialized:
            return
        
        try:
            await self._initialize_impl()
            self._initialized = True
            self.logger.info(f"Module {self.name} initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize module {self.name}: {e}")
            raise
    
    async def _initialize_impl(self) -> None:
        """Override this method to implement module-specific initialization."""
        pass
    
    @abstractmethod
    def get_function_schemas(self) -> List[FunctionSchema]:
        """Return list of function schemas provided by this module."""
        pass
    
    @abstractmethod
    async def execute_function(self, function_name: str, parameters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Execute a function provided by this module."""
        pass
    
    def get_functions(self) -> List[Dict[str, Any]]:
        """Get function definitions in format expected by AI system."""
        return [schema.to_dict() for schema in self.get_function_schemas()]
    
    def supports_test_mode(self) -> bool:
        """Override to indicate if module supports test mode."""
        return False
    
    def get_mock_data(self, function_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Override to provide mock data for test mode."""
        return {
            "success": True,
            "data": {"message": f"Mock data for {function_name}"},
            "test_mode": True
        }
    
    async def handle_test_mode(self, function_name: str, parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle test mode execution if supported."""
        if not self.supports_test_mode():
            return None
        
        test_mode = parameters.get("test_mode", False)
        if test_mode:
            self.logger.info(f"Executing {function_name} in test mode")
            return self.get_mock_data(function_name, parameters)
        
        return None
    
    def _create_success_response(self, data: Any, message: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Create standardized success response."""
        response = {
            "success": True,
            "data": data
        }
        if message:
            response["message"] = message
        response.update(kwargs)
        return response
    
    def _create_error_response(self, error: str, code: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Create standardized error response."""
        response = {
            "success": False,
            "error": error
        }
        if code:
            response["error_code"] = code
        response.update(kwargs)
        return response
    
    async def safe_execute(self, function_name: str, parameters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Safely execute function with error handling and test mode support."""
        try:
            # Check for test mode first
            test_result = await self.handle_test_mode(function_name, parameters)
            if test_result is not None:
                return test_result
            
            # Execute actual function
            return await self.execute_function(function_name, parameters, user_id)
            
        except Exception as e:
            self.logger.error(f"Error executing {function_name}: {e}")
            return self._create_error_response(
                f"Failed to execute {function_name}: {str(e)}",
                code="execution_error"
            )


class TestModeSupport:
    """Mixin class for modules that support test mode."""
    
    def supports_test_mode(self) -> bool:
        return True
    
    def add_test_mode_parameter(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Add test_mode parameter to function parameters."""
        if "properties" not in parameters:
            parameters["properties"] = {}
        
        parameters["properties"]["test_mode"] = {
            "type": "boolean",
            "description": "Tryb testowy (używa mock danych)",
            "default": False
        }
        return parameters


def create_standard_function_schema(
    name: str,
    description: str,
    properties: Dict[str, Any],
    required: Optional[List[str]] = None,
    include_test_mode: bool = True
) -> FunctionSchema:
    """Create a standardized function schema."""
    parameters = {
        "type": "object",
        "properties": properties,
        "required": required or []
    }
    
    if include_test_mode:
        test_support = TestModeSupport()
        parameters = test_support.add_test_mode_parameter(parameters)
    
    return FunctionSchema(name, description, parameters)
