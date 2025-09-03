"""
Advanced Real World Use Cases Tests
Tests for conversation history, short-term memory, day planning, and all loaded plugins.
"""

import pytest
import asyncio
import json
import time
import importlib
import inspect
import os
from unittest.mock import patch

# Import modules
from modules.ai_module import AIModule
from config.config_manager import DatabaseManager


def parse_ai_response(result):
    """Helper function to parse AI response consistently."""
    if isinstance(result, dict):
        if "text" in result:
            return result["text"]
        elif "response" in result:
            # Handle nested response structure
            response = result["response"]
            if isinstance(response, str):
                try:
                    # Try to parse JSON string
                    parsed = json.loads(response)
                    if isinstance(parsed, dict) and "text" in parsed:
                        return parsed["text"]
                    return response
                except json.JSONDecodeError:
                    return response
            elif isinstance(response, dict) and "text" in response:
                return response["text"]
            return str(response)
        return str(result)
    return str(result)


class TestConversationHistoryRealWorld:
    """Test conversation history and context persistence."""

    @pytest.fixture
    def ai_setup(self):
        """Setup for AI testing."""
        db_manager = DatabaseManager("test_ai.db")
        
        # Create proper config for AIModule
        config = {
            "ai": {
                "provider": "openai",
                "model": "gpt-5-nano"
            }
        }
        ai_module = AIModule(config=config)
        
        try:
            yield ai_module, db_manager
        finally:
            # Cleanup
            try:
                if os.path.exists("test_ai.db"):
                    os.remove("test_ai.db")
            except:
                pass

    @pytest.mark.asyncio  
    @pytest.mark.integration
    async def test_weather_question_with_context_memory(self, ai_setup):
        """
        Test: "czy będzie padać" → "gdzie mam sprawdzić pogodę" → "Warszawa" 
        Expected: FC→weather check, memory that user asked about rain.
        """
        ai_module, db_manager = ai_setup
        user_id = "test_user_weather"
        
        # First query - weather question
        with patch('modules.ai_module.generate_response') as mock_generate:
            mock_generate.return_value = {
                "type": "clarification_request",
                "text": "Gdzie mam sprawdzić pogodę? Podaj miasto.",
                "requires_location": True
            }
            
            start_time = time.time()
            result1 = await ai_module.process_query("czy będzie padać", {"user_id": user_id})
            end_time = time.time()
            
            response_text1 = parse_ai_response(result1)
            assert "gdzie" in response_text1.lower() or "miasto" in response_text1.lower()
            
            # Verify latency < 2.0s per call 
            latency = end_time - start_time
            assert latency < 2.0, f"Query latency {latency:.2f}s exceeds 2.0s limit"
            
        # Save first interaction
        await db_manager.save_interaction(user_id, "czy będzie padać", json.dumps({"text": response_text1}))
        
        # Second query - provide location
        with patch('modules.ai_module.generate_response') as mock_generate:
            mock_generate.return_value = {
                "type": "normal_response", 
                "text": "Sprawdzam pogodę dla Warszawy... Jutro będzie pochmurnie z możliwymi opadami deszczu po południu.",
                "weather_data": {
                    "location": "Warszawa",
                    "forecast": "rain_possible",
                    "temperature": "15°C"
                },
                "functions_called": ["weather_check_forecast"]
            }
            
            # Get conversation history
            history = await db_manager.get_user_history(user_id, limit=10)
            context = {"user_id": user_id, "history": history}
            
            start_time = time.time()
            result2 = await ai_module.process_query("Warszawa", context)
            end_time = time.time()
            
            response_text2 = parse_ai_response(result2)
            
            # Verify weather context is maintained
            assert "warszaw" in response_text2.lower()
            assert any(word in response_text2.lower() for word in ["pada", "deszcz", "opady", "rain"])
            
            # Verify latency
            latency = end_time - start_time
            assert latency < 2.0, f"Context query latency {latency:.2f}s exceeds 2.0s limit"

    @pytest.mark.asyncio
    @pytest.mark.integration 
    async def test_conversation_context_persistence(self, ai_setup):
        """Test that conversation context persists across multiple interactions."""
        ai_module, db_manager = ai_setup
        user_id = "test_user_persistence"
        
        # Simulate multi-turn conversation
        queries = [
            "Jutro mam ważne spotkanie",
            "O której godzinie?", 
            "O 15:00",
            "Przypomnij mi o tym 30 minut wcześniej"
        ]
        
        responses = [
            {"type": "normal_response", "text": "Rozumiem, jutro masz ważne spotkanie. O której godzinie?"},
            {"type": "clarification_request", "text": "O której godzinie jest spotkanie?"},
            {"type": "normal_response", "text": "Zanotowałem - spotkanie jutro o 15:00"}, 
            {"type": "normal_response", "text": "Ustawię przypomnienie na 14:30 o jutrzejszym spotkaniu o 15:00"}
        ]
        
        # Process each query with growing history
        for i, (query, expected_response) in enumerate(zip(queries, responses)):
            with patch('modules.ai_module.generate_response') as mock_generate:
                mock_generate.return_value = expected_response
                
                # Get current history
                history = await db_manager.get_user_history(user_id, limit=20)
                context = {"user_id": user_id, "history": history}
                
                # Process query
                result = await ai_module.process_query(query, context)
                
                # Parse response text
                response_text = parse_ai_response(result)
                assert len(response_text) > 0, f"Empty response in iteration {i}"
                
                # Save interaction
                response_json = json.dumps(result) if isinstance(result, dict) else str(result)
                await db_manager.save_interaction(user_id, query, response_json)
                
                # Check that history grows
                new_history = await db_manager.get_user_history(user_id, limit=20)
                assert len(new_history) == (i + 1) * 2  # Each turn adds user + assistant message
        
        # Final verification - context should contain full conversation
        final_history = await db_manager.get_user_history(user_id, limit=20)
        assert len(final_history) == 8  # 4 user + 4 assistant messages
        
        # Check conversation coherence - verify key elements exist in history
        history_str = str(final_history)
        assert "spotkanie" in history_str
        assert "15:00" in history_str 
        assert "przypomn" in history_str.lower()


class TestShortTermMemoryAnchors:
    """Test short-term memory anchors and context retention."""
    
    @pytest.fixture
    def memory_setup(self):
        """Setup for memory testing."""
        db_manager = DatabaseManager("test_memory.db")
        
        # Create proper config for AIModule
        config = {
            "ai": {
                "provider": "openai",
                "model": "gpt-5-nano"
            }
        }
        ai_module = AIModule(config=config)
        
        try:
            yield ai_module, db_manager
        finally:
            # Cleanup
            try:
                if os.path.exists("test_memory.db"):
                    os.remove("test_memory.db")
            except:
                pass

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_exam_reminder_memory_anchor(self, memory_setup):
        """
        Test: "Jutro mam kolokwium z algebry. Przypomnij mi o 18."
        → po 2 turach: "Ej, o której ta przypominajka?"
        Expected: "Jutro, 18:00, kolokwium z algebry."
        """
        ai_module, db_manager = memory_setup
        user_id = "test_user_exam"
        
        # First interaction - set reminder
        with patch('modules.ai_module.generate_response') as mock_generate:
            mock_generate.return_value = {
                "type": "normal_response",
                "text": "Zanotowałem - przypomnę Ci jutro o 18:00 o kolokwium z algebry.",
                "reminder_set": {
                    "time": "18:00",
                    "date": "tomorrow", 
                    "event": "kolokwium z algebry"
                }
            }
            
            context1 = {"user_id": user_id, "history": []}
            result1 = await ai_module.process_query("Jutro mam kolokwium z algebry. Przypomnij mi o 18.", context1)
            
            response_text1 = parse_ai_response(result1)
            assert "18" in response_text1
            assert "kolokwium" in response_text1.lower() or "algebra" in response_text1.lower()
            
            # Save interaction
            await db_manager.save_interaction(
                user_id, 
                "Jutro mam kolokwium z algebry. Przypomnij mi o 18.",
                json.dumps({"text": response_text1})
            )
        
        # Intermediate interactions (simulate time passing)
        intermediate_queries = ["Jak się masz?", "Co robimy dziś?"]
        for query in intermediate_queries:
            with patch('modules.ai_module.generate_response') as mock_generate:
                mock_generate.return_value = {
                    "type": "normal_response",
                    "text": f"Odpowiedź na: {query}"
                }
                
                history = await db_manager.get_user_history(user_id, limit=10)
                context = {"user_id": user_id, "history": history}
                result = await ai_module.process_query(query, context)
                
                response_text = parse_ai_response(result)
                await db_manager.save_interaction(user_id, query, json.dumps({"text": response_text}))
        
        # Memory recall test 
        with patch('modules.ai_module.generate_response') as mock_generate:
            mock_generate.return_value = {
                "type": "normal_response",
                "text": "Przypomnienie było na jutro o 18:00 - kolokwium z algebry.",
                "recalled_reminder": {
                    "time": "18:00",
                    "event": "kolokwium z algebry",
                    "date": "jutro"
                }
            }
            
            history = await db_manager.get_user_history(user_id, limit=10)
            context3 = {"user_id": user_id, "history": history}
            result3 = await ai_module.process_query("Ej, o której ta przypominajka?", context3)
            
            response_text3 = parse_ai_response(result3)
            
            # Memory anchor precision metrics
            hit_at_1 = (
                "18" in response_text3 and 
                ("kolokwium" in response_text3.lower() or "algebra" in response_text3.lower())
            )
            assert hit_at_1, "Memory anchor failed hit@1 precision test"
            
            # Hallucination check - shouldn't mention wrong time or subject
            hallucination_markers = ["19:00", "20:00", "matematyka", "fizyka", "chemia"]
            hallucination_rate = sum(1 for marker in hallucination_markers if marker in response_text3.lower())
            assert hallucination_rate == 0, f"Hallucination detected: {hallucination_rate} false elements"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_reminder_time_update_edge_case(self, memory_setup):
        """Test updating reminder time preserves event context."""
        ai_module, db_manager = memory_setup
        user_id = "test_user_update"
        
        # Initial reminder
        with patch('modules.ai_module.generate_response') as mock_generate:
            mock_generate.return_value = {
                "type": "normal_response",
                "text": "Ustawię przypomnienie na 18:00 o kolokwium z algebry."
            }
            
            context1 = {"user_id": user_id, "history": []}
            result1 = await ai_module.process_query("Jutro mam kolokwium z algebry. Przypomnij mi o 18.", context1)
            
            response_text1 = parse_ai_response(result1)
            await db_manager.save_interaction(
                user_id, 
                "Jutro mam kolokwium z algebry. Przypomnij mi o 18.",
                json.dumps({"text": response_text1})
            )
        
        # Update time
        with patch('modules.ai_module.generate_response') as mock_generate:
            mock_generate.return_value = {
                "type": "normal_response", 
                "text": "Przesunąłem przypomnienie z 18:00 na 19:00 - nadal o kolokwium z algebry."
            }
            
            history = await db_manager.get_user_history(user_id, limit=10)
            context2 = {"user_id": user_id, "history": history}
            result2 = await ai_module.process_query("W sumie przesuń na 19.", context2)
            
            response_text = parse_ai_response(result2)
            
            # Verify correct update
            assert "19" in response_text
            assert "kolokwium" in response_text.lower() or "algebra" in response_text.lower()
            assert "przesun" in response_text.lower() or "zmien" in response_text.lower()
            
            # Verify old time is not mentioned or is explicitly changed
            if "18" in response_text:
                assert "z 18" in response_text or "was 18" in response_text  # Should indicate change


class TestDayPlanningWithCalendar:
    """Test day planning with calendar integration."""
    
    @pytest.fixture
    def planning_setup(self):
        """Setup for planning testing."""
        db_manager = DatabaseManager("test_planning.db")
        
        # Create proper config for AIModule
        config = {
            "ai": {
                "provider": "openai",
                "model": "gpt-5-nano"
            }
        }
        ai_module = AIModule(config=config)
        
        try:
            yield ai_module, db_manager
        finally:
            # Cleanup
            try:
                if os.path.exists("test_planning.db"):
                    os.remove("test_planning.db")
            except:
                pass

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_study_break_planning_with_calendar(self, planning_setup):
        """
        Test: "Jutro mam 3h okienka między 13 a 16 — zaplanuj naukę z przerwami."
        Expected: FC→calendar read, generated study blocks (pomodoro), "przenieś na 14:30" capability.
        """
        ai_module, db_manager = planning_setup
        user_id = "test_user_planning"
        
        # Test planning request
        with patch('modules.ai_module.generate_response') as mock_generate:
            mock_generate.return_value = {
                "type": "normal_response",
                "text": "Zaplanowałem naukę w okienku 13:00-16:00 z przerwami pomodoro:\n13:00-13:25 Nauka\n13:25-13:30 Przerwa\n13:30-13:55 Nauka\n13:55-14:10 Długa przerwa\n14:10-14:35 Nauka\n14:35-14:40 Przerwa\n14:40-15:05 Nauka\n15:05-15:20 Długa przerwa\n15:20-15:45 Nauka",
                "study_plan": {
                    "total_duration": "3h",
                    "study_blocks": [
                        {"start": "13:00", "end": "13:25", "type": "study"},
                        {"start": "13:25", "end": "13:30", "type": "break"},
                        {"start": "13:30", "end": "13:55", "type": "study"},
                        {"start": "13:55", "end": "14:10", "type": "long_break"},
                        {"start": "14:10", "end": "14:35", "type": "study"},
                        {"start": "14:35", "end": "14:40", "type": "break"},
                        {"start": "14:40", "end": "15:05", "type": "study"},
                        {"start": "15:05", "end": "15:20", "type": "long_break"},
                        {"start": "15:20", "end": "15:45", "type": "study"}
                    ],
                    "timezone": "Europe/Warsaw"
                },
                "functions_called": ["calendar_read_schedule"]
            }
            
            start_time = time.time()
            context = {"user_id": user_id, "history": []}
            result = await ai_module.process_query("Jutro mam 3h okienka między 13 a 16 — zaplanuj naukę z przerwami.", context)
            end_time = time.time()
            
            # Verify response
            response_text = parse_ai_response(result)
            
            # Check study plan generation (if available in result)
            if "study_plan" in result:
                study_plan = result["study_plan"]
                assert len(study_plan["study_blocks"]) > 0
                assert study_plan["timezone"] == "Europe/Warsaw"
                
                # Check pomodoro pattern (study blocks ~25min, breaks ~5min)
                study_blocks = [block for block in study_plan["study_blocks"] if block["type"] == "study"]
                assert len(study_blocks) >= 3  # At least 3 study sessions in 3 hours
            
            # Verify latency < 2.0s per call 
            latency = end_time - start_time
            assert latency < 2.0, f"Planning latency {latency:.2f}s exceeds 2.0s limit"
        
        # Save planning interaction
        await db_manager.save_interaction(user_id, "Jutro mam 3h okienka między 13 a 16 — zaplanuj naukę z przerwami.", json.dumps({"text": response_text}))
        
        # Verify planning contains expected elements
        assert "13" in response_text and "16" in response_text
        assert any(word in response_text.lower() for word in ["przerw", "break", "pauza", "odpocz"])
        assert any(word in response_text.lower() for word in ["plan", "schemat", "harmonogram"])
        
        # Test rescheduling capability
        with patch('modules.ai_module.generate_response') as mock_generate:
            mock_generate.return_value = {
                "type": "normal_response",
                "text": "Przesunąłem plan nauki na 14:30-17:30. Nowy harmonogram pomodoro rozpocznie się o 14:30.",
                "updated_plan": {
                    "start_time": "14:30",
                    "end_time": "17:30", 
                    "timezone": "Europe/Warsaw"
                }
            }
            
            history = await db_manager.get_user_history(user_id, limit=10)
            context2 = {"user_id": user_id, "history": history}
            result2 = await ai_module.process_query("przenieś na 14:30", context2)
            
            response_text2 = parse_ai_response(result2)
            assert "14:30" in response_text2
            assert "plan" in response_text2.lower()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_calendar_conflict_detection(self, planning_setup):
        """Test conflict detection in calendar planning."""
        ai_module, db_manager = planning_setup
        user_id = "test_user_conflicts"
        
        # Mock calendar with conflicts
        with patch('modules.ai_module.generate_response') as mock_generate:
            mock_generate.return_value = {
                "type": "normal_response",
                "text": "Znalazłem konflikt o 14:00-15:00 (Meeting). Zaplanowałem naukę w dostępnych slotach: 13:00-14:00 i 15:00-16:00.",
                "conflicts_detected": 1,
                "available_time": "2h",
                "study_plan": {
                    "blocks": [
                        {"start": "13:00", "end": "14:00", "type": "study"},
                        {"start": "15:00", "end": "16:00", "type": "study"}
                    ]
                }
            }
            
            context = {"user_id": user_id, "history": []}
            result = await ai_module.process_query("Zaplanuj naukę jutro 13-16", context)
            
            response_text = parse_ai_response(result)
            assert "konflikt" in response_text.lower() or "meeting" in response_text
            
            if "conflicts_detected" in result:
                assert result["conflicts_detected"] == 1
            if "study_plan" in result:
                assert len(result["study_plan"]["blocks"]) == 2


class TestLoadedPluginsFunctionality:
    """Test all loaded plugins and their functionality."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_weather_module_availability(self):
        """Test weather module is loaded and functional."""
        try:
            module = importlib.import_module("modules.weather_module_refactored")
            
            # Check for expected functions
            expected_functions = ["execute_function", "get_functions", "create_standard_function_schema"]
            available_functions = [func for func in expected_functions if hasattr(module, func)]
            
            assert len(available_functions) > 0, "No weather functions found"
            
            # Test function signature if available
            if available_functions:
                func = getattr(module, available_functions[0])
                assert callable(func), f"Weather function {available_functions[0]} not callable"
                
        except ImportError:
            pytest.skip("Weather module not available")

    @pytest.mark.asyncio 
    @pytest.mark.integration
    async def test_search_module_availability(self):
        """Test search module is loaded and functional."""
        try:
            module = importlib.import_module("modules.search_module")
            
            # Check for expected functions
            expected_functions = ["execute_function", "get_functions", "get_search_module"]
            available_functions = [func for func in expected_functions if hasattr(module, func)]
            
            assert len(available_functions) > 0, "No search functions found"
            
            # Test function signature if available
            if available_functions:
                func = getattr(module, available_functions[0])
                assert callable(func), f"Search function {available_functions[0]} not callable"
                
        except ImportError:
            pytest.skip("Search module not available")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_vector_memory_module_availability(self):
        """Test vector memory module is loaded and functional."""
        try:
            module = importlib.import_module("modules.vector_memory_module")
            
            # Check for expected functions
            expected_functions = ["get_functions", "cosine_similarity", "get_database_manager"]
            available_functions = [func for func in expected_functions if hasattr(module, func)]
            
            assert len(available_functions) > 0, "No vector memory functions found"
            
            # Test function signature if available
            if available_functions:
                func = getattr(module, available_functions[0])
                assert callable(func), f"Vector memory function {available_functions[0]} not callable"
                
        except ImportError:
            pytest.skip("Vector memory module not available")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_core_module_availability(self):
        """Test core module is loaded and functional."""
        try:
            module = importlib.import_module("modules.core_module")
            
            # Check for expected functions
            expected_functions = ["execute_function", "get_functions", "set_reminder", "add_task", "view_calendar"]
            available_functions = [func for func in expected_functions if hasattr(module, func)]
            
            assert len(available_functions) > 0, "No core functions found"
            
            # Test function signature if available  
            if available_functions:
                func = getattr(module, available_functions[0])
                assert callable(func), f"Core function {available_functions[0]} not callable"
                
        except ImportError:
            pytest.skip("Core module not available")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_basic_module_discovery(self):
        """Test discovery of all available modules."""
        module_paths = [
            "modules.weather_module_refactored", 
            "modules.search_module",
            "modules.vector_memory_module",
            "modules.core_module",
            "modules.shopping_list_module",
            "modules.notes_module", 
            "modules.tasks_module",
            "modules.music_module",
            "modules.web_module"
        ]
        
        loaded_modules = []
        for module_name in module_paths:
            try:
                module = importlib.import_module(module_name)
                loaded_modules.append(module_name)
            except ImportError:
                continue
        
        # Should have at least some modules loaded
        assert len(loaded_modules) > 0, "No modules could be loaded"
        print(f"Successfully loaded {len(loaded_modules)} modules: {loaded_modules}")

    @pytest.mark.asyncio
    @pytest.mark.integration  
    async def test_module_function_signatures(self):
        """Test that loaded modules have proper function signatures."""
        module_name = "modules.core_module"  # Test one known module
        try:
            module = importlib.import_module(module_name)
            
            # Get all functions in the module
            functions = [name for name, obj in inspect.getmembers(module) if inspect.isfunction(obj)]
            
            # Should have some functions
            assert len(functions) > 0, f"Module {module_name} has no functions"
            
            # Test that functions are properly defined
            for func_name in functions[:3]:  # Test first 3 functions
                func = getattr(module, func_name)
                signature = inspect.signature(func)
                
                # Should be callable with some parameters
                assert callable(func), f"Function {func_name} not callable"
                # Should have signature information
                assert signature is not None, f"Function {func_name} has no signature"
                
        except ImportError:
            pytest.skip(f"Module {module_name} not available")


class TestRealWorldIntegrationMetrics:
    """Test end-to-end integration with performance metrics."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_end_to_end_conversation_flow_performance(self):
        """Test complete conversation flow with performance requirements."""
        
        # Setup
        db_manager = DatabaseManager("test_e2e.db")
        config = {
            "ai": {
                "provider": "openai",
                "model": "gpt-5-nano"
            }
        }
        ai_module = AIModule(config=config)
        
        try:
            user_id = "test_user_e2e"
            
            # Test conversation scenarios with performance requirements
            test_cases = [
                {
                    "module": "modules.weather_module_refactored",
                    "query": "Jaka będzie pogoda jutro?",
                    "expected_keywords": ["pogoda", "jutro"],
                    "max_latency": 2.0
                },
                {
                    "module": "modules.core_module", 
                    "query": "Zapisz notatkę: spotkanie 15:00",
                    "expected_keywords": ["zapisz", "notatka"],
                    "max_latency": 1.5
                }
            ]
            
            for test_case in test_cases:
                try:
                    # Test module availability
                    module = importlib.import_module(test_case["module"])
                    
                    # Performance test
                    with patch('modules.ai_module.generate_response') as mock_generate:
                        mock_generate.return_value = {
                            "type": "normal_response",
                            "text": f"Przetworzone: {test_case['query']}"
                        }
                        
                        start_time = time.time()
                        result = await ai_module.process_query(test_case["query"], {"user_id": user_id})
                        end_time = time.time()
                        
                        # Performance assertions
                        latency = end_time - start_time
                        assert latency < test_case["max_latency"], f"Latency {latency:.2f}s exceeds {test_case['max_latency']}s"
                        
                        # Functionality assertions
                        response_text = parse_ai_response(result)
                        assert len(response_text) > 0, "Empty response"
                        
                except ImportError:
                    print(f"Module {test_case['module']} not available, skipping test")
                    continue
                    
        finally:
            # Cleanup
            try:
                if os.path.exists("test_e2e.db"):
                    os.remove("test_e2e.db")
            except:
                pass