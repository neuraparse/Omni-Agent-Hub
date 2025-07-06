"""
ReAct (Reasoning and Acting) Orchestrator Agent.

This is the main orchestrator that implements the ReAct pattern:
1. Thought: Analyze the user request and plan approach
2. Action: Choose and execute appropriate tools/agents
3. Observation: Analyze the results
4. Reflection: Self-critique and improve if needed

The orchestrator coordinates all other agents and tools.
"""

import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from .base_agent import BaseAgent, AgentCapability, AgentContext, AgentResult
from ..services.llm_service import LLMService, LLMMessage, LLMProvider
from ..prompts.prompt_templates import PromptTemplateManager, PromptType, PromptComplexity
from ..learning.adaptive_learning import AdaptiveLearningEngine
from ..core.exceptions import AgentError


class ReActStep:
    """Represents a single step in ReAct reasoning."""
    
    def __init__(self, step_type: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        self.step_type = step_type  # thought, action, observation, reflection
        self.content = content
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_type": self.step_type,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class ReActOrchestrator(BaseAgent):
    """Advanced ReAct orchestrator with adaptive learning and sophisticated prompting."""

    def __init__(
        self,
        llm_service: LLMService,
        prompt_manager: PromptTemplateManager = None,
        learning_engine: AdaptiveLearningEngine = None,
        **kwargs
    ):
        capabilities = [
            AgentCapability(
                name="orchestrate",
                description="Coordinate multiple agents and tools to solve complex tasks",
                parameters={"max_iterations": 5, "reflection_enabled": True, "adaptive_learning": True}
            ),
            AgentCapability(
                name="reason",
                description="Analyze user requests and plan multi-step solutions with advanced prompting",
                parameters={"reasoning_depth": "expert", "prompt_optimization": True}
            ),
            AgentCapability(
                name="delegate",
                description="Delegate tasks to specialized agents with performance tracking",
                parameters={"available_agents": ["codeact", "search", "analysis"], "success_prediction": True}
            ),
            AgentCapability(
                name="learn",
                description="Learn from interactions and optimize performance over time",
                parameters={"pattern_recognition": True, "adaptive_prompting": True}
            )
        ]

        super().__init__(
            name="ReActOrchestrator",
            description="Advanced ReAct orchestrator with adaptive learning and sophisticated prompting",
            capabilities=capabilities,
            **kwargs
        )

        self.llm_service = llm_service
        self.prompt_manager = prompt_manager or PromptTemplateManager()
        self.learning_engine = learning_engine

        # Advanced configuration
        self.max_iterations = 5
        self.confidence_threshold = 0.8
        self.learning_enabled = True
        self.adaptive_prompting = True

        # Performance tracking
        self.interaction_count = 0
        self.success_history = []

        # Available tools with enhanced capabilities
        self.available_tools = {
            "web_search": self._web_search_tool,
            "code_execution": self._code_execution_tool,
            "knowledge_search": self._knowledge_search_tool,
            "file_analysis": self._file_analysis_tool,
            "data_analysis": self._data_analysis_tool,
            "api_integration": self._api_integration_tool
        }
    
    async def execute(self, task: str, context: AgentContext) -> AgentResult:
        """Execute task using advanced ReAct pattern with learning and optimization."""

        self.interaction_count += 1
        start_time = datetime.utcnow()

        # Determine task complexity and select optimal prompt
        task_complexity = self._assess_task_complexity(task, context)
        optimal_template = self.prompt_manager.get_best_template(
            PromptType.REASONING,
            task_complexity,
            context.memory
        )

        # Predict success probability if learning is enabled
        success_prediction = None
        if self.learning_engine:
            try:
                success_prediction, prediction_details = await self.learning_engine.predict_success(
                    task, context.memory, ""  # Empty response for initial prediction
                )
                self.logger.info(
                    "Success prediction",
                    probability=success_prediction,
                    details=prediction_details
                )
            except Exception as e:
                self.logger.warning("Success prediction failed", error=str(e))

        react_steps: List[ReActStep] = []
        final_answer = None
        confidence_score = 0.5

        try:
            # Enhanced initial thought with context awareness
            thought = await self._generate_enhanced_thought(task, context, react_steps, optimal_template)
            react_steps.append(ReActStep("thought", thought, {"template_used": optimal_template.id if optimal_template else "default"}))

            # Adaptive ReAct loop with dynamic iteration adjustment
            max_iterations = self._calculate_optimal_iterations(task, context, success_prediction)

            for iteration in range(max_iterations):
                self.logger.info(
                    f"ReAct iteration {iteration + 1}/{max_iterations}",
                    correlation_id=context.correlation_id,
                    confidence=confidence_score
                )

                # Enhanced action generation with tool selection optimization
                action = await self._generate_enhanced_action(task, context, react_steps, iteration)
                react_steps.append(ReActStep("action", action, {"iteration": iteration, "confidence": confidence_score}))

                # Execute action with performance monitoring
                observation, action_success = await self._execute_enhanced_action(action, context)
                react_steps.append(ReActStep("observation", observation, {"success": action_success}))

                # Update confidence based on action success
                confidence_score = self._update_confidence(confidence_score, action_success, iteration)

                # Enhanced completion check with confidence threshold
                final_answer, completion_confidence = await self._enhanced_completion_check(
                    task, context, react_steps, confidence_score
                )

                if final_answer and completion_confidence >= self.confidence_threshold:
                    confidence_score = completion_confidence
                    break

                # Adaptive reasoning with learning from previous steps
                if iteration < max_iterations - 1:
                    next_thought = await self._generate_adaptive_thought(
                        task, context, react_steps, iteration, confidence_score
                    )
                    react_steps.append(ReActStep("thought", next_thought, {"adaptive": True}))

            # Advanced self-reflection with learning integration
            if final_answer:
                reflection = await self._generate_advanced_reflection(
                    task, final_answer, context, react_steps, confidence_score
                )
                react_steps.append(ReActStep("reflection", reflection, {"final_confidence": confidence_score}))

                # Store reflection in context with enhanced metadata
                await self.add_reflection(
                    "task_completion",
                    reflection,
                    confidence_score,
                    context
                )
            else:
                final_answer = self._generate_fallback_response(task, react_steps, confidence_score)
                reflection = self._generate_failure_reflection(react_steps, max_iterations)
                react_steps.append(ReActStep("reflection", reflection, {"failure_mode": True}))
                confidence_score = 0.3

            # Calculate performance metrics
            execution_time = (datetime.utcnow() - start_time).total_seconds()

            # Enhanced result with comprehensive metadata
            result = AgentResult(
                success=final_answer is not None and confidence_score >= 0.5,
                content=final_answer,
                agent_name=self.name,
                context=context,
                metadata={
                    "react_steps": [step.to_dict() for step in react_steps],
                    "iterations_used": len([s for s in react_steps if s.step_type == "thought"]),
                    "max_iterations": max_iterations,
                    "tools_used": list(set([s.metadata.get("tool_name") for s in react_steps if s.metadata.get("tool_name")])),
                    "execution_time_seconds": execution_time,
                    "task_complexity": task_complexity.value,
                    "template_used": optimal_template.id if optimal_template else "default",
                    "success_prediction": success_prediction,
                    "adaptive_features_used": True,
                    "learning_enabled": self.learning_enabled
                },
                confidence=confidence_score
            )

            # Learn from this interaction if learning is enabled
            if self.learning_engine:
                await self._learn_from_interaction(task, result, context, react_steps)

            # Record performance metrics
            if self.performance_monitor:
                await self.performance_monitor.record_agent_performance(
                    agent_name=self.name,
                    task_type=self._identify_domain(task),
                    success=result.success,
                    execution_time=execution_time,
                    confidence=confidence_score,
                    context=context.memory
                )

            # Store enhanced memory
            await self._store_enhanced_memory(task, result, context, react_steps)

            # Update success history
            self.success_history.append({
                "timestamp": datetime.utcnow(),
                "success": result.success,
                "confidence": confidence_score,
                "execution_time": execution_time,
                "complexity": task_complexity.value
            })

            # Keep only last 100 interactions
            if len(self.success_history) > 100:
                self.success_history = self.success_history[-100:]

            return result

        except Exception as e:
            self.log_error(e, {"correlation_id": context.correlation_id, "interaction_count": self.interaction_count})

            # Enhanced error handling with learning
            error_result = AgentResult(
                success=False,
                content=f"ReAct orchestration failed: {str(e)}",
                agent_name=self.name,
                context=context,
                metadata={
                    "react_steps": [step.to_dict() for step in react_steps],
                    "error_type": type(e).__name__,
                    "error_details": str(e),
                    "execution_time_seconds": (datetime.utcnow() - start_time).total_seconds(),
                    "failure_point": len(react_steps)
                },
                confidence=0.0
            )

            # Learn from failure if learning is enabled
            if self.learning_engine:
                await self._learn_from_failure(task, error_result, context, e)

            return error_result

    def _assess_task_complexity(self, task: str, context: AgentContext) -> PromptComplexity:
        """Assess task complexity using advanced heuristics."""
        complexity_score = 0

        # Length and structure indicators
        word_count = len(task.split())
        if word_count > 50:
            complexity_score += 2
        elif word_count > 20:
            complexity_score += 1

        # Multi-step indicators
        multi_step_keywords = ["then", "after", "next", "following", "step", "phase", "stage"]
        if any(keyword in task.lower() for keyword in multi_step_keywords):
            complexity_score += 2

        # Technical complexity
        technical_keywords = ["algorithm", "optimization", "architecture", "integration", "analysis", "complex"]
        technical_count = sum(1 for keyword in technical_keywords if keyword in task.lower())
        complexity_score += technical_count

        # Domain complexity
        advanced_domains = ["machine learning", "data science", "blockchain", "microservices", "distributed"]
        if any(domain in task.lower() for domain in advanced_domains):
            complexity_score += 3

        # Code generation complexity
        if "code" in task.lower() or "function" in task.lower():
            if any(lang in task.lower() for lang in ["python", "javascript", "java", "c++", "rust"]):
                complexity_score += 1
            if any(framework in task.lower() for framework in ["react", "django", "spring", "tensorflow"]):
                complexity_score += 2

        # Context complexity
        if len(context.memory) > 10:
            complexity_score += 1

        # Map score to complexity level
        if complexity_score >= 8:
            return PromptComplexity.EXPERT
        elif complexity_score >= 5:
            return PromptComplexity.ADVANCED
        elif complexity_score >= 2:
            return PromptComplexity.INTERMEDIATE
        else:
            return PromptComplexity.SIMPLE

    def _calculate_optimal_iterations(
        self,
        task: str,
        context: AgentContext,
        success_prediction: Optional[float]
    ) -> int:
        """Calculate optimal number of iterations based on task and prediction."""
        base_iterations = self.max_iterations

        # Adjust based on task complexity
        complexity = self._assess_task_complexity(task, context)
        if complexity == PromptComplexity.EXPERT:
            base_iterations = min(8, base_iterations + 3)
        elif complexity == PromptComplexity.ADVANCED:
            base_iterations = min(7, base_iterations + 2)
        elif complexity == PromptComplexity.SIMPLE:
            base_iterations = max(3, base_iterations - 2)

        # Adjust based on success prediction
        if success_prediction:
            if success_prediction > 0.8:
                base_iterations = max(3, base_iterations - 1)  # High confidence, fewer iterations
            elif success_prediction < 0.4:
                base_iterations = min(8, base_iterations + 2)  # Low confidence, more iterations

        return base_iterations

    async def _generate_enhanced_thought(
        self,
        task: str,
        context: AgentContext,
        react_steps: List[ReActStep],
        template: Optional[Any]
    ) -> str:
        """Generate enhanced initial thought using optimal template."""

        if template:
            # Use sophisticated template
            template_context = {
                "task": task,
                "domain": self._identify_domain(task),
                "complexity": self._assess_task_complexity(task, context).value,
                "tools": list(self.available_tools.keys()),
                "previous_context": context.memory.get("last_interaction", "None"),
                "examples": template.examples if hasattr(template, 'examples') else []
            }

            prompt = template.render(template_context)
        else:
            # Fallback to basic prompt
            prompt = f"""You are analyzing this task using advanced reasoning:

TASK: {task}

Think deeply about:
1. What is the user really asking for?
2. What domain knowledge is required?
3. What tools or information do I need?
4. What are potential challenges?

Provide your initial analysis:"""

        messages = [LLMMessage("user", prompt)]

        response = await self.llm_service.generate_response(
            messages=messages,
            temperature=0.7,
            max_tokens=500,
            task_type="reasoning"
        )

        return response.content

    async def _generate_enhanced_action(
        self,
        task: str,
        context: AgentContext,
        react_steps: List[ReActStep],
        iteration: int
    ) -> str:
        """Generate enhanced action with tool selection optimization."""

        # Build context from previous steps
        steps_context = "\n".join([
            f"{step.step_type.upper()}: {step.content}"
            for step in react_steps[-3:]  # Last 3 steps for context
        ])

        # Analyze which tools have been most successful
        successful_tools = self._analyze_tool_success_history()

        # Get domain-specific tool recommendations
        domain = self._identify_domain(task)
        recommended_tools = self._get_domain_tools(domain)

        available_tools_desc = "\n".join([
            f"- {tool}: {desc}" for tool, desc in {
                "web_search": "Search the internet for current information",
                "code_execution": "Execute Python code to solve problems",
                "knowledge_search": "Search internal knowledge base",
                "file_analysis": "Analyze uploaded files and documents",
                "data_analysis": "Perform statistical and data analysis",
                "api_integration": "Integrate with external APIs and services"
            }.items()
        ])

        system_prompt = f"""You are deciding the next action in an advanced ReAct loop.

ORIGINAL TASK: {task}
ITERATION: {iteration + 1}
DOMAIN: {domain}

PREVIOUS STEPS:
{steps_context}

AVAILABLE TOOLS:
{available_tools_desc}

TOOL PERFORMANCE INSIGHTS:
- Most successful tools recently: {', '.join(successful_tools[:3])}
- Recommended for {domain}: {', '.join(recommended_tools)}

DECISION FRAMEWORK:
1. Analyze what information/capability is needed next
2. Consider which tool is most likely to succeed
3. Avoid repeating failed approaches
4. Choose the most efficient path to solution

Respond with either:
1. "TOOL: tool_name | parameters" (e.g., "TOOL: web_search | query: latest AI developments")
2. "FINAL_ANSWER: your complete answer"

Choose the most strategic action:"""

        messages = [LLMMessage("user", system_prompt)]

        response = await self.llm_service.generate_response(
            messages=messages,
            temperature=0.3,
            max_tokens=300,
            task_type="reasoning"
        )

        return response.content

    async def _execute_enhanced_action(self, action: str, context: AgentContext) -> Tuple[str, bool]:
        """Execute action with enhanced monitoring and error handling."""

        action_start_time = datetime.utcnow()
        action_success = False

        try:
            if action.startswith("FINAL_ANSWER:"):
                return "Action completed - final answer provided", True

            if action.startswith("TOOL:"):
                # Parse tool call with enhanced error handling
                tool_part = action[5:].strip()
                if "|" in tool_part:
                    tool_name, params_str = tool_part.split("|", 1)
                    tool_name = tool_name.strip()
                    params_str = params_str.strip()
                else:
                    tool_name = tool_part.strip()
                    params_str = ""

                # Execute tool with monitoring
                if tool_name in self.available_tools:
                    try:
                        result = await self.available_tools[tool_name](params_str, context)
                        action_success = True

                        # Log tool usage with performance metrics
                        execution_time = (datetime.utcnow() - action_start_time).total_seconds()
                        await self.log_tool_call(
                            tool_name=tool_name,
                            tool_input={"params": params_str},
                            tool_output=result,
                            context=context,
                            execution_time_ms=int(execution_time * 1000)
                        )

                        return result, action_success

                    except Exception as tool_error:
                        error_msg = f"Tool '{tool_name}' failed: {str(tool_error)}"
                        self.logger.error("Tool execution failed", tool=tool_name, error=str(tool_error))
                        return error_msg, False
                else:
                    error_msg = f"Error: Tool '{tool_name}' not available. Available tools: {list(self.available_tools.keys())}"
                    return error_msg, False

            return f"Error: Invalid action format: {action}", False

        except Exception as e:
            error_msg = f"Action execution failed: {str(e)}"
            self.logger.error("Action execution error", error=str(e))
            return error_msg, False

    def _update_confidence(self, current_confidence: float, action_success: bool, iteration: int) -> float:
        """Update confidence score based on action success and iteration."""

        if action_success:
            # Boost confidence for successful actions
            confidence_boost = 0.2 * (1.0 - iteration / self.max_iterations)  # Diminishing boost
            new_confidence = min(1.0, current_confidence + confidence_boost)
        else:
            # Reduce confidence for failed actions
            confidence_penalty = 0.15 * (1.0 + iteration / self.max_iterations)  # Increasing penalty
            new_confidence = max(0.1, current_confidence - confidence_penalty)

        return new_confidence

    # Placeholder methods for missing functionality
    async def _enhanced_completion_check(self, task: str, context: AgentContext, react_steps: List[ReActStep], confidence: float) -> Tuple[Optional[str], float]:
        """Enhanced completion check - placeholder."""
        # Use existing completion check logic
        final_answer = await self._check_for_completion(task, context, react_steps)
        return final_answer, confidence

    async def _generate_adaptive_thought(self, task: str, context: AgentContext, react_steps: List[ReActStep], iteration: int, confidence: float) -> str:
        """Generate adaptive thought - placeholder."""
        return await self._generate_next_thought(task, context, react_steps)

    async def _generate_advanced_reflection(self, task: str, final_answer: str, context: AgentContext, react_steps: List[ReActStep], confidence: float) -> str:
        """Generate advanced reflection - placeholder."""
        return await self._generate_reflection(task, final_answer, context, react_steps)

    def _generate_fallback_response(self, task: str, react_steps: List[ReActStep], confidence: float) -> str:
        """Generate fallback response."""
        return "I was unable to complete the task within the maximum number of iterations. Please try rephrasing your request or breaking it into smaller parts."

    def _generate_failure_reflection(self, react_steps: List[ReActStep], max_iterations: int) -> str:
        """Generate failure reflection."""
        return f"Task incomplete due to iteration limit ({max_iterations}). Consider simplifying the request or providing more specific guidance."

    async def _learn_from_interaction(self, task: str, result: AgentResult, context: AgentContext, react_steps: List[ReActStep]) -> None:
        """Learn from interaction - placeholder."""
        if self.learning_engine:
            success_indicators = {
                "task_completed": result.success,
                "response_time": result.metadata.get("execution_time_seconds", 0),
                "error_occurred": not result.success
            }
            await self.learning_engine.learn_from_interaction(
                task, result.content, context.memory, success_indicators
            )

    async def _learn_from_failure(self, task: str, result: AgentResult, context: AgentContext, error: Exception) -> None:
        """Learn from failure - placeholder."""
        if self.learning_engine:
            success_indicators = {
                "task_completed": False,
                "error_occurred": True,
                "error_type": type(error).__name__
            }
            await self.learning_engine.learn_from_interaction(
                task, str(error), context.memory, success_indicators
            )

    async def _store_enhanced_memory(self, task: str, result: AgentResult, context: AgentContext, react_steps: List[ReActStep]) -> None:
        """Store enhanced memory - placeholder."""
        await self.store_memory("last_react_steps", [step.to_dict() for step in react_steps], context)

    def _analyze_tool_success_history(self) -> List[str]:
        """Analyze tool success history - placeholder."""
        return ["code_execution", "knowledge_search", "web_search"]

    def _get_domain_tools(self, domain: str) -> List[str]:
        """Get domain-specific tools - placeholder."""
        domain_tools = {
            "code": ["code_execution", "file_analysis"],
            "data": ["data_analysis", "knowledge_search"],
            "general": ["web_search", "knowledge_search"]
        }
        return domain_tools.get(domain, ["web_search", "knowledge_search"])

    def _identify_domain(self, task: str) -> str:
        """Identify domain from task - placeholder."""
        task_lower = task.lower()
        if any(keyword in task_lower for keyword in ["code", "function", "script", "programming"]):
            return "code"
        elif any(keyword in task_lower for keyword in ["data", "analysis", "statistics", "chart"]):
            return "data"
        else:
            return "general"
        """Execute task using ReAct pattern."""
        
        react_steps: List[ReActStep] = []
        final_answer = None

        # This code was replaced by the enhanced execute method above
        pass
    
    async def _generate_thought(self, task: str, context: AgentContext, react_steps: List[ReActStep]) -> str:
        """Generate initial thought about the task."""
        
        system_prompt = """You are a ReAct (Reasoning and Acting) orchestrator. Your job is to think through user requests step by step.

For the given task, provide your initial analysis:
1. What is the user asking for?
2. What information or capabilities might be needed?
3. What would be a good approach to solve this?

Be concise but thorough in your reasoning."""
        
        messages = [
            LLMMessage("system", system_prompt),
            LLMMessage("user", f"Task: {task}")
        ]
        
        response = await self.llm_service.generate_response(
            messages=messages,
            temperature=0.7,
            max_tokens=500,
            task_type="reasoning"
        )
        
        return response.content
    
    async def _generate_action(self, task: str, context: AgentContext, react_steps: List[ReActStep]) -> str:
        """Generate next action based on current state."""
        
        # Build context from previous steps
        steps_context = "\n".join([
            f"{step.step_type.upper()}: {step.content}"
            for step in react_steps[-3:]  # Last 3 steps for context
        ])
        
        available_tools_desc = "\n".join([
            f"- {tool}: {desc}" for tool, desc in {
                "web_search": "Search the internet for information",
                "code_execution": "Execute Python code to solve problems",
                "knowledge_search": "Search internal knowledge base",
                "file_analysis": "Analyze uploaded files"
            }.items()
        ])
        
        system_prompt = f"""You are deciding the next action in a ReAct loop. Based on the task and previous steps, choose what to do next.

Available tools:
{available_tools_desc}

You can also provide a final answer if you have enough information.

Previous steps:
{steps_context}

Respond with either:
1. "TOOL: tool_name | parameters" (e.g., "TOOL: web_search | query: latest AI news")
2. "FINAL_ANSWER: your complete answer"

Choose the most appropriate action."""
        
        messages = [
            LLMMessage("system", system_prompt),
            LLMMessage("user", f"Original task: {task}\n\nWhat should I do next?")
        ]
        
        response = await self.llm_service.generate_response(
            messages=messages,
            temperature=0.3,
            max_tokens=300,
            task_type="reasoning"
        )
        
        return response.content
    
    async def _execute_action(self, action: str, context: AgentContext) -> str:
        """Execute the chosen action."""
        
        if action.startswith("FINAL_ANSWER:"):
            return "Action completed - final answer provided"
        
        if action.startswith("TOOL:"):
            try:
                # Parse tool call
                tool_part = action[5:].strip()  # Remove "TOOL:"
                if "|" in tool_part:
                    tool_name, params_str = tool_part.split("|", 1)
                    tool_name = tool_name.strip()
                    params_str = params_str.strip()
                else:
                    tool_name = tool_part.strip()
                    params_str = ""
                
                # Execute tool
                if tool_name in self.available_tools:
                    result = await self.available_tools[tool_name](params_str, context)
                    
                    # Log tool usage
                    await self.log_tool_call(
                        tool_name=tool_name,
                        tool_input={"params": params_str},
                        tool_output=result,
                        context=context
                    )
                    
                    return result
                else:
                    return f"Error: Tool '{tool_name}' not available. Available tools: {list(self.available_tools.keys())}"
                    
            except Exception as e:
                return f"Error executing tool: {str(e)}"
        
        return f"Error: Invalid action format: {action}"
    
    async def _check_for_completion(self, task: str, context: AgentContext, react_steps: List[ReActStep]) -> Optional[str]:
        """Check if we have a complete answer."""
        
        # Look for FINAL_ANSWER in the last action
        last_action = None
        for step in reversed(react_steps):
            if step.step_type == "action":
                last_action = step.content
                break
        
        if last_action and last_action.startswith("FINAL_ANSWER:"):
            return last_action[13:].strip()  # Remove "FINAL_ANSWER:"
        
        return None
    
    async def _generate_next_thought(self, task: str, context: AgentContext, react_steps: List[ReActStep]) -> str:
        """Generate next thought based on observations."""
        
        recent_steps = "\n".join([
            f"{step.step_type.upper()}: {step.content}"
            for step in react_steps[-2:]  # Last 2 steps
        ])
        
        system_prompt = """Based on the recent action and observation, think about what to do next.

Consider:
1. Did the action provide useful information?
2. Do you need more information?
3. Are you ready to provide a final answer?
4. What should be the next step?"""
        
        messages = [
            LLMMessage("system", system_prompt),
            LLMMessage("user", f"Task: {task}\n\nRecent steps:\n{recent_steps}\n\nWhat should I think about next?")
        ]
        
        response = await self.llm_service.generate_response(
            messages=messages,
            temperature=0.7,
            max_tokens=300
        )
        
        return response.content
    
    async def _generate_reflection(self, task: str, final_answer: str, context: AgentContext, react_steps: List[ReActStep]) -> str:
        """Generate self-reflection on the completed task."""
        
        system_prompt = """Reflect on how well you completed the task. Consider:

1. Did you fully address the user's request?
2. Was your reasoning process effective?
3. Could you have been more efficient?
4. Is the final answer complete and accurate?
5. What could be improved for similar tasks?

Provide a brief, honest self-assessment."""
        
        messages = [
            LLMMessage("system", system_prompt),
            LLMMessage("user", f"Task: {task}\nFinal Answer: {final_answer}\n\nPlease reflect on this completion.")
        ]
        
        response = await self.llm_service.generate_response(
            messages=messages,
            temperature=0.5,
            max_tokens=200
        )
        
        return response.content
    
    # Tool implementations (placeholders for now)
    async def _web_search_tool(self, params: str, context: AgentContext) -> str:
        """Placeholder for web search tool."""
        return f"Web search results for: {params} (placeholder - implement with Kagi API)"
    
    async def _code_execution_tool(self, params: str, context: AgentContext) -> str:
        """Placeholder for code execution tool."""
        return f"Code execution result for: {params} (placeholder - implement with Docker sandbox)"
    
    async def _knowledge_search_tool(self, params: str, context: AgentContext) -> str:
        """Enhanced knowledge search using Milvus vector database."""
        try:
            if not self.vector_db_manager:
                return f"Knowledge search results for: {params} (vector database not available)"

            # Generate embedding for search query
            if hasattr(self.llm_service, 'generate_embeddings'):
                embeddings = await self.llm_service.generate_embeddings([params])
                query_embedding = embeddings[0] if embeddings else None
            else:
                query_embedding = None

            if query_embedding:
                # Search in vector database
                search_results = await self.vector_db_manager.search_vectors(
                    query_vector=query_embedding,
                    limit=5,
                    filter_conditions={}
                )

                if search_results:
                    results = []
                    for result in search_results:
                        metadata = result.get('metadata', {})
                        content = metadata.get('content', 'No content available')
                        score = result.get('score', 0.0)
                        results.append(f"Score: {score:.3f} - {content[:200]}...")

                    return f"Knowledge search results for '{params}':\n" + "\n".join(results)
                else:
                    return f"No relevant knowledge found for: {params}"
            else:
                # Fallback to simple text search
                return f"Knowledge search results for: {params} (using fallback search - embedding generation not available)"

        except Exception as e:
            self.logger.error("Knowledge search failed", error=str(e))
            return f"Knowledge search failed for: {params} (error: {str(e)})"
    
    async def _file_analysis_tool(self, params: str, context: AgentContext) -> str:
        """Placeholder for file analysis tool."""
        return f"File analysis result for: {params} (placeholder - implement file processing)"

    async def _data_analysis_tool(self, params: str, context: AgentContext) -> str:
        """Placeholder for data analysis tool."""
        return f"Data analysis result for: {params} (placeholder - implement statistical analysis)"

    async def _api_integration_tool(self, params: str, context: AgentContext) -> str:
        """Placeholder for API integration tool."""
        return f"API integration result for: {params} (placeholder - implement external API calls)"
