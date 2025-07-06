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
from typing import Any, Dict, List, Optional
from datetime import datetime

from .base_agent import BaseAgent, AgentCapability, AgentContext, AgentResult
from ..services.llm_service import LLMService, LLMMessage, LLMProvider
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
    """Main orchestrator agent implementing ReAct pattern."""
    
    def __init__(self, llm_service: LLMService, **kwargs):
        capabilities = [
            AgentCapability(
                name="orchestrate",
                description="Coordinate multiple agents and tools to solve complex tasks",
                parameters={"max_iterations": 5, "reflection_enabled": True}
            ),
            AgentCapability(
                name="reason",
                description="Analyze user requests and plan multi-step solutions",
                parameters={"reasoning_depth": "deep"}
            ),
            AgentCapability(
                name="delegate",
                description="Delegate tasks to specialized agents",
                parameters={"available_agents": ["codeact", "search", "analysis"]}
            )
        ]
        
        super().__init__(
            name="ReActOrchestrator",
            description="Main orchestrator implementing ReAct reasoning pattern",
            capabilities=capabilities,
            **kwargs
        )
        
        self.llm_service = llm_service
        self.max_iterations = 5
        self.available_tools = {
            "web_search": self._web_search_tool,
            "code_execution": self._code_execution_tool,
            "knowledge_search": self._knowledge_search_tool,
            "file_analysis": self._file_analysis_tool
        }
    
    async def execute(self, task: str, context: AgentContext) -> AgentResult:
        """Execute task using ReAct pattern."""
        
        react_steps: List[ReActStep] = []
        final_answer = None
        
        try:
            # Initial thought
            thought = await self._generate_thought(task, context, react_steps)
            react_steps.append(ReActStep("thought", thought))
            
            # ReAct loop
            for iteration in range(self.max_iterations):
                self.logger.info(f"ReAct iteration {iteration + 1}", correlation_id=context.correlation_id)
                
                # Action: Decide what to do next
                action = await self._generate_action(task, context, react_steps)
                react_steps.append(ReActStep("action", action))
                
                # Execute the action
                observation = await self._execute_action(action, context)
                react_steps.append(ReActStep("observation", observation))
                
                # Check if we have a final answer
                final_answer = await self._check_for_completion(task, context, react_steps)
                if final_answer:
                    break
                
                # Continue reasoning if not complete
                if iteration < self.max_iterations - 1:
                    next_thought = await self._generate_next_thought(task, context, react_steps)
                    react_steps.append(ReActStep("thought", next_thought))
            
            # Self-reflection on the final result
            if final_answer:
                reflection = await self._generate_reflection(task, final_answer, context, react_steps)
                react_steps.append(ReActStep("reflection", reflection))
                
                # Store reflection in context
                await self.add_reflection(
                    "task_completion",
                    reflection,
                    0.8,  # Default confidence
                    context
                )
            else:
                final_answer = "I was unable to complete the task within the maximum number of iterations. Please try rephrasing your request or breaking it into smaller parts."
                reflection = "Task incomplete due to iteration limit. Consider simplifying the request."
                react_steps.append(ReActStep("reflection", reflection))
            
            # Store the ReAct steps in memory
            await self.store_memory("last_react_steps", [step.to_dict() for step in react_steps], context)
            
            return AgentResult(
                success=final_answer is not None,
                content=final_answer,
                agent_name=self.name,
                context=context,
                metadata={
                    "react_steps": [step.to_dict() for step in react_steps],
                    "iterations_used": len([s for s in react_steps if s.step_type == "thought"]),
                    "tools_used": list(set([s.metadata.get("tool_name") for s in react_steps if s.metadata.get("tool_name")]))
                },
                confidence=0.8 if final_answer else 0.3
            )
            
        except Exception as e:
            self.log_error(e, {"correlation_id": context.correlation_id})
            return AgentResult(
                success=False,
                content=f"ReAct orchestration failed: {str(e)}",
                agent_name=self.name,
                context=context,
                metadata={"react_steps": [step.to_dict() for step in react_steps]}
            )
    
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
            max_tokens=500
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
            max_tokens=300
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
        """Placeholder for knowledge search tool."""
        return f"Knowledge search results for: {params} (placeholder - implement with Milvus)"
    
    async def _file_analysis_tool(self, params: str, context: AgentContext) -> str:
        """Placeholder for file analysis tool."""
        return f"File analysis result for: {params} (placeholder - implement file processing)"
