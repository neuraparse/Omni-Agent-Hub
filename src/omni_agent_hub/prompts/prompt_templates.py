"""
Advanced Prompt Template Management System.

This module provides sophisticated prompt templates with dynamic variables,
conditional logic, and performance-based optimization.
"""

import json
import yaml
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass
from jinja2 import Template, Environment, BaseLoader
import re

from ..core.logging import LoggerMixin


class PromptType(str, Enum):
    """Types of prompts for different use cases."""
    REASONING = "reasoning"
    CODE_GENERATION = "code_generation"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    REFLECTION = "reflection"
    PLANNING = "planning"
    EXECUTION = "execution"
    CRITIQUE = "critique"
    SYNTHESIS = "synthesis"


class PromptComplexity(str, Enum):
    """Complexity levels for prompt selection."""
    SIMPLE = "simple"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class PromptMetrics:
    """Metrics for prompt performance tracking."""
    success_rate: float = 0.0
    avg_response_time: float = 0.0
    avg_token_usage: int = 0
    user_satisfaction: float = 0.0
    task_completion_rate: float = 0.0
    error_rate: float = 0.0
    usage_count: int = 0
    last_updated: str = ""


@dataclass
class PromptTemplate:
    """Advanced prompt template with metadata and optimization."""
    id: str
    name: str
    type: PromptType
    complexity: PromptComplexity
    template: str
    variables: Dict[str, Any]
    conditions: Dict[str, str]
    examples: List[Dict[str, str]]
    metrics: PromptMetrics
    tags: List[str]
    version: str = "1.0"
    author: str = "system"
    description: str = ""
    
    def render(self, context: Dict[str, Any]) -> str:
        """Render template with context variables."""
        env = Environment(loader=BaseLoader())
        template = env.from_string(self.template)
        return template.render(**context, **self.variables)


class PromptTemplateManager(LoggerMixin):
    """Advanced prompt template management with optimization."""
    
    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}
        self.performance_history: Dict[str, List[PromptMetrics]] = {}
        self.load_default_templates()
    
    def load_default_templates(self):
        """Load sophisticated default prompt templates."""
        
        # Advanced ReAct Reasoning Template
        react_template = PromptTemplate(
            id="react_advanced",
            name="Advanced ReAct Reasoning",
            type=PromptType.REASONING,
            complexity=PromptComplexity.ADVANCED,
            template="""You are an advanced AI agent using the ReAct (Reasoning and Acting) framework.

CONTEXT:
- Task: {{ task }}
- Domain: {{ domain | default("general") }}
- Complexity: {{ complexity | default("intermediate") }}
- Available Tools: {{ tools | join(", ") }}
- Previous Context: {{ previous_context | default("None") }}

REASONING FRAMEWORK:
1. **THOUGHT**: Analyze the task deeply, considering:
   - What is the user really asking for?
   - What domain knowledge is required?
   - What are the potential challenges?
   - What tools or information do I need?

2. **ACTION**: Choose the most appropriate action:
   - Use a tool if external information/execution is needed
   - Provide a direct answer if I have sufficient information
   - Ask for clarification if the request is ambiguous

3. **OBSERVATION**: Carefully analyze the results:
   - What did I learn from the action?
   - Does this fully address the user's need?
   - What additional steps might be required?

4. **REFLECTION**: Self-evaluate the process:
   - Was my reasoning sound?
   - Could I have been more efficient?
   - Is the user likely to be satisfied?

QUALITY STANDARDS:
- Be thorough but concise
- Show your reasoning process
- Acknowledge uncertainty when appropriate
- Provide actionable insights
- Consider edge cases and limitations

{% if examples %}
EXAMPLES:
{% for example in examples %}
User: {{ example.input }}
Response: {{ example.output }}
---
{% endfor %}
{% endif %}

Now, please process this request following the ReAct framework:""",
            variables={
                "max_iterations": 5,
                "confidence_threshold": 0.8,
                "reflection_enabled": True
            },
            conditions={
                "use_examples": "len(examples) > 0",
                "enable_reflection": "reflection_enabled == True"
            },
            examples=[
                {
                    "input": "Help me analyze customer churn data",
                    "output": "THOUGHT: The user wants to analyze customer churn, which requires data analysis skills..."
                }
            ],
            metrics=PromptMetrics(),
            tags=["reasoning", "react", "advanced", "multi-step"]
        )
        
        # Code Generation Template with Best Practices
        code_gen_template = PromptTemplate(
            id="code_generation_expert",
            name="Expert Code Generation",
            type=PromptType.CODE_GENERATION,
            complexity=PromptComplexity.EXPERT,
            template="""You are an expert software engineer with deep knowledge across multiple programming languages and paradigms.

TASK: {{ task }}
LANGUAGE: {{ language | default("Python") }}
FRAMEWORK: {{ framework | default("None specified") }}
REQUIREMENTS: {{ requirements | default("Standard best practices") }}

CODING STANDARDS:
1. **Clean Code Principles**:
   - Write self-documenting code
   - Use meaningful variable and function names
   - Follow language-specific conventions
   - Keep functions small and focused

2. **Security Considerations**:
   - Validate all inputs
   - Handle errors gracefully
   - Avoid common vulnerabilities
   - Use secure coding practices

3. **Performance Optimization**:
   - Consider time and space complexity
   - Use appropriate data structures
   - Optimize for the expected use case
   - Include performance comments where relevant

4. **Testing & Documentation**:
   - Include docstrings/comments
   - Suggest test cases
   - Explain complex logic
   - Provide usage examples

5. **Modern Best Practices**:
   - Use type hints (Python)
   - Follow async/await patterns where appropriate
   - Consider error handling strategies
   - Include logging where beneficial

RESPONSE FORMAT:
```{{ language | lower }}
# [Brief description of the solution]

[Your code here with comments]
```

**Explanation:**
[Explain the approach, key decisions, and any trade-offs]

**Usage Example:**
[Show how to use the code]

**Testing Suggestions:**
[Suggest test cases and edge cases to consider]

{% if complexity == "expert" %}
**Advanced Considerations:**
[Discuss scalability, maintainability, and potential improvements]
{% endif %}

Now, please generate the code:""",
            variables={
                "include_tests": True,
                "include_docs": True,
                "security_focus": True
            },
            conditions={
                "include_advanced": "complexity == 'expert'",
                "include_security": "security_focus == True"
            },
            examples=[],
            metrics=PromptMetrics(),
            tags=["code", "generation", "expert", "best-practices"]
        )
        
        # Advanced Analysis Template
        analysis_template = PromptTemplate(
            id="deep_analysis",
            name="Deep Analysis Framework",
            type=PromptType.ANALYSIS,
            complexity=PromptComplexity.ADVANCED,
            template="""You are a senior analyst with expertise in data interpretation, pattern recognition, and strategic insights.

ANALYSIS REQUEST: {{ request }}
DATA TYPE: {{ data_type | default("Mixed") }}
ANALYSIS DEPTH: {{ depth | default("Comprehensive") }}
STAKEHOLDER: {{ stakeholder | default("General") }}

ANALYSIS FRAMEWORK:

1. **INITIAL ASSESSMENT**:
   - What type of analysis is being requested?
   - What are the key questions to answer?
   - What limitations or assumptions should I note?

2. **DATA EXAMINATION**:
   - What patterns are immediately visible?
   - What anomalies or outliers exist?
   - What additional context might be needed?

3. **DEEP DIVE ANALYSIS**:
   - Statistical significance of findings
   - Correlation vs causation considerations
   - Trend analysis and projections
   - Comparative analysis where relevant

4. **INSIGHTS & IMPLICATIONS**:
   - Key findings and their significance
   - Business/practical implications
   - Risk factors and opportunities
   - Confidence levels in conclusions

5. **RECOMMENDATIONS**:
   - Actionable next steps
   - Priority ranking of recommendations
   - Resource requirements
   - Success metrics to track

6. **VALIDATION & LIMITATIONS**:
   - Assumptions made in the analysis
   - Potential biases or limitations
   - Suggestions for validation
   - Areas requiring further investigation

RESPONSE STRUCTURE:
## Executive Summary
[2-3 sentence overview of key findings]

## Detailed Analysis
[Comprehensive analysis following the framework above]

## Key Insights
- [Bullet points of most important findings]

## Recommendations
1. [Prioritized, actionable recommendations]

## Next Steps
[Specific actions to take]

{% if stakeholder == "executive" %}
## Strategic Implications
[High-level strategic considerations]
{% endif %}

Now, please conduct the analysis:""",
            variables={
                "include_visualizations": True,
                "confidence_intervals": True,
                "executive_summary": True
            },
            conditions={
                "executive_focus": "stakeholder == 'executive'",
                "technical_details": "depth == 'technical'"
            },
            examples=[],
            metrics=PromptMetrics(),
            tags=["analysis", "insights", "strategic", "comprehensive"]
        )
        
        # Self-Reflection and Improvement Template
        reflection_template = PromptTemplate(
            id="self_reflection_advanced",
            name="Advanced Self-Reflection",
            type=PromptType.REFLECTION,
            complexity=PromptComplexity.EXPERT,
            template="""You are conducting a thorough self-reflection on your previous response and reasoning process.

ORIGINAL TASK: {{ original_task }}
YOUR RESPONSE: {{ your_response }}
CONTEXT: {{ context | default("Standard interaction") }}
PERFORMANCE METRICS: {{ metrics | default("Not available") }}

REFLECTION FRAMEWORK:

1. **RESPONSE QUALITY ASSESSMENT**:
   - Accuracy: How factually correct was my response?
   - Completeness: Did I address all aspects of the request?
   - Clarity: Was my explanation clear and well-structured?
   - Relevance: Did I stay focused on what was asked?

2. **REASONING PROCESS EVALUATION**:
   - Logic: Was my reasoning sound and well-founded?
   - Methodology: Did I use appropriate analytical approaches?
   - Assumptions: What assumptions did I make, and were they valid?
   - Bias Check: Did any cognitive biases influence my response?

3. **EFFICIENCY ANALYSIS**:
   - Resource Usage: Did I use tools and information efficiently?
   - Time Management: Could I have reached the conclusion faster?
   - Iteration Count: Was the number of steps appropriate?
   - Alternative Approaches: What other methods could I have used?

4. **USER EXPERIENCE EVALUATION**:
   - Helpfulness: How useful was my response to the user?
   - Engagement: Was the interaction smooth and natural?
   - Satisfaction: Would the user likely be satisfied?
   - Follow-up Needs: What might the user need next?

5. **LEARNING OPPORTUNITIES**:
   - Knowledge Gaps: What did I not know that I should have?
   - Skill Improvements: What capabilities could I develop?
   - Process Refinements: How could I improve my approach?
   - Pattern Recognition: What patterns can I learn from this interaction?

6. **FUTURE OPTIMIZATION**:
   - Template Improvements: How could the prompt template be better?
   - Context Utilization: How could I better use available context?
   - Tool Selection: Could I have chosen better tools or approaches?
   - Quality Metrics: What metrics should I track for improvement?

REFLECTION OUTPUT:

## Overall Assessment
**Quality Score**: [1-10 with justification]
**Efficiency Score**: [1-10 with justification]
**User Satisfaction Prediction**: [1-10 with justification]

## Strengths Identified
- [What went well in this interaction]

## Areas for Improvement
- [Specific areas that could be enhanced]

## Lessons Learned
- [Key insights from this interaction]

## Optimization Recommendations
- [Specific suggestions for future similar tasks]

## Confidence Calibration
- [How confident was I vs how confident should I have been]

Now, please conduct this self-reflection:""",
            variables={
                "include_metrics": True,
                "learning_enabled": True,
                "optimization_focus": True
            },
            conditions={
                "detailed_metrics": "include_metrics == True",
                "learning_mode": "learning_enabled == True"
            },
            examples=[],
            metrics=PromptMetrics(),
            tags=["reflection", "self-improvement", "quality", "learning"]
        )
        
        # Store templates
        self.templates = {
            "react_advanced": react_template,
            "code_generation_expert": code_gen_template,
            "deep_analysis": analysis_template,
            "self_reflection_advanced": reflection_template
        }
        
        self.logger.info("Loaded advanced prompt templates", count=len(self.templates))
    
    def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """Get template by ID."""
        return self.templates.get(template_id)
    
    def get_best_template(
        self,
        prompt_type: PromptType,
        complexity: PromptComplexity,
        context: Dict[str, Any] = None
    ) -> Optional[PromptTemplate]:
        """Get the best performing template for given criteria."""
        candidates = [
            template for template in self.templates.values()
            if template.type == prompt_type and template.complexity == complexity
        ]
        
        if not candidates:
            # Fallback to any template of the same type
            candidates = [
                template for template in self.templates.values()
                if template.type == prompt_type
            ]
        
        if not candidates:
            return None
        
        # Sort by performance metrics
        candidates.sort(
            key=lambda t: (
                t.metrics.success_rate * 0.4 +
                t.metrics.user_satisfaction * 0.3 +
                t.metrics.task_completion_rate * 0.3
            ),
            reverse=True
        )
        
        return candidates[0]
    
    def update_metrics(self, template_id: str, metrics: PromptMetrics):
        """Update performance metrics for a template."""
        if template_id in self.templates:
            self.templates[template_id].metrics = metrics
            
            # Store in history
            if template_id not in self.performance_history:
                self.performance_history[template_id] = []
            
            self.performance_history[template_id].append(metrics)
            
            # Keep only last 100 entries
            if len(self.performance_history[template_id]) > 100:
                self.performance_history[template_id] = self.performance_history[template_id][-100:]
            
            self.logger.info(
                "Updated template metrics",
                template_id=template_id,
                success_rate=metrics.success_rate
            )
    
    def optimize_template(self, template_id: str, feedback: Dict[str, Any]) -> bool:
        """Optimize template based on performance feedback."""
        template = self.templates.get(template_id)
        if not template:
            return False
        
        # Simple optimization logic - can be enhanced with ML
        if feedback.get("too_verbose", False):
            # Make template more concise
            template.template = self._make_concise(template.template)
        
        if feedback.get("needs_more_examples", False):
            # Add more examples if available
            template.examples.extend(feedback.get("suggested_examples", []))
        
        if feedback.get("unclear_instructions", False):
            # Enhance clarity
            template.template = self._enhance_clarity(template.template)
        
        return True
    
    def _make_concise(self, template: str) -> str:
        """Make template more concise."""
        # Remove excessive explanations, keep core structure
        lines = template.split('\n')
        filtered_lines = []
        
        for line in lines:
            # Keep important structural elements
            if any(keyword in line.lower() for keyword in ['thought:', 'action:', 'observation:', 'framework:', 'standards:']):
                filtered_lines.append(line)
            # Remove overly verbose explanations
            elif len(line.strip()) > 100 and not line.strip().startswith('-'):
                # Shorten long explanatory lines
                filtered_lines.append(line[:100] + "...")
            else:
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    def _enhance_clarity(self, template: str) -> str:
        """Enhance template clarity."""
        # Add more explicit instructions and structure
        enhanced = template.replace(
            "Now, please",
            "IMPORTANT: Follow the framework above step by step. Now, please"
        )
        
        return enhanced
