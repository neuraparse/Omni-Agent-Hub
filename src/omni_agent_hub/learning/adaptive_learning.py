"""
Adaptive Learning System for Omni-Agent Hub.

This module implements sophisticated learning mechanisms that allow the system
to improve over time based on user interactions, success patterns, and feedback.
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import asyncio

from ..core.logging import LoggerMixin
from ..services.redis_manager import RedisManager
from ..services.database import DatabaseManager


@dataclass
class InteractionPattern:
    """Represents a learned interaction pattern."""
    pattern_id: str
    user_input_features: Dict[str, Any]
    successful_response_features: Dict[str, Any]
    context_features: Dict[str, Any]
    success_rate: float
    usage_count: int
    last_updated: datetime
    confidence_score: float


@dataclass
class LearningMetrics:
    """Metrics for tracking learning performance."""
    total_interactions: int = 0
    successful_interactions: int = 0
    failed_interactions: int = 0
    avg_response_time: float = 0.0
    avg_user_satisfaction: float = 0.0
    pattern_recognition_accuracy: float = 0.0
    adaptation_speed: float = 0.0
    learning_efficiency: float = 0.0


class AdaptiveLearningEngine(LoggerMixin):
    """Advanced adaptive learning engine with pattern recognition and optimization."""
    
    def __init__(self, redis_manager: RedisManager, db_manager: DatabaseManager):
        self.redis_manager = redis_manager
        self.db_manager = db_manager
        
        # Learning state
        self.interaction_patterns: Dict[str, InteractionPattern] = {}
        self.user_preferences: Dict[str, Dict[str, Any]] = {}
        self.context_embeddings: Dict[str, np.ndarray] = {}
        self.success_predictors: Dict[str, float] = {}
        
        # Learning parameters
        self.learning_rate = 0.1
        self.pattern_threshold = 0.8
        self.min_pattern_occurrences = 5
        self.memory_window = timedelta(days=30)
        
        # Performance tracking
        self.metrics = LearningMetrics()
        self.recent_interactions = deque(maxlen=1000)
        
        self.logger.info("Adaptive learning engine initialized")
    
    async def learn_from_interaction(
        self,
        user_input: str,
        agent_response: str,
        context: Dict[str, Any],
        success_indicators: Dict[str, Any],
        user_feedback: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        agent_name: Optional[str] = None
    ) -> None:
        """Learn from a single interaction."""
        
        interaction_data = {
            "timestamp": datetime.utcnow(),
            "user_input": user_input,
            "agent_response": agent_response,
            "context": context,
            "success_indicators": success_indicators,
            "user_feedback": user_feedback or {},
            "session_id": session_id,
            "agent_name": agent_name
        }
        
        # Add to recent interactions
        self.recent_interactions.append(interaction_data)
        
        # Extract features
        input_features = self._extract_input_features(user_input, context)
        response_features = self._extract_response_features(agent_response)
        
        # Determine success
        success_score = self._calculate_success_score(success_indicators, user_feedback)
        
        # Update patterns
        await self._update_patterns(input_features, response_features, context, success_score)
        
        # Update user preferences
        await self._update_user_preferences(context.get("user_id"), input_features, success_score)
        
        # Update metrics
        self._update_metrics(success_score)
        
        # Store learning data
        await self._store_learning_data(interaction_data, success_score)
        
        self.logger.info(
            "Learned from interaction",
            success_score=success_score,
            pattern_count=len(self.interaction_patterns)
        )
    
    def _extract_input_features(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from user input and context."""
        features = {
            "length": len(user_input),
            "word_count": len(user_input.split()),
            "has_question": "?" in user_input,
            "has_code_request": any(keyword in user_input.lower() for keyword in ["code", "function", "script", "program"]),
            "has_analysis_request": any(keyword in user_input.lower() for keyword in ["analyze", "analysis", "report", "insights"]),
            "complexity_indicators": self._assess_complexity(user_input),
            "domain": self._identify_domain(user_input),
            "urgency": self._assess_urgency(user_input),
            "specificity": self._assess_specificity(user_input),
            "context_richness": len(context),
            "session_length": context.get("session_length", 1),
            "user_expertise": context.get("user_expertise", "intermediate"),
            "time_of_day": datetime.utcnow().hour,
            "day_of_week": datetime.utcnow().weekday()
        }
        
        return features
    
    def _extract_response_features(self, agent_response: str) -> Dict[str, Any]:
        """Extract features from agent response."""
        features = {
            "length": len(agent_response),
            "word_count": len(agent_response.split()),
            "has_code": "```" in agent_response,
            "has_examples": "example" in agent_response.lower(),
            "has_steps": any(indicator in agent_response.lower() for indicator in ["step", "1.", "2.", "first", "then"]),
            "has_explanation": len(agent_response.split(".")) > 3,
            "confidence_indicators": self._count_confidence_words(agent_response),
            "structure_quality": self._assess_structure(agent_response),
            "technical_depth": self._assess_technical_depth(agent_response)
        }
        
        return features
    
    def _calculate_success_score(
        self,
        success_indicators: Dict[str, Any],
        user_feedback: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate overall success score for the interaction."""
        score = 0.0
        weight_sum = 0.0
        
        # Success indicators from system
        if "task_completed" in success_indicators:
            score += success_indicators["task_completed"] * 0.3
            weight_sum += 0.3
        
        if "response_time" in success_indicators:
            # Normalize response time (lower is better)
            time_score = max(0, 1 - (success_indicators["response_time"] / 30.0))  # 30s baseline
            score += time_score * 0.2
            weight_sum += 0.2
        
        if "error_occurred" in success_indicators:
            error_penalty = 1.0 if not success_indicators["error_occurred"] else 0.0
            score += error_penalty * 0.2
            weight_sum += 0.2
        
        # User feedback
        if user_feedback:
            if "satisfaction" in user_feedback:
                score += user_feedback["satisfaction"] * 0.3
                weight_sum += 0.3
            
            if "helpful" in user_feedback:
                score += (1.0 if user_feedback["helpful"] else 0.0) * 0.2
                weight_sum += 0.2
            
            if "accurate" in user_feedback:
                score += (1.0 if user_feedback["accurate"] else 0.0) * 0.2
                weight_sum += 0.2
        
        # Normalize score
        if weight_sum > 0:
            score = score / weight_sum
        else:
            score = 0.5  # Default neutral score
        
        return min(1.0, max(0.0, score))
    
    async def _update_patterns(
        self,
        input_features: Dict[str, Any],
        response_features: Dict[str, Any],
        context: Dict[str, Any],
        success_score: float
    ) -> None:
        """Update interaction patterns based on new data."""
        
        # Create pattern signature
        pattern_signature = self._create_pattern_signature(input_features, context)
        
        if pattern_signature in self.interaction_patterns:
            # Update existing pattern
            pattern = self.interaction_patterns[pattern_signature]
            
            # Exponential moving average for success rate
            alpha = self.learning_rate
            pattern.success_rate = (1 - alpha) * pattern.success_rate + alpha * success_score
            pattern.usage_count += 1
            pattern.last_updated = datetime.utcnow()
            
            # Update confidence based on usage count and consistency
            pattern.confidence_score = min(1.0, pattern.usage_count / 20.0) * (
                1.0 - abs(pattern.success_rate - success_score)
            )
            
        else:
            # Create new pattern
            pattern = InteractionPattern(
                pattern_id=pattern_signature,
                user_input_features=input_features,
                successful_response_features=response_features,
                context_features=context,
                success_rate=success_score,
                usage_count=1,
                last_updated=datetime.utcnow(),
                confidence_score=0.1  # Low initial confidence
            )
            
            self.interaction_patterns[pattern_signature] = pattern
        
        # Store pattern in Redis for fast access
        await self.redis_manager.set(
            f"pattern:{pattern_signature}",
            json.dumps(asdict(pattern), default=str),
            expire=86400 * 30  # 30 days
        )
    
    def _create_pattern_signature(
        self,
        input_features: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Create a unique signature for interaction patterns."""
        
        # Key features for pattern matching
        key_features = [
            input_features.get("domain", "general"),
            input_features.get("complexity_indicators", "medium"),
            "code" if input_features.get("has_code_request", False) else "text",
            "analysis" if input_features.get("has_analysis_request", False) else "general",
            context.get("user_expertise", "intermediate"),
            "urgent" if input_features.get("urgency", 0) > 0.7 else "normal"
        ]
        
        return "|".join(str(f) for f in key_features)
    
    async def predict_success(
        self,
        user_input: str,
        context: Dict[str, Any],
        proposed_response: str
    ) -> Tuple[float, Dict[str, Any]]:
        """Predict success probability for a proposed response."""
        
        input_features = self._extract_input_features(user_input, context)
        response_features = self._extract_response_features(proposed_response)
        pattern_signature = self._create_pattern_signature(input_features, context)
        
        # Check for matching patterns
        if pattern_signature in self.interaction_patterns:
            pattern = self.interaction_patterns[pattern_signature]
            base_prediction = pattern.success_rate * pattern.confidence_score
        else:
            # Use similar patterns
            similar_patterns = self._find_similar_patterns(input_features, context)
            if similar_patterns:
                base_prediction = np.mean([p.success_rate for p in similar_patterns])
            else:
                base_prediction = 0.5  # Neutral prediction
        
        # Adjust based on response features
        adjustments = self._calculate_response_adjustments(response_features, input_features)
        final_prediction = min(1.0, max(0.0, base_prediction + adjustments))
        
        prediction_details = {
            "base_prediction": base_prediction,
            "adjustments": adjustments,
            "pattern_match": pattern_signature in self.interaction_patterns,
            "confidence": self.interaction_patterns.get(pattern_signature, {}).confidence_score if pattern_signature in self.interaction_patterns else 0.5
        }
        
        return final_prediction, prediction_details
    
    def _find_similar_patterns(
        self,
        input_features: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[InteractionPattern]:
        """Find similar interaction patterns."""
        
        similar_patterns = []
        target_domain = input_features.get("domain", "general")
        target_complexity = input_features.get("complexity_indicators", "medium")
        
        for pattern in self.interaction_patterns.values():
            # Check domain similarity
            if pattern.user_input_features.get("domain") == target_domain:
                similarity_score = 0.8
            elif self._domains_related(pattern.user_input_features.get("domain"), target_domain):
                similarity_score = 0.6
            else:
                similarity_score = 0.3
            
            # Check complexity similarity
            if pattern.user_input_features.get("complexity_indicators") == target_complexity:
                similarity_score += 0.2
            
            # Require minimum similarity and confidence
            if similarity_score >= 0.6 and pattern.confidence_score >= 0.3:
                similar_patterns.append(pattern)
        
        # Sort by confidence and return top 5
        similar_patterns.sort(key=lambda p: p.confidence_score, reverse=True)
        return similar_patterns[:5]
    
    def _calculate_response_adjustments(
        self,
        response_features: Dict[str, Any],
        input_features: Dict[str, Any]
    ) -> float:
        """Calculate adjustments to success prediction based on response features."""
        
        adjustment = 0.0
        
        # Code requests should have code in response
        if input_features.get("has_code_request", False):
            if response_features.get("has_code", False):
                adjustment += 0.1
            else:
                adjustment -= 0.2
        
        # Analysis requests should have structured responses
        if input_features.get("has_analysis_request", False):
            if response_features.get("structure_quality", 0) > 0.7:
                adjustment += 0.1
            else:
                adjustment -= 0.1
        
        # Complex requests need detailed responses
        if input_features.get("complexity_indicators", "medium") == "high":
            if response_features.get("technical_depth", 0) > 0.7:
                adjustment += 0.1
            else:
                adjustment -= 0.15
        
        # Response length should match request complexity
        expected_length = self._estimate_expected_length(input_features)
        actual_length = response_features.get("length", 0)
        
        if 0.7 <= actual_length / expected_length <= 1.5:
            adjustment += 0.05
        else:
            adjustment -= 0.1
        
        return adjustment
    
    # Helper methods for feature extraction
    def _assess_complexity(self, text: str) -> str:
        """Assess complexity of user input."""
        complexity_indicators = [
            "complex", "advanced", "sophisticated", "detailed", "comprehensive",
            "multi-step", "integration", "optimization", "algorithm", "architecture"
        ]
        
        simple_indicators = [
            "simple", "basic", "quick", "easy", "straightforward"
        ]
        
        text_lower = text.lower()
        complex_count = sum(1 for indicator in complexity_indicators if indicator in text_lower)
        simple_count = sum(1 for indicator in simple_indicators if indicator in text_lower)
        
        if complex_count > simple_count and len(text.split()) > 20:
            return "high"
        elif simple_count > complex_count or len(text.split()) < 10:
            return "low"
        else:
            return "medium"
    
    def _identify_domain(self, text: str) -> str:
        """Identify the domain of the request."""
        domains = {
            "code": ["code", "programming", "function", "script", "algorithm", "debug"],
            "data": ["data", "analysis", "analytics", "statistics", "visualization"],
            "business": ["business", "strategy", "market", "revenue", "profit"],
            "technical": ["technical", "system", "infrastructure", "deployment"],
            "creative": ["creative", "design", "content", "writing", "marketing"]
        }
        
        text_lower = text.lower()
        domain_scores = {}
        
        for domain, keywords in domains.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        else:
            return "general"
    
    def _assess_urgency(self, text: str) -> float:
        """Assess urgency of the request."""
        urgent_words = ["urgent", "asap", "immediately", "quickly", "fast", "emergency"]
        text_lower = text.lower()
        
        urgency_score = sum(1 for word in urgent_words if word in text_lower)
        return min(1.0, urgency_score / 3.0)
    
    def _assess_specificity(self, text: str) -> float:
        """Assess how specific the request is."""
        specific_indicators = ["specific", "exactly", "precisely", "particular", "detailed"]
        vague_indicators = ["something", "anything", "general", "overview", "basic"]
        
        text_lower = text.lower()
        specific_count = sum(1 for indicator in specific_indicators if indicator in text_lower)
        vague_count = sum(1 for indicator in vague_indicators if indicator in text_lower)
        
        # Also consider presence of specific details
        has_numbers = any(char.isdigit() for char in text)
        has_specific_terms = len([word for word in text.split() if len(word) > 8]) > 2
        
        specificity = (specific_count - vague_count) / 5.0
        if has_numbers:
            specificity += 0.2
        if has_specific_terms:
            specificity += 0.2
        
        return min(1.0, max(0.0, specificity))
    
    def _count_confidence_words(self, text: str) -> float:
        """Count confidence indicators in response."""
        confident_words = ["definitely", "certainly", "clearly", "obviously", "precisely"]
        uncertain_words = ["maybe", "perhaps", "possibly", "might", "could be"]
        
        text_lower = text.lower()
        confident_count = sum(1 for word in confident_words if word in text_lower)
        uncertain_count = sum(1 for word in uncertain_words if word in text_lower)
        
        return (confident_count - uncertain_count) / 10.0
    
    def _assess_structure(self, text: str) -> float:
        """Assess structural quality of response."""
        structure_indicators = [
            "##" in text,  # Headers
            "1." in text or "2." in text,  # Numbered lists
            "- " in text,  # Bullet points
            "```" in text,  # Code blocks
            len(text.split("\n")) > 5  # Multiple paragraphs
        ]
        
        structure_score = sum(1 for indicator in structure_indicators if indicator) / len(structure_indicators)
        return structure_score
    
    def _assess_technical_depth(self, text: str) -> float:
        """Assess technical depth of response."""
        technical_terms = [
            "algorithm", "implementation", "optimization", "architecture", "framework",
            "protocol", "interface", "abstraction", "encapsulation", "inheritance"
        ]
        
        text_lower = text.lower()
        technical_count = sum(1 for term in technical_terms if term in text_lower)
        
        # Normalize by text length
        words_count = len(text.split())
        if words_count > 0:
            return min(1.0, technical_count / (words_count / 100))
        else:
            return 0.0
    
    def _domains_related(self, domain1: str, domain2: str) -> bool:
        """Check if two domains are related."""
        related_domains = {
            "code": ["technical", "data"],
            "data": ["code", "business"],
            "business": ["data", "creative"],
            "technical": ["code"],
            "creative": ["business"]
        }
        
        return domain2 in related_domains.get(domain1, [])
    
    def _estimate_expected_length(self, input_features: Dict[str, Any]) -> int:
        """Estimate expected response length based on input features."""
        base_length = 500  # Base response length
        
        # Adjust for complexity
        complexity = input_features.get("complexity_indicators", "medium")
        if complexity == "high":
            base_length *= 2
        elif complexity == "low":
            base_length *= 0.5
        
        # Adjust for request type
        if input_features.get("has_code_request", False):
            base_length *= 1.5
        
        if input_features.get("has_analysis_request", False):
            base_length *= 1.8
        
        return int(base_length)
    
    async def _update_user_preferences(
        self,
        user_id: Optional[str],
        input_features: Dict[str, Any],
        success_score: float
    ) -> None:
        """Update user-specific preferences."""
        if not user_id:
            return
        
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {
                "preferred_complexity": "medium",
                "preferred_response_length": "medium",
                "domains_of_interest": [],
                "interaction_count": 0,
                "avg_satisfaction": 0.0
            }
        
        prefs = self.user_preferences[user_id]
        prefs["interaction_count"] += 1
        
        # Update average satisfaction
        alpha = 1.0 / prefs["interaction_count"]
        prefs["avg_satisfaction"] = (1 - alpha) * prefs["avg_satisfaction"] + alpha * success_score
        
        # Track domain interests
        domain = input_features.get("domain", "general")
        if domain not in prefs["domains_of_interest"]:
            prefs["domains_of_interest"].append(domain)
        
        # Store in Redis
        await self.redis_manager.set(
            f"user_prefs:{user_id}",
            json.dumps(prefs),
            expire=86400 * 90  # 90 days
        )
    
    def _update_metrics(self, success_score: float) -> None:
        """Update learning metrics."""
        self.metrics.total_interactions += 1
        
        if success_score >= 0.7:
            self.metrics.successful_interactions += 1
        elif success_score <= 0.3:
            self.metrics.failed_interactions += 1
        
        # Update averages
        alpha = 1.0 / self.metrics.total_interactions
        self.metrics.avg_user_satisfaction = (
            (1 - alpha) * self.metrics.avg_user_satisfaction + alpha * success_score
        )
    
    async def _store_learning_data(
        self,
        interaction_data: Dict[str, Any],
        success_score: float
    ) -> None:
        """Store learning data for future analysis."""
        learning_record = {
            **interaction_data,
            "success_score": success_score,
            "pattern_count": len(self.interaction_patterns),
            "learning_metrics": asdict(self.metrics)
        }
        
        # Store in database for long-term analysis
        if self.db_manager:
            try:
                await self.db_manager.execute_command(
                    """
                    INSERT INTO learning_interactions
                    (timestamp, user_input, agent_response, context, success_score, metrics, session_id, agent_name)
                    VALUES (:timestamp, :user_input, :agent_response, :context, :success_score, :metrics, :session_id, :agent_name)
                    """,
                    {
                        "timestamp": interaction_data["timestamp"],
                        "user_input": interaction_data["user_input"],
                        "agent_response": interaction_data["agent_response"],
                        "context": json.dumps(interaction_data["context"]),
                        "success_score": success_score,
                        "metrics": json.dumps(asdict(self.metrics)),
                        "session_id": interaction_data.get("session_id"),
                        "agent_name": interaction_data.get("agent_name")
                    }
                )
            except Exception as e:
                self.logger.error("Failed to store learning data", error=str(e))
    
    async def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from learning data."""
        insights = {
            "total_patterns": len(self.interaction_patterns),
            "high_confidence_patterns": len([
                p for p in self.interaction_patterns.values() 
                if p.confidence_score >= 0.8
            ]),
            "avg_success_rate": np.mean([
                p.success_rate for p in self.interaction_patterns.values()
            ]) if self.interaction_patterns else 0.0,
            "most_successful_domains": self._get_top_domains_by_success(),
            "learning_metrics": asdict(self.metrics),
            "recent_performance_trend": self._calculate_performance_trend()
        }
        
        return insights
    
    def _get_top_domains_by_success(self) -> List[Tuple[str, float]]:
        """Get domains with highest success rates."""
        domain_stats = defaultdict(list)
        
        for pattern in self.interaction_patterns.values():
            domain = pattern.user_input_features.get("domain", "general")
            domain_stats[domain].append(pattern.success_rate)
        
        domain_averages = [
            (domain, np.mean(scores))
            for domain, scores in domain_stats.items()
        ]
        
        domain_averages.sort(key=lambda x: x[1], reverse=True)
        return domain_averages[:5]
    
    def _calculate_performance_trend(self) -> str:
        """Calculate recent performance trend."""
        if len(self.recent_interactions) < 10:
            return "insufficient_data"
        
        recent_scores = []
        for interaction in list(self.recent_interactions)[-20:]:
            # Calculate success score from stored data
            success_indicators = interaction.get("success_indicators", {})
            user_feedback = interaction.get("user_feedback", {})
            score = self._calculate_success_score(success_indicators, user_feedback)
            recent_scores.append(score)
        
        if len(recent_scores) >= 10:
            first_half = np.mean(recent_scores[:len(recent_scores)//2])
            second_half = np.mean(recent_scores[len(recent_scores)//2:])
            
            if second_half > first_half + 0.1:
                return "improving"
            elif second_half < first_half - 0.1:
                return "declining"
            else:
                return "stable"
        
        return "stable"
