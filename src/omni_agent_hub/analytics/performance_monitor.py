"""
Advanced Performance Monitoring and Analytics System.

This module provides comprehensive performance tracking, analytics, and 
optimization insights for the Omni-Agent Hub system.
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import asyncio
from enum import Enum

from ..core.logging import LoggerMixin
from ..services.redis_manager import RedisManager
from ..services.database import DatabaseManager


class MetricType(str, Enum):
    """Types of performance metrics."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    SUCCESS_RATE = "success_rate"
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"
    USER_SATISFACTION = "user_satisfaction"
    COST = "cost"
    QUALITY = "quality"


class AlertLevel(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Represents a performance metric measurement."""
    metric_type: MetricType
    value: float
    timestamp: datetime
    context: Dict[str, Any]
    tags: Dict[str, str]
    source: str


@dataclass
class PerformanceAlert:
    """Represents a performance alert."""
    alert_id: str
    level: AlertLevel
    metric_type: MetricType
    message: str
    value: float
    threshold: float
    timestamp: datetime
    context: Dict[str, Any]
    resolved: bool = False


@dataclass
class PerformanceInsight:
    """Represents an analytical insight."""
    insight_id: str
    title: str
    description: str
    impact: str
    recommendation: str
    confidence: float
    supporting_data: Dict[str, Any]
    timestamp: datetime


class PerformanceMonitor(LoggerMixin):
    """Advanced performance monitoring and analytics system."""
    
    def __init__(
        self,
        redis_manager: RedisManager,
        db_manager: DatabaseManager,
        alert_thresholds: Dict[str, float] = None
    ):
        self.redis_manager = redis_manager
        self.db_manager = db_manager
        
        # Performance data storage
        self.metrics_buffer: deque = deque(maxlen=10000)
        self.alerts: List[PerformanceAlert] = []
        self.insights: List[PerformanceInsight] = []
        
        # Alert thresholds
        self.alert_thresholds = alert_thresholds or {
            "latency_p95": 5.0,  # 5 seconds
            "error_rate": 0.05,  # 5%
            "success_rate": 0.95,  # 95%
            "memory_usage": 0.85,  # 85%
            "cpu_usage": 0.80,  # 80%
            "user_satisfaction": 0.70  # 70%
        }
        
        # Analytics state
        self.baseline_metrics: Dict[str, float] = {}
        self.trend_data: Dict[str, List[float]] = defaultdict(list)
        self.anomaly_detector = AnomalyDetector()
        
        # Performance tracking
        self.agent_performance: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.tool_performance: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.prompt_performance: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        self.logger.info("Performance monitor initialized")
    
    async def record_metric(
        self,
        metric_type: MetricType,
        value: float,
        context: Dict[str, Any] = None,
        tags: Dict[str, str] = None,
        source: str = "system"
    ) -> None:
        """Record a performance metric."""
        
        metric = PerformanceMetric(
            metric_type=metric_type,
            value=value,
            timestamp=datetime.utcnow(),
            context=context or {},
            tags=tags or {},
            source=source
        )
        
        # Add to buffer
        self.metrics_buffer.append(metric)
        
        # Store in Redis for real-time access
        await self.redis_manager.lpush(
            f"metrics:{metric_type.value}",
            json.dumps(asdict(metric), default=str)
        )
        
        # Keep only last 1000 metrics per type
        await self.redis_manager.ltrim(f"metrics:{metric_type.value}", 0, 999)
        
        # Update trend data
        self.trend_data[metric_type.value].append(value)
        if len(self.trend_data[metric_type.value]) > 100:
            self.trend_data[metric_type.value] = self.trend_data[metric_type.value][-100:]
        
        # Check for alerts
        await self._check_alerts(metric)
        
        # Detect anomalies
        await self._detect_anomalies(metric)
        
        self.logger.debug(
            "Recorded metric",
            type=metric_type.value,
            value=value,
            source=source
        )
    
    async def record_agent_performance(
        self,
        agent_name: str,
        task_type: str,
        success: bool,
        execution_time: float,
        confidence: float,
        context: Dict[str, Any] = None
    ) -> None:
        """Record agent-specific performance metrics."""
        
        agent_key = f"{agent_name}:{task_type}"
        
        if agent_key not in self.agent_performance:
            self.agent_performance[agent_key] = {
                "total_executions": 0,
                "successful_executions": 0,
                "total_time": 0.0,
                "avg_confidence": 0.0,
                "last_updated": datetime.utcnow()
            }
        
        perf = self.agent_performance[agent_key]
        perf["total_executions"] += 1
        if success:
            perf["successful_executions"] += 1
        perf["total_time"] += execution_time
        
        # Update average confidence
        alpha = 1.0 / perf["total_executions"]
        perf["avg_confidence"] = (1 - alpha) * perf["avg_confidence"] + alpha * confidence
        perf["last_updated"] = datetime.utcnow()
        
        # Record individual metrics
        await self.record_metric(
            MetricType.SUCCESS_RATE,
            perf["successful_executions"] / perf["total_executions"],
            context={"agent": agent_name, "task_type": task_type},
            source="agent_performance"
        )
        
        await self.record_metric(
            MetricType.LATENCY,
            execution_time,
            context={"agent": agent_name, "task_type": task_type},
            source="agent_performance"
        )
        
        await self.record_metric(
            MetricType.QUALITY,
            confidence,
            context={"agent": agent_name, "task_type": task_type},
            source="agent_performance"
        )
    
    async def record_tool_performance(
        self,
        tool_name: str,
        success: bool,
        execution_time: float,
        context: Dict[str, Any] = None
    ) -> None:
        """Record tool-specific performance metrics."""
        
        if tool_name not in self.tool_performance:
            self.tool_performance[tool_name] = {
                "total_uses": 0,
                "successful_uses": 0,
                "total_time": 0.0,
                "avg_time": 0.0,
                "last_updated": datetime.utcnow()
            }
        
        perf = self.tool_performance[tool_name]
        perf["total_uses"] += 1
        if success:
            perf["successful_uses"] += 1
        perf["total_time"] += execution_time
        perf["avg_time"] = perf["total_time"] / perf["total_uses"]
        perf["last_updated"] = datetime.utcnow()
        
        # Record metrics
        await self.record_metric(
            MetricType.SUCCESS_RATE,
            perf["successful_uses"] / perf["total_uses"],
            context={"tool": tool_name},
            source="tool_performance"
        )
        
        await self.record_metric(
            MetricType.LATENCY,
            execution_time,
            context={"tool": tool_name},
            source="tool_performance"
        )
    
    async def record_prompt_performance(
        self,
        prompt_id: str,
        success: bool,
        user_satisfaction: float,
        token_usage: int,
        context: Dict[str, Any] = None
    ) -> None:
        """Record prompt template performance metrics."""
        
        if prompt_id not in self.prompt_performance:
            self.prompt_performance[prompt_id] = {
                "total_uses": 0,
                "successful_uses": 0,
                "total_satisfaction": 0.0,
                "total_tokens": 0,
                "avg_satisfaction": 0.0,
                "avg_tokens": 0.0,
                "last_updated": datetime.utcnow()
            }
        
        perf = self.prompt_performance[prompt_id]
        perf["total_uses"] += 1
        if success:
            perf["successful_uses"] += 1
        perf["total_satisfaction"] += user_satisfaction
        perf["total_tokens"] += token_usage
        perf["avg_satisfaction"] = perf["total_satisfaction"] / perf["total_uses"]
        perf["avg_tokens"] = perf["total_tokens"] / perf["total_uses"]
        perf["last_updated"] = datetime.utcnow()
        
        # Record metrics
        await self.record_metric(
            MetricType.SUCCESS_RATE,
            perf["successful_uses"] / perf["total_uses"],
            context={"prompt_id": prompt_id},
            source="prompt_performance"
        )
        
        await self.record_metric(
            MetricType.USER_SATISFACTION,
            user_satisfaction,
            context={"prompt_id": prompt_id},
            source="prompt_performance"
        )
        
        await self.record_metric(
            MetricType.COST,
            token_usage * 0.00002,  # Approximate cost per token
            context={"prompt_id": prompt_id},
            source="prompt_performance"
        )
    
    async def _check_alerts(self, metric: PerformanceMetric) -> None:
        """Check if metric triggers any alerts."""
        
        alert_triggered = False
        alert_level = AlertLevel.INFO
        
        # Check specific thresholds
        if metric.metric_type == MetricType.LATENCY:
            if metric.value > self.alert_thresholds.get("latency_p95", 5.0):
                alert_triggered = True
                alert_level = AlertLevel.WARNING if metric.value < 10.0 else AlertLevel.ERROR
        
        elif metric.metric_type == MetricType.ERROR_RATE:
            if metric.value > self.alert_thresholds.get("error_rate", 0.05):
                alert_triggered = True
                alert_level = AlertLevel.ERROR if metric.value > 0.1 else AlertLevel.WARNING
        
        elif metric.metric_type == MetricType.SUCCESS_RATE:
            if metric.value < self.alert_thresholds.get("success_rate", 0.95):
                alert_triggered = True
                alert_level = AlertLevel.ERROR if metric.value < 0.8 else AlertLevel.WARNING
        
        elif metric.metric_type == MetricType.USER_SATISFACTION:
            if metric.value < self.alert_thresholds.get("user_satisfaction", 0.70):
                alert_triggered = True
                alert_level = AlertLevel.WARNING if metric.value > 0.5 else AlertLevel.ERROR
        
        if alert_triggered:
            alert = PerformanceAlert(
                alert_id=f"alert_{datetime.utcnow().timestamp()}",
                level=alert_level,
                metric_type=metric.metric_type,
                message=f"{metric.metric_type.value} threshold exceeded: {metric.value}",
                value=metric.value,
                threshold=self.alert_thresholds.get(f"{metric.metric_type.value}_threshold", 0),
                timestamp=datetime.utcnow(),
                context=metric.context
            )
            
            self.alerts.append(alert)
            
            # Store alert in Redis
            await self.redis_manager.lpush(
                "performance_alerts",
                json.dumps(asdict(alert), default=str)
            )
            
            self.logger.warning(
                "Performance alert triggered",
                level=alert_level.value,
                metric=metric.metric_type.value,
                value=metric.value
            )
    
    async def _detect_anomalies(self, metric: PerformanceMetric) -> None:
        """Detect anomalies in metric values."""
        
        metric_history = self.trend_data.get(metric.metric_type.value, [])
        
        if len(metric_history) >= 10:  # Need sufficient history
            is_anomaly, anomaly_score = self.anomaly_detector.detect(
                metric.value, metric_history
            )
            
            if is_anomaly:
                insight = PerformanceInsight(
                    insight_id=f"anomaly_{datetime.utcnow().timestamp()}",
                    title=f"Anomaly detected in {metric.metric_type.value}",
                    description=f"Unusual value {metric.value} detected (score: {anomaly_score:.2f})",
                    impact="Potential performance degradation or system issue",
                    recommendation="Investigate recent changes and system state",
                    confidence=anomaly_score,
                    supporting_data={
                        "metric_value": metric.value,
                        "historical_mean": np.mean(metric_history),
                        "historical_std": np.std(metric_history),
                        "anomaly_score": anomaly_score
                    },
                    timestamp=datetime.utcnow()
                )
                
                self.insights.append(insight)
                
                self.logger.info(
                    "Anomaly detected",
                    metric=metric.metric_type.value,
                    value=metric.value,
                    score=anomaly_score
                )
    
    async def generate_performance_report(
        self,
        time_range: timedelta = timedelta(hours=24)
    ) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        
        end_time = datetime.utcnow()
        start_time = end_time - time_range
        
        # Filter metrics by time range
        recent_metrics = [
            m for m in self.metrics_buffer
            if start_time <= m.timestamp <= end_time
        ]
        
        # Calculate summary statistics
        summary = self._calculate_summary_stats(recent_metrics)
        
        # Get top performers and issues
        top_agents = self._get_top_performers("agent")
        top_tools = self._get_top_performers("tool")
        top_prompts = self._get_top_performers("prompt")
        
        # Get recent alerts and insights
        recent_alerts = [
            a for a in self.alerts
            if start_time <= a.timestamp <= end_time
        ]
        
        recent_insights = [
            i for i in self.insights
            if start_time <= i.timestamp <= end_time
        ]
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(summary, recent_alerts)
        
        report = {
            "report_id": f"perf_report_{end_time.timestamp()}",
            "generated_at": end_time.isoformat(),
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "duration_hours": time_range.total_seconds() / 3600
            },
            "summary": summary,
            "top_performers": {
                "agents": top_agents,
                "tools": top_tools,
                "prompts": top_prompts
            },
            "alerts": [asdict(a) for a in recent_alerts],
            "insights": [asdict(i) for i in recent_insights],
            "recommendations": recommendations,
            "metrics_analyzed": len(recent_metrics)
        }
        
        return report
    
    def _calculate_summary_stats(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Calculate summary statistics from metrics."""
        
        if not metrics:
            return {}
        
        # Group metrics by type
        metrics_by_type = defaultdict(list)
        for metric in metrics:
            metrics_by_type[metric.metric_type.value].append(metric.value)
        
        summary = {}
        for metric_type, values in metrics_by_type.items():
            if values:
                summary[metric_type] = {
                    "count": len(values),
                    "mean": np.mean(values),
                    "median": np.median(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "p95": np.percentile(values, 95),
                    "p99": np.percentile(values, 99)
                }
        
        return summary
    
    def _get_top_performers(self, category: str) -> List[Dict[str, Any]]:
        """Get top performing entities in a category."""
        
        if category == "agent":
            performance_data = self.agent_performance
        elif category == "tool":
            performance_data = self.tool_performance
        elif category == "prompt":
            performance_data = self.prompt_performance
        else:
            return []
        
        # Sort by success rate and average performance
        sorted_performers = sorted(
            performance_data.items(),
            key=lambda x: (
                x[1].get("successful_uses", x[1].get("successful_executions", 0)) / 
                max(1, x[1].get("total_uses", x[1].get("total_executions", 1))),
                -x[1].get("avg_time", x[1].get("avg_satisfaction", 0))
            ),
            reverse=True
        )
        
        return [
            {
                "name": name,
                "performance": perf
            }
            for name, perf in sorted_performers[:5]
        ]
    
    async def _generate_recommendations(
        self,
        summary: Dict[str, Any],
        alerts: List[PerformanceAlert]
    ) -> List[str]:
        """Generate performance optimization recommendations."""
        
        recommendations = []
        
        # Latency recommendations
        if "latency" in summary:
            latency_stats = summary["latency"]
            if latency_stats["p95"] > 3.0:
                recommendations.append(
                    f"High latency detected (P95: {latency_stats['p95']:.2f}s). "
                    "Consider optimizing slow operations or adding caching."
                )
        
        # Error rate recommendations
        if "error_rate" in summary:
            error_stats = summary["error_rate"]
            if error_stats["mean"] > 0.05:
                recommendations.append(
                    f"High error rate detected ({error_stats['mean']:.1%}). "
                    "Review error logs and improve error handling."
                )
        
        # Success rate recommendations
        if "success_rate" in summary:
            success_stats = summary["success_rate"]
            if success_stats["mean"] < 0.90:
                recommendations.append(
                    f"Low success rate detected ({success_stats['mean']:.1%}). "
                    "Review task complexity and prompt effectiveness."
                )
        
        # Alert-based recommendations
        critical_alerts = [a for a in alerts if a.level == AlertLevel.CRITICAL]
        if critical_alerts:
            recommendations.append(
                f"Critical alerts detected ({len(critical_alerts)}). "
                "Immediate attention required for system stability."
            )
        
        # Tool performance recommendations
        underperforming_tools = [
            name for name, perf in self.tool_performance.items()
            if perf.get("successful_uses", 0) / max(1, perf.get("total_uses", 1)) < 0.8
        ]
        
        if underperforming_tools:
            recommendations.append(
                f"Underperforming tools detected: {', '.join(underperforming_tools)}. "
                "Consider tool optimization or replacement."
            )
        
        return recommendations


class AnomalyDetector:
    """Simple anomaly detection using statistical methods."""
    
    def detect(self, value: float, history: List[float]) -> Tuple[bool, float]:
        """Detect if value is anomalous compared to history."""
        
        if len(history) < 5:
            return False, 0.0
        
        mean = np.mean(history)
        std = np.std(history)
        
        if std == 0:
            return False, 0.0
        
        # Z-score based detection
        z_score = abs(value - mean) / std
        
        # Consider anomaly if z-score > 2.5
        is_anomaly = z_score > 2.5
        anomaly_score = min(1.0, z_score / 5.0)  # Normalize to 0-1
        
        return is_anomaly, anomaly_score
