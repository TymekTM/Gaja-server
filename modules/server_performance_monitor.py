"""Performance monitoring for server-side latency tracking.

This module tracks server processing times for:
- AI module query processing
- TTS generation 
- WebSocket communication
- Overall server response times

Coordinates with client-side performance monitoring for complete pipeline analysis.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any
from statistics import mean, median, stdev

logger = logging.getLogger(__name__)


@dataclass
class ServerPerformanceMetrics:
    """Server-side performance metrics for a single query."""
    query_id: Optional[str] = None
    user_id: Optional[str] = None
    timestamp: float = 0.0
    
    # Server processing times (ms)
    websocket_receive_time: Optional[float] = None
    ai_processing_time: Optional[float] = None
    tts_generation_time: Optional[float] = None
    websocket_send_time: Optional[float] = None
    total_server_time: Optional[float] = None
    
    # Query context
    query_text: Optional[str] = None
    response_text: Optional[str] = None
    response_length: Optional[int] = None
    tts_audio_size: Optional[int] = None
    
    # Status
    error_occurred: bool = False
    error_message: Optional[str] = None


class ServerPerformanceMonitor:
    """Server-side performance monitoring."""
    
    def __init__(self):
        self.metrics: Dict[str, ServerPerformanceMetrics] = {}
        self._start_times: Dict[str, Dict[str, float]] = {}
        self.enabled = True
        
    def start_query_tracking(self, query_id: str, user_id: str, query_text: str = "") -> None:
        """Start tracking a server query."""
        if not self.enabled:
            return
            
        self.metrics[query_id] = ServerPerformanceMetrics(
            query_id=query_id,
            user_id=user_id,
            timestamp=time.time(),
            query_text=query_text
        )
        
        self._start_times[query_id] = {}
        logger.debug(f"🔍 Server tracking started for query {query_id}")
    
    def start_timer(self, query_id: str, component: str) -> None:
        """Start timing a server component."""
        if not self.enabled or query_id not in self._start_times:
            return
            
        self._start_times[query_id][component] = time.time()
        logger.debug(f"⏱️ Server timer started: {component} for query {query_id}")
    
    def end_timer(self, query_id: str, component: str) -> Optional[float]:
        """End timing for a server component."""
        if not self.enabled or query_id not in self._start_times:
            return None
            
        if component not in self._start_times[query_id]:
            logger.warning(f"Timer for {component} was not started for query {query_id}")
            return None
            
        duration_ms = (time.time() - self._start_times[query_id][component]) * 1000
        
        # Record in metrics
        if query_id in self.metrics:
            metric = self.metrics[query_id]
            
            if component == "websocket_receive":
                metric.websocket_receive_time = duration_ms
            elif component == "ai_processing":
                metric.ai_processing_time = duration_ms
            elif component == "tts_generation":
                metric.tts_generation_time = duration_ms
            elif component == "websocket_send":
                metric.websocket_send_time = duration_ms
            elif component == "total_server":
                metric.total_server_time = duration_ms
        
        logger.debug(f"⏱️ Server {component}: {duration_ms:.1f}ms for query {query_id}")
        del self._start_times[query_id][component]
        return duration_ms
    
    def record_response_data(self, query_id: str, response_text: str, tts_audio_size: Optional[int] = None) -> None:
        """Record response data for analysis."""
        if query_id in self.metrics:
            metric = self.metrics[query_id]
            metric.response_text = response_text
            metric.response_length = len(response_text) if response_text else 0
            metric.tts_audio_size = tts_audio_size
    
    def record_error(self, query_id: str, error_message: str) -> None:
        """Record an error for the query."""
        if query_id in self.metrics:
            self.metrics[query_id].error_occurred = True
            self.metrics[query_id].error_message = error_message
            logger.error(f"❌ Server error for query {query_id}: {error_message}")
    
    def finish_query(self, query_id: str) -> None:
        """Finish tracking a query."""
        if not self.enabled or query_id not in self.metrics:
            return
            
        metric = self.metrics[query_id]
        
        # Calculate total server time if not already set
        if not metric.total_server_time:
            total_time = 0.0
            for time_component in [
                metric.websocket_receive_time,
                metric.ai_processing_time, 
                metric.tts_generation_time,
                metric.websocket_send_time
            ]:
                if time_component:
                    total_time += time_component
            metric.total_server_time = total_time
        
        logger.info(f"✅ Server query {query_id} completed in {metric.total_server_time:.1f}ms")
        
        # Clean up start times
        if query_id in self._start_times:
            del self._start_times[query_id]
    
    def get_metrics(self, completed_only: bool = True) -> List[ServerPerformanceMetrics]:
        """Get collected metrics."""
        if completed_only:
            return [m for m in self.metrics.values() if m.total_server_time is not None]
        return list(self.metrics.values())
    
    def get_analysis(self) -> Dict[str, Any]:
        """Get server performance analysis."""
        completed_metrics = self.get_metrics(completed_only=True)
        successful_metrics = [m for m in completed_metrics if not m.error_occurred]
        
        if not successful_metrics:
            return {"error": "No successful server queries to analyze"}
        
        analysis = {
            "summary": {
                "total_queries": len(completed_metrics),
                "successful_queries": len(successful_metrics),
                "error_rate": (len(completed_metrics) - len(successful_metrics)) / len(completed_metrics) * 100 if completed_metrics else 0
            },
            "server_performance": {},
            "component_breakdown": {},
            "bottlenecks": {},
            "recommendations": []
        }
        
        # Analyze total server performance
        total_times = [m.total_server_time for m in successful_metrics if m.total_server_time]
        if total_times:
            analysis["server_performance"] = self._analyze_component(total_times)
        
        # Analyze each component
        components = [
            ("websocket_receive_time", "WebSocket Receive"),
            ("ai_processing_time", "AI Processing"),
            ("tts_generation_time", "TTS Generation"),
            ("websocket_send_time", "WebSocket Send")
        ]
        
        for attr, name in components:
            times = [getattr(m, attr) for m in successful_metrics if getattr(m, attr)]
            if times:
                analysis["component_breakdown"][name] = self._analyze_component(times)
        
        # Identify server bottlenecks
        avg_times = {}
        for name, stats in analysis["component_breakdown"].items():
            avg_times[name] = stats["mean"]
        
        if avg_times:
            sorted_components = sorted(avg_times.items(), key=lambda x: x[1], reverse=True)
            analysis["bottlenecks"] = {
                "slowest_server_component": sorted_components[0],
                "server_component_ranking": sorted_components
            }
            
            # Generate server-specific recommendations
            recommendations = self._generate_server_recommendations(analysis)
            analysis["recommendations"] = recommendations
        
        return analysis
    
    def _analyze_component(self, times: List[float]) -> Dict[str, float]:
        """Analyze timing statistics for a component."""
        if not times:
            return {}
            
        return {
            "mean": mean(times),
            "median": median(times),
            "min": min(times),
            "max": max(times),
            "std_dev": stdev(times) if len(times) > 1 else 0.0,
            "samples": len(times)
        }
    
    def _generate_server_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate server optimization recommendations."""
        recommendations = []
        
        bottlenecks = analysis.get("bottlenecks", {})
        component_ranking = bottlenecks.get("server_component_ranking", [])
        
        if not component_ranking:
            return recommendations
        
        # Check server performance
        server_stats = analysis.get("server_performance", {})
        avg_server = server_stats.get("mean", 0)
        
        if avg_server > 3000:  # 3 seconds
            recommendations.append(f"🚨 CRITICAL: Server processing time is {avg_server:.1f}ms (should be <3000ms)")
        elif avg_server > 2000:  # 2 seconds
            recommendations.append(f"⚠️ WARNING: Server processing time is {avg_server:.1f}ms (should be <2000ms)")
        else:
            recommendations.append(f"✅ GOOD: Server processing time is {avg_server:.1f}ms")
        
        # Analyze top server bottlenecks
        for i, (component, avg_time) in enumerate(component_ranking[:3]):
            if avg_time > 500:  # More than 500ms
                if "AI Processing" in component:
                    recommendations.append(f"🔧 Optimize {component} ({avg_time:.1f}ms): Consider model optimization or caching")
                elif "TTS Generation" in component:
                    recommendations.append(f"🔧 Optimize {component} ({avg_time:.1f}ms): Consider streaming or local TTS model")
                elif "WebSocket" in component:
                    recommendations.append(f"🔧 Optimize {component} ({avg_time:.1f}ms): Check network or serialization overhead")
                else:
                    recommendations.append(f"🔧 Optimize {component} ({avg_time:.1f}ms): Server bottleneck identified")
        
        return recommendations
    
    def save_results(self, filename: Optional[str] = None):
        """Save server performance results to file."""
        if not filename:
            timestamp = int(time.time())
            filename = f"server_performance_results_{timestamp}.json"
        
        results = {
            "metadata": {
                "completed_queries": len(self.get_metrics()),
                "timestamp": time.time(),
                "analysis": self.get_analysis()
            },
            "raw_metrics": [asdict(m) for m in self.get_metrics()]
        }
        
        # Save to user_data directory
        save_path = Path(__file__).parent.parent / "user_data" / filename
        save_path.parent.mkdir(exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Server performance results saved to {save_path}")
        return save_path
    
    def print_summary(self):
        """Print server performance summary."""
        analysis = self.get_analysis()
        
        print("\n" + "="*50)
        print("🖥️ SERVER PERFORMANCE ANALYSIS")
        print("="*50)
        
        summary = analysis.get("summary", {})
        print(f"📊 Total Queries: {summary.get('total_queries', 0)}")
        print(f"✅ Successful: {summary.get('successful_queries', 0)}")
        print(f"❌ Error Rate: {summary.get('error_rate', 0):.1f}%")
        
        server_stats = analysis.get("server_performance", {})
        if server_stats:
            print(f"\n⏱️  SERVER PERFORMANCE:")
            print(f"   Average: {server_stats.get('mean', 0):.1f}ms")
            print(f"   Median:  {server_stats.get('median', 0):.1f}ms")
            print(f"   Range:   {server_stats.get('min', 0):.1f}ms - {server_stats.get('max', 0):.1f}ms")
        
        print(f"\n🔧 SERVER COMPONENT BREAKDOWN:")
        breakdown = analysis.get("component_breakdown", {})
        for component, stats in breakdown.items():
            print(f"   {component:20} {stats.get('mean', 0):6.1f}ms (±{stats.get('std_dev', 0):5.1f}ms)")
        
        print(f"\n🎯 SERVER RECOMMENDATIONS:")
        recommendations = analysis.get("recommendations", [])
        for rec in recommendations:
            print(f"   {rec}")
        
        print("="*50 + "\n")


# Global instance
server_performance_monitor = ServerPerformanceMonitor()


# Convenience functions
def start_query_tracking(query_id: str, user_id: str, query_text: str = "") -> None:
    """Start tracking a server query."""
    server_performance_monitor.start_query_tracking(query_id, user_id, query_text)

def start_server_timer(query_id: str, component: str) -> None:
    """Start timing a server component."""
    server_performance_monitor.start_timer(query_id, component)

def end_server_timer(query_id: str, component: str) -> Optional[float]:
    """End timing for a server component."""
    return server_performance_monitor.end_timer(query_id, component)

def record_server_response_data(query_id: str, response_text: str, tts_audio_size: Optional[int] = None) -> None:
    """Record server response data."""
    server_performance_monitor.record_response_data(query_id, response_text, tts_audio_size)

def record_server_error(query_id: str, error_message: str) -> None:
    """Record a server error."""
    server_performance_monitor.record_error(query_id, error_message)

def finish_server_query(query_id: str) -> None:
    """Finish tracking a server query."""
    server_performance_monitor.finish_query(query_id)

def get_server_analysis() -> Dict[str, Any]:
    """Get server performance analysis."""
    return server_performance_monitor.get_analysis()

def save_server_results(filename: Optional[str] = None) -> Path:
    """Save server results to file."""
    return server_performance_monitor.save_results(filename)

def print_server_summary():
    """Print server performance summary."""
    server_performance_monitor.print_summary()
