"""Simple Proactive Assistant Module for GAJA A basic implementation to get the system
working."""

import logging
import time
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SimpleNotification:
    """Simple notification data structure."""

    id: str
    type: str
    priority: str
    title: str
    message: str
    timestamp: float
    dismissed: bool = False


class SimpleProactiveAssistant:
    """Simple proactive assistant for basic functionality."""

    def __init__(self):
        self.notifications = []
        self.user_contexts = {}
        self.is_active = False
        self.notification_id_counter = 1

    def start(self):
        """Start the proactive assistant."""
        self.is_active = True
        logger.info("Simple Proactive Assistant started")

        # Add a welcome notification
        self._add_simple_notification(
            "system",
            "info",
            "Proactive Assistant Ready",
            "Your proactive assistant is now monitoring your activity and ready to help optimize your workflow.",
        )

    def stop(self):
        """Stop the proactive assistant."""
        self.is_active = False
        logger.info("Simple Proactive Assistant stopped")

    def _add_simple_notification(
        self, type_: str, priority: str, title: str, message: str
    ):
        """Add a simple notification."""
        notification = SimpleNotification(
            id=str(self.notification_id_counter),
            type=type_,
            priority=priority,
            title=title,
            message=message,
            timestamp=time.time(),
        )
        self.notifications.append(notification)
        self.notification_id_counter += 1
        return notification

    # Async interface methods for server integration
    async def get_notifications(self, user_id: str) -> list[dict]:
        """Get notifications for user."""
        try:
            # Return non-dismissed notifications
            active_notifications = [
                {
                    "id": notif.id,
                    "type": notif.type,
                    "priority": notif.priority,
                    "title": notif.title,
                    "message": notif.message,
                    "timestamp": datetime.fromtimestamp(notif.timestamp).isoformat(),
                    "dismissed": notif.dismissed,
                }
                for notif in self.notifications
                if not notif.dismissed
            ]
            return active_notifications
        except Exception as e:
            logger.error(f"Error getting notifications: {e}")
            return []

    async def add_notification(
        self,
        user_id: str,
        notification_type: str,
        message: str,
        priority: str = "medium",
        title: str = None,
        suggested_action: str = None,
        context: dict = None,
    ):
        """Add a notification."""
        try:
            if not title:
                title = f"{notification_type.title()} Notification"

            self._add_simple_notification(notification_type, priority, title, message)
            return True
        except Exception as e:
            logger.error(f"Error adding notification: {e}")
            return False

    async def dismiss_notification(self, user_id: str, notification_id: str) -> bool:
        """Dismiss a notification."""
        try:
            for notif in self.notifications:
                if notif.id == notification_id:
                    notif.dismissed = True
                    return True
            return False
        except Exception as e:
            logger.error(f"Error dismissing notification: {e}")
            return False

    async def update_user_context(self, user_id: str, context_data: dict):
        """Update user context."""
        try:
            self.user_contexts[user_id] = {
                **self.user_contexts.get(user_id, {}),
                **context_data,
                "last_update": time.time(),
            }

            # Generate contextual notifications based on activity
            await self._generate_contextual_notifications(user_id, context_data)

        except Exception as e:
            logger.error(f"Error updating user context: {e}")

    async def get_predictions(self, user_id: str) -> list[dict]:
        """Get predictions for user."""
        try:
            # Simple predictions based on current context
            predictions = [
                {
                    "type": "productivity",
                    "message": "Based on your activity patterns, consider taking a short break soon.",
                    "confidence": 0.7,
                    "suggested_action": "Take a 5-minute break and hydrate",
                }
            ]
            return predictions
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return []

    async def _generate_contextual_notifications(
        self, user_id: str, context_data: dict
    ):
        """Generate notifications based on user context."""
        try:
            current_context = self.user_contexts.get(user_id, {})

            # Example: If user has been active for a while, suggest a break
            if "active_window" in context_data:
                activity_count = current_context.get("activity_count", 0) + 1
                self.user_contexts[user_id]["activity_count"] = activity_count

                # Suggest break every 10 context updates (simplified)
                if activity_count % 10 == 0:
                    await self.add_notification(
                        user_id,
                        "wellness",
                        "Consider taking a short break to refresh your mind and maintain productivity.",
                        "medium",
                        "Break Reminder",
                    )

        except Exception as e:
            logger.error(f"Error generating contextual notifications: {e}")


# Singleton instance
_simple_proactive_assistant = None


def get_proactive_assistant():
    """Get the singleton proactive assistant instance."""
    global _simple_proactive_assistant
    if _simple_proactive_assistant is None:
        _simple_proactive_assistant = SimpleProactiveAssistant()
    return _simple_proactive_assistant
