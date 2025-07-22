#!/usr/bin/env python3
"""
GAJA Assistant - Security Monitoring and Rate Limiting System
System monitorowania bezpieczeństwa i ograniczania żądań.
"""

import time
from collections import defaultdict, deque
from datetime import UTC, datetime, timedelta
from typing import Any

from loguru import logger


class RateLimiter:
    """System ograniczania częstotliwości żądań."""

    def __init__(self):
        self.requests: dict[str, deque] = defaultdict(deque)
        self.blocked_ips: dict[str, datetime] = {}
        self.suspicious_ips: dict[str, int] = defaultdict(int)

        # Konfiguracja limitów
        self.limits = {
            "login": {"requests": 5, "window": 300},  # 5 prób logowania na 5 minut
            "api": {"requests": 100, "window": 60},  # 100 żądań API na minutę
            "upload": {"requests": 10, "window": 3600},  # 10 uploadów na godzinę
            "password_reset": {"requests": 3, "window": 1800},  # 3 resety na 30 minut
        }

        # Automatyczne blokowanie
        self.auto_block_thresholds = {
            "failed_login": 10,  # 10 nieudanych logowań = 1h blokada
            "suspicious_requests": 50,  # 50 podejrzanych żądań = 24h blokada
            "rate_limit_exceeded": 5,  # 5 przekroczeń = 1h blokada
        }

    def is_allowed(
        self,
        identifier: str,
        action: str = "api",
        request_metadata: dict[str, Any] | None = None,
        user_id: str | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """Sprawdza czy żądanie jest dozwolone.

        Args:
            identifier: Domyślny identyfikator (IP)
            action: Typ akcji (api, login, upload, etc.)
            request_metadata: Dodatkowe metadane żądania
            user_id: ID użytkownika (jeśli uwierzytelniony) - ma priorytet nad IP
        """
        now = time.time()

        # Użyj user_id jeśli dostępny, w przeciwnym razie IP
        rate_limit_key = f"user:{user_id}" if user_id else f"ip:{identifier}"

        # Sprawdź czy identyfikator jest zablokowany
        if rate_limit_key in self.blocked_ips:
            if datetime.now(UTC) < self.blocked_ips[rate_limit_key]:
                return False, {
                    "reason": "blocked",
                    "identifier_type": "user" if user_id else "ip",
                    "unblock_time": self.blocked_ips[rate_limit_key].isoformat(),
                    "retry_after": int(
                        (
                            self.blocked_ips[rate_limit_key] - datetime.now(UTC)
                        ).total_seconds()
                    ),
                }
            else:
                # Odblokuj identyfikator
                del self.blocked_ips[rate_limit_key]

        # Pobierz limit dla akcji z uwzględnieniem typu użytkownika
        limit_config = self._get_user_specific_limits(action, user_id)
        max_requests = limit_config["requests"]
        window_seconds = limit_config["window"]

        # Wyczyść stare żądania
        requests_queue = self.requests[rate_limit_key]
        while requests_queue and requests_queue[0] < now - window_seconds:
            requests_queue.popleft()

        # Sprawdź limit
        if len(requests_queue) >= max_requests:
            self._handle_rate_limit_exceeded(
                rate_limit_key, action, request_metadata, user_id
            )
            return False, {
                "reason": "rate_limit_exceeded",
                "identifier_type": "user" if user_id else "ip",
                "limit": max_requests,
                "window": window_seconds,
                "retry_after": int(window_seconds - (now - requests_queue[0]))
                if requests_queue
                else window_seconds,
            }

        # Dodaj żądanie i zwróć sukces
        requests_queue.append(now)
        return True, {
            "remaining": max_requests - len(requests_queue),
            "identifier_type": "user" if user_id else "ip",
        }

    def _get_user_specific_limits(
        self, action: str, user_id: str | None = None
    ) -> dict[str, int]:
        """Pobiera limity specyficzne dla użytkownika."""
        base_limit = self.limits.get(action, self.limits["api"])

        # Jeśli użytkownik jest uwierzytelniony, daj mu wyższe limity
        if user_id:
            # Zwiększ limity dla uwierzytelnionych użytkowników
            multiplier = 3  # 3x więcej requestów dla zalogowanych użytkowników
            return {
                "requests": base_limit["requests"] * multiplier,
                "window": base_limit["window"],
            }

        return base_limit

    def block_identifier(
        self,
        identifier: str,
        duration_hours: int = 24,
        reason: str = "security_violation",
    ) -> None:
        """Blokuje identyfikator (IP lub user) na określony czas."""
        try:
            unblock_time = datetime.now(UTC) + timedelta(hours=duration_hours)
            self.blocked_ips[identifier] = unblock_time

            identifier_type = "user" if identifier.startswith("user:") else "ip"
            logger.warning(
                f"{identifier_type.upper()} {identifier} blocked for {duration_hours}h: {reason}"
            )

        except Exception as e:
            logger.error(f"Failed to block {identifier}: {e}")

    def block_ip(
        self, ip: str, duration_hours: int = 24, reason: str = "security_violation"
    ) -> None:
        """Blokuje IP na określony czas (zachowane dla kompatybilności)."""
        ip_key = f"ip:{ip}"
        self.block_identifier(ip_key, duration_hours, reason)

    def _handle_rate_limit_exceeded(
        self,
        identifier: str,
        action: str,
        request_metadata: dict[str, Any] | None,
        user_id: str | None = None,
    ) -> None:
        """Obsługuje przekroczenie limitu żądań."""
        self.suspicious_ips[identifier] += 1

        identifier_type = "user" if user_id else "ip"
        logger.warning(
            f"Rate limit exceeded: {identifier} ({identifier_type}) for action {action}"
        )

        # Sprawdź czy należy zablokować identyfikator
        if (
            self.suspicious_ips[identifier]
            >= self.auto_block_thresholds["rate_limit_exceeded"]
        ):
            self.block_identifier(
                identifier, duration_hours=1, reason="repeated_rate_limit_violations"
            )

    def record_failed_login(self, ip: str, email: str) -> None:
        """Rejestruje nieudaną próbę logowania."""
        key = f"{ip}:failed_login"
        self.suspicious_ips[key] += 1

        if self.suspicious_ips[key] >= self.auto_block_thresholds["failed_login"]:
            self.block_ip(ip, duration_hours=1, reason="too_many_failed_logins")
            logger.warning(
                f"IP {ip} blocked due to {self.suspicious_ips[key]} failed login attempts"
            )

    def record_suspicious_activity(
        self, ip: str, activity_type: str, details: dict[str, Any]
    ) -> None:
        """Rejestruje podejrzaną aktywność."""
        key = f"{ip}:suspicious"
        self.suspicious_ips[key] += 1

        logger.warning(f"Suspicious activity from {ip}: {activity_type} - {details}")

        if (
            self.suspicious_ips[key]
            >= self.auto_block_thresholds["suspicious_requests"]
        ):
            self.block_ip(ip, duration_hours=24, reason="repeated_suspicious_activity")

    def get_stats(self) -> dict[str, Any]:
        """Zwraca statystyki rate limitera."""
        now = time.time()
        active_requests = 0

        for requests_queue in self.requests.values():
            # Wyczyść stare żądania z każdej kolejki
            while requests_queue and requests_queue[0] < now - 3600:  # 1 godzina
                requests_queue.popleft()
            active_requests += len(requests_queue)

        blocked_count = len(
            [
                ip
                for ip, unblock_time in self.blocked_ips.items()
                if datetime.now(UTC) < unblock_time
            ]
        )

        return {
            "active_requests_last_hour": active_requests,
            "blocked_ips_count": blocked_count,
            "suspicious_ips_count": len(self.suspicious_ips),
            "total_tracked_ips": len(self.requests),
            "blocked_ips": [
                {"ip": ip, "unblock_time": unblock_time.isoformat()}
                for ip, unblock_time in self.blocked_ips.items()
                if datetime.now(UTC) < unblock_time
            ],
        }


class SecurityMonitor:
    """System monitorowania bezpieczeństwa."""

    def __init__(self):
        self.security_events: list[dict[str, Any]] = []
        self.max_events = 10000  # Maksymalna liczba przechowywanych zdarzeń

        # Liczniki zagrożeń
        self.threat_counters = defaultdict(int)

        # Konfiguracja alertów
        self.alert_thresholds = {
            "failed_login_attempts": 20,
            "injection_attempts": 5,
            "suspicious_patterns": 10,
            "rate_limit_violations": 15,
            "blocked_ips": 5,
        }

    def log_security_event(
        self,
        event_type: str,
        details: dict[str, Any],
        severity: str = "medium",
        ip: str = "unknown",
    ) -> None:
        """Loguje zdarzenie bezpieczeństwa."""
        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": event_type,
            "severity": severity,
            "ip": ip,
            "details": details,
            "id": len(self.security_events) + 1,
        }

        self.security_events.append(event)
        self.threat_counters[event_type] += 1

        # Ogranicz liczbę przechowywanych zdarzeń
        if len(self.security_events) > self.max_events:
            self.security_events = self.security_events[-self.max_events :]

        # Loguj w zależności od powagi
        if severity == "critical":
            logger.critical(f"SECURITY ALERT: {event_type} from {ip} - {details}")
        elif severity == "high":
            logger.error(f"Security event: {event_type} from {ip} - {details}")
        elif severity == "medium":
            logger.warning(f"Security event: {event_type} from {ip} - {details}")
        else:
            logger.info(f"Security event: {event_type} from {ip} - {details}")

        # Sprawdź czy przekroczono progi alertów
        self._check_alert_thresholds(event_type)

    def _check_alert_thresholds(self, event_type: str) -> None:
        """Sprawdza czy przekroczono progi alertów."""
        threshold = self.alert_thresholds.get(event_type)
        if threshold and self.threat_counters[event_type] >= threshold:
            self.log_security_event(
                "alert_threshold_exceeded",
                {
                    "original_event_type": event_type,
                    "count": self.threat_counters[event_type],
                    "threshold": threshold,
                },
                severity="critical",
            )

    def get_security_summary(self, hours: int = 24) -> dict[str, Any]:
        """Zwraca podsumowanie bezpieczeństwa."""
        cutoff_time = datetime.now(UTC) - timedelta(hours=hours)

        recent_events = [
            event
            for event in self.security_events
            if datetime.fromisoformat(event["timestamp"]) > cutoff_time
        ]

        # Grupuj zdarzenia według typu
        event_types: dict[str, int] = defaultdict(int)
        severity_counts: dict[str, int] = defaultdict(int)
        ip_counts: dict[str, int] = defaultdict(int)

        for event in recent_events:
            event_types[event["event_type"]] += 1
            severity_counts[event["severity"]] += 1
            if event["ip"] != "unknown":
                ip_counts[event["ip"]] += 1

        # Znajdź najczęstsze IP
        top_ips = sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "period_hours": hours,
            "total_events": len(recent_events),
            "event_types": dict(event_types),
            "severity_distribution": dict(severity_counts),
            "top_source_ips": top_ips,
            "threat_level": self._calculate_threat_level(recent_events),
            "recommendations": self._generate_recommendations(recent_events),
        }

    def _calculate_threat_level(self, events: list[dict[str, Any]]) -> str:
        """Oblicza poziom zagrożenia."""
        if not events:
            return "low"

        critical_count = sum(1 for e in events if e["severity"] == "critical")
        high_count = sum(1 for e in events if e["severity"] == "high")

        if critical_count > 0 or high_count > 10:
            return "critical"
        elif high_count > 0 or len(events) > 50:
            return "high"
        elif len(events) > 20:
            return "medium"
        else:
            return "low"

    def _generate_recommendations(self, events: list[dict[str, Any]]) -> list[str]:
        """Generuje rekomendacje bezpieczeństwa."""
        recommendations = []

        # Analiza typów zdarzeń
        event_types: dict[str, int] = defaultdict(int)
        for event in events:
            event_types[event["event_type"]] += 1

        if event_types.get("failed_login", 0) > 10:
            recommendations.append("Consider implementing stronger password policies")
            recommendations.append("Enable two-factor authentication")

        if event_types.get("injection_attempt", 0) > 0:
            recommendations.append("Review input validation and sanitization")
            recommendations.append("Consider using a Web Application Firewall")

        if event_types.get("rate_limit_exceeded", 0) > 5:
            recommendations.append("Review rate limiting configuration")
            recommendations.append("Consider implementing progressive delays")

        if len({e["ip"] for e in events if e["ip"] != "unknown"}) > 20:
            recommendations.append(
                "High number of unique IPs - consider geographic filtering"
            )

        return recommendations

    def get_recent_events(
        self, limit: int = 100, severity: str | None = None
    ) -> list[dict[str, Any]]:
        """Zwraca ostatnie zdarzenia bezpieczeństwa."""
        events = self.security_events

        if severity:
            events = [e for e in events if e["severity"] == severity]

        return sorted(events, key=lambda x: x["timestamp"], reverse=True)[:limit]


class SecurityMiddleware:
    """Middleware bezpieczeństwa dla FastAPI."""

    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.security_monitor = SecurityMonitor()

    async def __call__(self, request, call_next):
        """Główna funkcja middleware."""
        start_time = time.time()

        # Pobierz IP klienta
        client_ip = self._get_client_ip(request)

        # Pobierz user_id z tokena jeśli dostępny
        user_id = None
        try:

            # Pobierz Authorization header
            authorization = request.headers.get("Authorization")
            if authorization and authorization.startswith("Bearer "):
                token = authorization.split(" ")[1]

                # Importuj funkcję sprawdzania tokena
                from server.api.routes import security_manager

                try:
                    payload = security_manager.verify_token(token, "access")
                    user_id = payload.get("userId") or payload.get("user_id")
                except Exception:
                    # Token nie jest ważny, user_id pozostaje None
                    pass

        except Exception as e:
            logger.debug(f"Could not extract user_id from token: {e}")

        # Sprawdź rate limiting
        is_allowed, rate_info = self.rate_limiter.is_allowed(
            client_ip,
            self._get_action_type(request),
            {"url": str(request.url), "method": request.method},
            user_id,
        )

        if not is_allowed:
            self.security_monitor.log_security_event(
                "rate_limit_exceeded",
                {"url": str(request.url), "rate_info": rate_info},
                severity="medium",
                ip=client_ip,
            )

            from fastapi import HTTPException

            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "retry_after": rate_info.get("retry_after", 60),
                },
                headers={"Retry-After": str(rate_info.get("retry_after", 60))},
            )

        # Przetwórz żądanie
        try:
            response = await call_next(request)

            # Loguj udane żądanie
            if response.status_code >= 400:
                self.security_monitor.log_security_event(
                    "http_error",
                    {
                        "status_code": response.status_code,
                        "url": str(request.url),
                        "method": request.method,
                    },
                    severity="low" if response.status_code < 500 else "medium",
                    ip=client_ip,
                )

            # Dodaj nagłówki bezpieczeństwa
            self._add_security_headers(response)

            return response

        except Exception as e:
            # Loguj błąd
            self.security_monitor.log_security_event(
                "request_error",
                {"error": str(e), "url": str(request.url), "method": request.method},
                severity="high",
                ip=client_ip,
            )
            raise

        finally:
            # Loguj czas przetwarzania
            process_time = time.time() - start_time
            if process_time > 10:  # Powolne żądania
                self.security_monitor.log_security_event(
                    "slow_request",
                    {"process_time": process_time, "url": str(request.url)},
                    severity="low",
                    ip=client_ip,
                )

    def _get_client_ip(self, request) -> str:
        """Pobiera IP klienta z nagłówków."""
        # Sprawdź nagłówki proxy
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fallback do IP z połączenia
        if hasattr(request, "client") and request.client:
            return request.client.host

        return "unknown"

    def _get_action_type(self, request) -> str:
        """Określa typ akcji na podstawie URL."""
        path = str(request.url.path).lower()

        if "/auth/login" in path:
            return "login"
        elif "/upload" in path or request.method == "POST":
            return "upload"
        elif "/auth/password-reset" in path:
            return "password_reset"
        else:
            return "api"

    def _add_security_headers(self, response) -> None:
        """Dodaje nagłówki bezpieczeństwa."""
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "microphone=(), camera=(), geolocation=()",
        }

        for header, value in security_headers.items():
            response.headers[header] = value


# Globalne instancje
security_middleware = SecurityMiddleware()
rate_limiter = security_middleware.rate_limiter
security_monitor = security_middleware.security_monitor


def get_security_stats() -> dict[str, Any]:
    """Zwraca statystyki bezpieczeństwa."""
    return {
        "rate_limiter": rate_limiter.get_stats(),
        "security_summary": security_monitor.get_security_summary(),
        "recent_events": security_monitor.get_recent_events(50),
    }


if __name__ == "__main__":
    """Test security monitoring system."""
    monitor = SecurityMonitor()
    limiter = RateLimiter()

    # Test rate limitera
    print("=== Rate Limiter Test ===")
    test_ip = "192.168.1.100"

    for i in range(10):
        allowed, info = limiter.is_allowed(test_ip, "login")
        print(f"Request {i+1}: {'ALLOWED' if allowed else 'BLOCKED'} - {info}")
        time.sleep(0.1)

    # Test monitoringu
    print("\n=== Security Monitor Test ===")

    # Symuluj zdarzenia
    monitor.log_security_event(
        "failed_login", {"email": "test@example.com"}, "medium", test_ip
    )
    monitor.log_security_event(
        "injection_attempt", {"payload": "'; DROP TABLE users;"}, "high", test_ip
    )
    monitor.log_security_event(
        "rate_limit_exceeded", {"action": "api"}, "medium", test_ip
    )

    # Podsumowanie
    summary = monitor.get_security_summary(1)
    print(f"Security summary: {summary}")

    print("\n✅ Security monitoring test completed!")
