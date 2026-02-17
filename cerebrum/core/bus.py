"""
Async event bus for CerebrumCoin.

Central nervous system that decouples all components. Publishers emit events,
subscribers receive them. No component ever calls another directly.

@decision DEC-BUS-001
@title Async event bus with type-based subscriptions
@status accepted
@rationale Enables hot-swapping components, plugin architecture, and future agentic
integration. Subscribers filter by EventType. Async queues prevent blocking.
Each subscriber gets its own queue to isolate failures.
"""

import asyncio
from collections import defaultdict
from typing import Callable, Coroutine

import structlog

from cerebrum.core.events import Event
from cerebrum.core.types import EventType

logger = structlog.get_logger()

# Type alias for subscriber callbacks
EventHandler = Callable[[Event], Coroutine[None, None, None]]


class EventBus:
    """
    Async event bus for decoupled component communication.

    Features:
    - Type-based subscriptions (subscribe to specific EventTypes)
    - Async event delivery via queues
    - Isolated subscriber queues (one failure doesn't block others)
    - Graceful shutdown with queue draining
    """

    def __init__(self, queue_size: int = 1000) -> None:
        """
        Initialize event bus.

        Args:
            queue_size: Maximum events queued per subscriber
        """
        self._subscribers: dict[EventType, list[tuple[str, asyncio.Queue[Event]]]] = defaultdict(list)
        self._queue_size = queue_size
        self._running = False
        self._tasks: list[asyncio.Task[None]] = []
        self._log = logger.bind(component="event_bus")

    def subscribe(
        self,
        event_type: EventType,
        handler: EventHandler,
        subscriber_name: str,
    ) -> None:
        """
        Subscribe to events of a specific type.

        Args:
            event_type: Type of events to receive
            handler: Async callback to handle events
            subscriber_name: Human-readable subscriber identifier
        """
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers[event_type].append((subscriber_name, queue))

        # Start async task to process this subscriber's queue
        task = asyncio.create_task(
            self._process_queue(subscriber_name, handler, queue),
            name=f"bus_subscriber_{subscriber_name}_{event_type.value}"
        )
        self._tasks.append(task)

        self._log.info(
            "subscriber_registered",
            event_type=event_type.value,
            subscriber=subscriber_name,
        )

    async def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribers of its type.

        Args:
            event: Event to publish
        """
        if not self._running:
            self._log.warning("publish_before_start", event_type=event.event_type.value)
            return

        subscribers = self._subscribers.get(event.event_type, [])

        if not subscribers:
            self._log.debug(
                "no_subscribers",
                event_type=event.event_type.value,
            )
            return

        # Fan out to all subscribers
        for subscriber_name, queue in subscribers:
            try:
                # Non-blocking put with timeout
                await asyncio.wait_for(
                    queue.put(event),
                    timeout=1.0
                )
            except asyncio.TimeoutError:
                self._log.error(
                    "subscriber_queue_full",
                    subscriber=subscriber_name,
                    event_type=event.event_type.value,
                )
            except Exception as e:
                self._log.error(
                    "publish_error",
                    subscriber=subscriber_name,
                    event_type=event.event_type.value,
                    error=str(e),
                )

        self._log.debug(
            "event_published",
            event_type=event.event_type.value,
            subscriber_count=len(subscribers),
        )

    async def _process_queue(
        self,
        subscriber_name: str,
        handler: EventHandler,
        queue: asyncio.Queue[Event],
    ) -> None:
        """
        Process events from a subscriber's queue.

        Runs until bus is stopped and queue is drained.
        """
        log = self._log.bind(subscriber=subscriber_name)
        log.info("subscriber_started")

        try:
            while self._running or not queue.empty():
                try:
                    # Wait for event with timeout to check running flag
                    event = await asyncio.wait_for(queue.get(), timeout=0.1)

                    try:
                        await handler(event)
                    except Exception as e:
                        log.error(
                            "handler_error",
                            event_type=event.event_type.value,
                            error=str(e),
                        )
                    finally:
                        queue.task_done()

                except asyncio.TimeoutError:
                    continue

        except asyncio.CancelledError:
            log.info("subscriber_cancelled")
        finally:
            log.info("subscriber_stopped", queue_size=queue.qsize())

    async def start(self) -> None:
        """Start the event bus."""
        self._running = True
        self._log.info("event_bus_started")

    async def stop(self) -> None:
        """Stop the event bus and drain all queues."""
        self._log.info("event_bus_stopping")
        self._running = False

        # Cancel all subscriber tasks
        for task in self._tasks:
            task.cancel()

        # Wait for graceful shutdown
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        self._log.info("event_bus_stopped")

    def get_subscriber_count(self, event_type: EventType) -> int:
        """Get number of subscribers for an event type."""
        return len(self._subscribers.get(event_type, []))
