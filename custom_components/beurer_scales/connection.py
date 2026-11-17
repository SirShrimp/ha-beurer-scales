"""Reaching a scale from Home Assistant: holding it, waiting for a wake and
opening a session.

The scale takes one connection at a time, so whoever talks to it (a flow, or
the coordinator) holds it first with `async_hold_scale`. Flows wait their turn
and keep the scale from the wait for the wake through the end of their session;
the coordinator never waits behind a flow: it leaves the scale alone while
`async_scale_held` and syncs at a later wake.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from functools import partial
import logging
import re

from bleak.backends.device import BLEDevice

from homeassistant.components.bluetooth import (
    BluetoothServiceInfoBleak,
    async_ble_device_from_address,
    async_discovered_service_info,
    async_register_advertisement_callback,
)
from homeassistant.const import CONF_ADDRESS
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.event import async_call_later

from .bond import async_remove_bond
from .const import CONF_BLUETOOTH_SOURCE, DOMAIN
from .protocol.errors import ScaleConnectionError
from .protocol.models import ScaleModel
from .protocol.session import ScaleSession, open_session

_LOGGER = logging.getLogger(__name__)

# How long a flow waits for someone to step on the scale.
WAKE_TIMEOUT = 180
# Proxies report a wake's first advertisements at slightly different times.
# Connecting at the very first one leaves HA a single path to choose from,
# which in the field was a distant proxy whose link broke (#17).
PATH_SETTLE = 1.5

MAC_REGEX = re.compile(r"^([0-9A-F]{2}:){5}[0-9A-F]{2}$", re.IGNORECASE)


_LOCKS = "scale_locks"
_SESSION_ENDED = "session_ended_at"


def session_ended_signal(address: str) -> str:
    """The dispatcher signal sent when any HA session with this scale has ended.

    The scale sleeps ~1 s after HA disconnects, whoever held the session.
    """
    return f"{DOMAIN}_session_ended_{address}"


@callback
def session_ended_at(hass: HomeAssistant, address: str) -> float | None:
    """When the last HA session with the scale ended, in event loop time.

    Kept outside the coordinator, which a flow's new scale user reloads.
    """
    return _session_ends(hass).get(address)


def _session_ends(hass: HomeAssistant) -> dict[str, float]:
    ends: dict[str, float] = hass.data.setdefault(DOMAIN, {}).setdefault(
        _SESSION_ENDED, {}
    )
    return ends


class WakeTimeout(Exception):
    """Nobody woke the scale in time."""


def device_source(device: BLEDevice) -> str | None:
    """Return the source identifier of a BLEDevice, if available."""
    details = device.details
    if isinstance(details, dict) and (source := details.get("source")):
        return str(source)
    return None


def is_remote_bluetooth_source(hass: HomeAssistant, source: str | None) -> bool:
    """Whether a Bluetooth source belongs to a remote scanner (e.g. ESPHome proxy)."""
    if not source or (src := source.strip().lower()) == "local":
        return False
    try:
        from homeassistant.components.bluetooth import async_scanner_by_source

        scanner = async_scanner_by_source(hass, source)
        if scanner is not None:
            return getattr(scanner, "connector", None) is not None
    except Exception:
        pass
    # If the scanner isn't found in manager or during test mocks,
    # assume any non-local source represents a remote proxy.
    return True


def bluetooth_source_for_address(hass: HomeAssistant, address: str) -> str | None:
    """Return the scale's pinned Bluetooth source, if configured."""
    for entry in hass.config_entries.async_entries(DOMAIN):
        if entry.data.get(CONF_ADDRESS) == address:
            source = entry.options.get(CONF_BLUETOOTH_SOURCE, "")
            if isinstance(source, str) and (source := source.strip()):
                return source.upper()
            return None
    return None


def async_pin_bluetooth_source(
    hass: HomeAssistant, address: str, device: BLEDevice | str
) -> None:
    """Pin the Bluetooth source for the scale if not already set."""
    if isinstance(device, BLEDevice):
        source = device_source(device)
    else:
        source = str(device)
    if not source:
        return
    source = source.strip().upper()
    if not MAC_REGEX.match(source):
        return

    for entry in hass.config_entries.async_entries(DOMAIN):
        if entry.data.get(CONF_ADDRESS) == address:
            current = entry.options.get(CONF_BLUETOOTH_SOURCE, "")
            if not isinstance(current, str) or not current.strip():
                _LOGGER.info(
                    "Pinning scale %s to Bluetooth source %s after successful session",
                    address,
                    source,
                )
                hass.config_entries.async_update_entry(
                    entry,
                    options={**entry.options, CONF_BLUETOOTH_SOURCE: source},
                )


def async_ble_device_for_scale(hass: HomeAssistant, address: str) -> BLEDevice | None:
    """Return a connectable path for the scale, respecting its source option or preferring proxies."""
    source = bluetooth_source_for_address(hass, address)
    if source is not None:
        return next(
            (
                info.device
                for info in async_discovered_service_info(hass)
                if info.address == address
                and info.connectable
                and (info.source or "").upper() == source
            ),
            None,
        )

    # When unpinned, find all connectable candidates and prefer remote/proxy paths over local
    candidates = [
        info
        for info in async_discovered_service_info(hass)
        if info.address == address and info.connectable
    ]
    if not candidates:
        return async_ble_device_from_address(hass, address, connectable=True)

    candidates.sort(
        key=lambda info: (
            1 if is_remote_bluetooth_source(hass, info.source) else 0,
            info.rssi if info.rssi is not None else -999,
        ),
        reverse=True,
    )
    return candidates[0].device


def _scale_lock(hass: HomeAssistant, address: str) -> asyncio.Lock:
    locks: dict[str, asyncio.Lock] = hass.data.setdefault(DOMAIN, {}).setdefault(
        _LOCKS, {}
    )
    return locks.setdefault(address, asyncio.Lock())


@callback
def async_scale_held(hass: HomeAssistant, address: str) -> bool:
    """Whether a flow or the coordinator holds the scale right now."""
    return _scale_lock(hass, address).locked()


@asynccontextmanager
async def async_hold_scale(hass: HomeAssistant, address: str) -> AsyncIterator[None]:
    """Hold the scale, after whoever holds it now; nobody else connects meanwhile."""
    async with _scale_lock(hass, address):
        yield


async def async_wait_for_wake(
    hass: HomeAssistant, address: str, timeout: float = WAKE_TIMEOUT
) -> None:
    """Return at the scale's next advertisement; raise WakeTimeout after `timeout`.

    A sleeping scale doesn't advertise, so any advertisement means it is awake.
    """
    source = bluetooth_source_for_address(hass, address)
    woke: asyncio.Future[None] = hass.loop.create_future()

    @callback
    def on_advertisement(info: BluetoothServiceInfoBleak) -> None:
        if not woke.done() and (
            source is None or (info.source or "").upper() == source
        ):
            woke.set_result(None)

    @callback
    def on_timeout(_now: datetime) -> None:
        if not woke.done():
            woke.set_exception(WakeTimeout())

    cancel_advertisements = async_register_advertisement_callback(
        hass, on_advertisement, address
    )
    cancel_timer = async_call_later(hass, timeout, on_timeout)
    try:
        await woke
    finally:
        cancel_advertisements()
        cancel_timer()
    await async_let_paths_settle(hass, hass.loop.time())


async def async_let_paths_settle(hass: HomeAssistant, woke_at: float) -> None:
    """Wait until `PATH_SETTLE` after the wake, so every proxy in range has
    reported the scale and HA connects through the best one.

    `woke_at` is in event loop time.
    """
    if (delay := woke_at + PATH_SETTLE - hass.loop.time()) > 0:
        await asyncio.sleep(delay)


@asynccontextmanager
async def async_open_session(
    hass: HomeAssistant,
    address: str,
    model: ScaleModel,
    device: BLEDevice | None = None,
    session_timeout: float | None = None,
    deadline: float | None = None,
) -> AsyncIterator[ScaleSession]:
    """Connect to an awake scale through any connectable path; always disconnect.

    Pass `device` from a connectable advertisement that is being handled right
    now: HA hasn't recorded it yet. `session_timeout` replaces the default budget
    that fits the wake window, for sessions that keep the scale awake (teaching).
    `deadline` (event loop time) ends the session earlier, for work that starts
    late in a wake.

    A failed bond (a stale one after a factory reset) is cleared on the proxy
    and the connect repeated once (#23).
    """
    source = bluetooth_source_for_address(hass, address)
    if device is not None and source is not None:
        dev_source = device_source(device)
        if dev_source and dev_source.strip().upper() != source:
            device = None
    if device is None:
        device = async_ble_device_for_scale(hass, address)
    if device is None:
        raise ScaleConnectionError(f"no connectable Bluetooth path to {address}")
    opened = False
    try:
        async with open_session(
            device,
            model,
            session_timeout=session_timeout,
            deadline=deadline,
            remove_bond=partial(async_remove_bond, hass, device),
        ) as session:
            opened = True
            yield session
            # Auto-pin after a successful session if not already pinned
            async_pin_bluetooth_source(hass, address, device)
    finally:
        if opened:
            _session_ends(hass)[address] = hass.loop.time()
            async_dispatcher_send(hass, session_ended_signal(address))
