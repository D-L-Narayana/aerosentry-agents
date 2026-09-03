# Drone-in-a-Box Dock Operations Manual (v3.2)

## 1. Pre-flight readiness

A mission may only be dispatched when the dock reports state READY, the aircraft battery is at or above 40 percent, and the flight controller has a 3D GPS fix with at least 12 satellites. If any of these conditions fail the mission must remain queued and the operator is notified.

The dock performs an automatic self-check every 30 minutes: hatch actuator, charging contacts, internal temperature (operating range -20 C to 50 C), and weather sensor calibration. A failed self-check places the dock in MAINTENANCE state until cleared by a technician.

## 2. Weather limits

Launches are prohibited when sustained wind exceeds the site limit (default 10 m/s, refinery sites 8 m/s), when precipitation exceeds 0.5 mm per hour, or when visibility drops below 1.5 km. Gusts above 12 m/s trigger an automatic return-to-dock (RTL) for airborne aircraft.

## 3. Return-to-dock and abort procedures

An abort command issues RTL immediately. The aircraft climbs to the site RTL altitude (default 60 m), flies a direct path to the dock, and lands with precision-landing enabled. Operators must confirm the landing zone is clear before the hatch opens. Battery below 25 percent during flight forces RTL regardless of mission progress.

## 4. Link and telemetry

Telemetry is streamed at 1 Hz over 4G/5G with an MQTT fallback over Ethernet at the dock. Link quality below 60 percent for more than 20 seconds pauses the mission and holds position; below 30 percent triggers RTL. All telemetry is retained for 90 days for audit.

## 5. Maintenance intervals

Propellers are replaced every 200 flight hours or on visible damage. Batteries are retired after 300 charge cycles or when capacity falls below 80 percent of rated. Dock hatch seals are inspected monthly.
