# Solar Farm Thermal Inspection SOP (solar-farm-01)

## Purpose

Detect underperforming or failing photovoltaic modules using radiometric thermal imagery captured by docked drones, and route confirmed anomalies to the maintenance team within 24 hours.

## Flight parameters

Thermal surveys fly at 60 m AGL with 70 percent front overlap and 60 percent side overlap, camera at nadir, between 10:00 and 15:00 local time when irradiance exceeds 600 W/m2. Ground speed is limited to 5 m/s to avoid motion blur on the thermal sensor.

## Anomaly classes

Hotspot: a single cell more than 10 C above the module average, usually cell cracking or soldering faults. Hot string: an entire string warmer than neighbours, indicating an open string or blown fuse. Bypass diode failure: one third of a module uniformly hot. Soiling and shading produce cooler or irregular patterns and are not electrical faults.

## Severity and response

Medium severity (isolated hotspot below 15 C delta): log for the next scheduled maintenance window. High severity (multiple hotspots or hot string): raise an operator alert and schedule a confirmation flight within 48 hours. Critical severity (module delta above 40 C or visible smoke): raise a critical alert immediately, notify the site electrical supervisor, and consider isolating the inverter.

## Data handling

Radiometric JPEGs are uploaded to site storage with GPS tags. Analysts must attach a baseline image of the same array from a previous survey when reporting state-change anomalies, so that transient effects such as cloud shadow are excluded.
