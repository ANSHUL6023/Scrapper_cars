#!/usr/bin/env python3
"""
acf_mapper.py  —  Map carandbike scraped JSON → WordPress ACF payload

ACF Structure (as of 2026-06-15 export):
  cars-data post type:
    group_6a2952b35c81d  Cars Overview       → cars_gallery, launched_date,
                                               price_range, body_type, fuel_type,
                                               transmission, mileage
    group_6a29555155e67  Key Specifications   → engine_cc, max_torque, no_of_airbags,
                                               bootspace, ground_clearance,
                                               fuel_tank_capacity, seating_capacity,
                                               drive_type
    group_6a29561a24983  Latest Updates       → latest_updates_list (repeater)
    group_6a29578691443  What's New Content   → intro/exterior/interior/powertrain

  car-variant post type:
    group_6a2a3bd25d238  Specifications       → repeaters per category
    group_6a2a3bd4f1ae6  Features             → repeaters per category

FIX LOG (2026-06-15):
  Bug 1 — Repeater rows must use sub-field *names* ("spec_name", "spec_value"),
           not field keys. ACF REST API matches rows by sub-field name.

  Bug 2 — tyres_and_wheels ACF repeater in WP has lighting choices (copy-paste error
           in the field group). SPEC_TO_TYRES_CHOICE keys never matched any valid
           WP choice, causing silent drops.
           Fixed: added correct tyre choice keys to SPEC_TO_TYRES_CHOICE (you must
           also add matching choices in WP ACF admin — see comment in that dict).

  Bug 3 — Model + variant ACF payloads must use field *names* (max_torque,
           comfort_and_convenience, etc.), not field keys (field_6a…).
           ACF REST API only accepts names unless admin-mode field keys are used.

  Bug 4 — ACF field groups must have "Show in REST API" enabled in WordPress.
           Without it the REST API returns "acf": {} and silently ignores writes.
           Re-import acf-export-2026-06-15-rest-enabled.json to fix this.
"""

import argparse
import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Spec name → ACF select choice KEY used inside repeater sub-fields
# These are the "value" keys of the select fields (not field_6a… IDs).
# ---------------------------------------------------------------------------

# Engine & Performance repeater (field_6a2a3bcd9b778)
SPEC_TO_ENGINE_CHOICE: dict[str, str] = {
    "engine type":                      "engine_type",
    "displacement":                     "displacement_cc",
    "displacement (cc)":                "displacement_cc",
    "engine displacement":              "displacement_cc",
    "no. of cylinders":                 "no_of_cylinders",
    "number of cylinders":              "no_of_cylinders",
    "max power":                        "max_power",
    "power":                            "power",
    "max torque":                       "max_torque",
    "torque":                           "torque",
    "top speed":                        "top_speed",
    "acceleration":                     "acceleration_0_to_100_kmph",
    "0-100 kmph":                       "acceleration_0_to_100_kmph",
    "0 to 100":                         "acceleration_0_to_100_kmph",
    "no. of gears":                     "no_of_gears",
    "number of gears":                  "no_of_gears",
    "drive type":                       "drive_train",
    "drive train":                      "drive_train",
    "drivetrain":                       "drivetrain",
    "turbocharger":                     "turbocharger",
    "supercharger":                     "turbocharger",
    "emission standard":                "emissions_standard",
    "emissions standard":               "emissions_standard",
    "motor type":                       "motor_type",
    "electric motor":                   "motor_type",
    "motor power":                      "motor_power",
}

# Dimensions & Weight repeater (field_6a2a4e60ae7e7)
SPEC_TO_DIMENSIONS_CHOICE: dict[str, str] = {
    "length":                           "length",
    "width":                            "width",
    "height":                           "height",
    "wheelbase":                        "wheelbase",
    "ground clearance":                 "ground_clearance",
    "kerb weight":                      "kerb_weight",
}

# Capacity repeater (field_6a2a4e63ae7ea)
SPEC_TO_CAPACITY_CHOICE: dict[str, str] = {
    "boot space":                       "boot_space",
    "no. of doors":                     "no_of_doors",
    "number of doors":                  "no_of_doors",
    "no. of rows":                      "no_of_rows",
    "number of rows":                   "no_of_rows",
    "seating capacity":                 "seating_capacity",
    "fuel tank capacity":               "fuel_tank_capacity",
    "fuel tank":                        "fuel_tank_capacity",
}

# Fuel Efficiency repeater (field_6a2a4e64ae7ed)
SPEC_TO_FUEL_CHOICE: dict[str, str] = {
    "arai mileage":                     "mileage_arai",
    "mileage arai":                     "mileage_arai",
    "mileage (arai)":                   "mileage_arai",
    "mileage":                          "mileage_arai",
    "city mileage":                     "mileage_city",
    "mileage - city":                   "mileage_city",
    "mileage city":                     "mileage_city",
    "highway mileage":                  "mileage_highway",
    "mileage - highway":                "mileage_highway",
    "mileage highway":                  "mileage_highway",
}

# Suspension, Brakes & Steering repeater (field_6a2a4e6dae7f0)
SPEC_TO_SUSPENSION_CHOICE: dict[str, str] = {
    "front suspension":                 "front_suspension",
    "rear suspension":                  "rear_suspension",
    "front brake":                      "front_brake_type",
    "front brake type":                 "front_brake_type",
    "rear brake":                       "rear_brake_type",
    "rear brake type":                  "rear_brake_type",
    "steering type":                    "steering_type",
    "steering":                         "steering_type",
    "steering adjustability":           "steering_adjustability",
}

# Tyres & Wheels repeater (field_6a2a4e71ae7f3)
# BUG 2 FIX: The WP ACF export shows this repeater currently has *lighting* choices
# (copy-paste error). To make tyre data actually save, you must add the following
# choice keys in WP Admin → ACF → Specifications → Tyres and Wheels → spec_name:
#   wheel_type       → Wheel Type
#   tyre_type        → Tyre Type
#   front_tyre_size  → Front Tyre Size
#   rear_tyre_size   → Rear Tyre Size
#   spare_wheel      → Spare Wheel
#   spare_wheel_size → Spare Wheel Size
SPEC_TO_TYRES_CHOICE: dict[str, str] = {
    "wheel type":                       "wheel_type",
    "tyre type":                        "tyre_type",
    "front tyre size":                  "front_tyre_size",
    "rear tyre size":                   "rear_tyre_size",
    "spare wheel":                      "spare_wheel",
    "spare wheel type":                 "spare_wheel",
    "spare wheel size":                 "spare_wheel_size",
}

# Battery, Range & Charging repeater (field_6a2bf0f2292c4)
SPEC_TO_BATTERY_CHOICE: dict[str, str] = {
    "battery type":                     "battery_type",
    "battery capacity":                 "battery_capacity",
    "charging time":                    "charging_time",
    "charging time (ac) fast charging": "charging_time_ac_fast_charging",
    "fast charging time":               "fast_charging_time",
    "ac fast charging":                 "ac_fast_charging",
    "battery range":                    "battery_range",
    "range":                            "battery_range",
    "electric range":                   "battery_range",
    "wltp range":                       "battery_range",
    "arai range":                       "battery_range",
}

# ---------------------------------------------------------------------------
# Features category → ACF repeater field key + sub-field keys
# ---------------------------------------------------------------------------

FEATURE_REPEATERS = {
    # Comfort & Convenience
    "comfort_and_convenience": {
        "repeater_key": "field_6a2a3bcf845cf",
        "spec_name_key": "field_6a2a4fc5749ee",
        "spec_value_key": "field_6a2a4fd6749ef",
        "choices": {
            "air conditioner":                          "air_conditioner",
            "ac":                                       "air_conditioner",
            "front ac":                                 "front_ac",
            "rear ac":                                  "rear_ac",
            "third row ac":                             "third_row_ac_zone",
            "heater":                                   "heater",
            "climate control":                          "climate_control",
            "paddle shifters":                          "paddle_shift",
            "paddle shift":                             "paddle_shift",
            "idle start/stop":                          "idle_start_stop_button",
            "start/stop button":                        "idle_start_stop_button",
            "cruise control":                           "cruise_control",
            "power windows":                            "power_windows",
            "keyless entry":                            "keyless_entry",
            "push start":                               "push_start_stop_button",
            "push button start":                        "push_start_stop_button",
            "remote engine start":                      "remote_engine_start",
            "hands-free tailgate":                      "hands_free_tailgate",
            "steering mounted audio":                   "steering_mounted_audio",
            "cooled glovebox":                          "cooled_glovebox",
            "rain sensing wipers":                      "rain_sensing_wipers",
            "automatic headlamps":                      "automatic_headlamps",
            "electric tailgate":                        "electric_tailgate",
            "adjustable orvms":                         "adjustable_orvms",
            "adjustable rear headrests":                "adjustable_rear_headrests",
            "power outlets":                            "power_outlets_12v",
            "12v power outlet":                         "power_outlets_12v",
            "wireless charger":                         "wireless_charger",
            "wireless phone charger":                   "wireless_phone_charger",
            "rear ac vent":                             "rear_ac_vent",
            "sunroof":                                  "electric_sunroof",
            "heads-up display":                         "head_up_display",
            "hud":                                      "head_up_display",
        },
    },
    # Locks & Security
    "locks_and_security": {
        "repeater_key": "field_6a2a4fed749fc",
        "spec_name_key": "field_6a2a4fed749fd",
        "spec_value_key": "field_6a2a4fed749fe",
        "choices": {
            "engine immobilizer":                       "engine_immobilizer",
            "central locking":                          "central_locking",
            "speed sensing door lock":                  "speed_sensing_door_lock",
            "speed sensing auto door lock":             "speed_sensing_door_lock",
            "child safety lock":                        "child_safety_lock",
        },
    },
    # Seats & Upholstery
    "seats_and_upholstery": {
        "repeater_key": "field_6a2a4fec749f9",
        "spec_name_key": "field_6a2a4fec749fa",
        "spec_value_key": "field_6a2a4fec749fb",
        "choices": {
            "driver seat adjustment":                   "driver_seat_adjustment",
            "front passenger seat":                     "front_passenger_seat_adjustment",
            "seat upholstery":                          "seat_upholstery",
            "upholstery":                               "seat_upholstery",
            "rear armrest":                             "rear_armrest",
            "front armrest":                            "front_armrest",
            "folding rear seat":                        "folding_rear_seat",
            "split rear seat":                          "split_rear_seat",
            "seatback pockets":                         "front_seatback_pockets",
            "front seatback pockets":                   "front_seatback_pockets",
            "headrests":                                "headrests",
            "ventilated seats":                         "ventilated_seats",
            "seat memory":                              "seat_memory_function",
            "isofix":                                   "isofix_child_seat_mounts",
            "isofix child seat":                        "isofix_child_seat_mounts",
        },
    },
    # Safety & Security
    "safety_and_security": {
        "repeater_key": "field_6a2a4fea749f6",
        "spec_name_key": "field_6a2a4fea749f7",
        "spec_value_key": "field_6a2a4fea749f8",
        "choices": {
            "no. of airbags":                           "no_of_airbags",
            "number of airbags":                        "no_of_airbags",
            "airbags":                                  "no_of_airbags",
            "driver airbag":                            "driver_airbags",
            "driver airbags":                           "driver_airbags",
            "passenger airbag":                         "passenger_airbags",
            "passenger airbags":                        "passenger_airbags",
            "side airbag":                              "side_airbags",
            "side airbags":                             "side_airbags",
            "rear side airbag":                         "rear_side_airbag",
            "curtain airbag":                           "curtain_airbag",
            "curtain airbags":                          "curtain_airbag",
            "knee airbag":                              "knee_airbags",
            "knee airbags":                             "knee_airbags",
            "abs":                                      "abs",
            "anti-lock braking system":                 "abs",
            "ebd":                                      "electronic_brakeforce_distribution",
            "electronic brakeforce distribution":       "electronic_brakeforce_distribution",
            "esp":                                      "electronic_stability_program",
            "electronic stability program":             "electronic_stability_program",
            "electronic stability control":             "electronic_stability_program",
            "day & night rear view mirror":             "day_night_rear_view_mirror",
            "day and night rear view mirror":           "day_night_rear_view_mirror",
            "hill hold control":                        "hill_hold_control",
            "hill assist":                              "hill_assist",
            "hill descent control":                     "hill_descent_control",
            "tyre pressure monitoring":                 "tyre_pressure_monitoring_system_tpms",
            "tpms":                                     "tyre_pressure_monitoring_system_tpms",
            "traction control":                         "traction_control",
            "blind spot monitoring":                    "blind_spot_monitoring",
            "bsm":                                      "blind_spot_monitoring",
            "lane departure warning":                   "lane_assist",
            "lane assist":                              "lane_assist",
            "forward collision warning":                "forward_collision_warning",
            "auto emergency braking":                   "auto_emergency_braking",
            "autonomous emergency braking":             "auto_emergency_braking",
            "driver drowsiness detection":              "driver_drowsiness_detection",
            "child seat anchor points":                 "child_seat_anchor_points",
            "parking sensors":                          "parking_sensors",
            "speed alert":                              "speed_alert_system",
            "speed alert system":                       "speed_alert_system",
            "parking assistance":                       "parking_assistance",
            "dashcam":                                  "dashcam",
            "electric orvm heating":                    "electric_orvm_heating",
            "360 degree camera":                        "360_view_camera",
            "360 view camera":                          "360_view_camera",
            "rear defogger":                            "rear_defogger",
        },
    },
    # Lights
    "lights": {
        "repeater_key": "field_6a2a4fe4749f3",
        "spec_name_key": "field_6a2a4fe4749f4",
        "spec_value_key": "field_6a2a4fe4749f5",
        "choices": {
            "light type":                               "light_type",
            "headlight type":                           "light_type",
            "drls":                                     "daytime_running_lamps_drls",
            "daytime running lamps":                    "daytime_running_lamps_drls",
            "daytime running lamps (drls)":             "daytime_running_lamps_drls",
            "turn indicators on orvm":                  "turn_indicators_on_orvm",
            "fog lamps":                                "fog_lamps",
            "ambient lighting":                         "ambient_lighting",
            "auto high beam":                           "automatic_high_beam_assist",
            "automatic high beam assist":               "automatic_high_beam_assist",
            "follow me home headlamps":                 "follow_me_home_headlamps",
            "follow-me-home headlamps":                 "follow_me_home_headlamps",
            "cornering lights":                         "cornering_lights",
            "cabin lamps":                              "cabin_lamps",
            "automatic headlamps":                      "automatic_headlamps",
        },
    },
    # Instrumentation
    "instrumentation": {
        "repeater_key": "field_6a2a4fe3749f0",
        "spec_name_key": "field_6a2a4fe3749f1",
        "spec_value_key": "field_6a2a4fe3749f2",
        "choices": {
            "instrument cluster":                       "instrument_cluster",
            "trip meter":                               "trip_meter",
            "average fuel efficiency":                  "average_fuel_efficiency_indicator",
            "average speed":                            "average_speed",
            "clock":                                    "clock",
            "low fuel warning":                         "low_fuel_warning",
            "door ajar warning":                        "door_ajar_warning",
            "adjustable cluster brightness":            "adjustable_cluster_brightness",
            "gear indicator":                           "gear_indicator",
            "shift indicator":                          "shift_indicator",
            "tachometer":                               "tachometer",
            "seat belt warning":                        "seat_belt_warning",
        },
    },
    # Entertainment
    "entertainment": {
        "repeater_key": "field_6a2a4ff6749ff",
        "spec_name_key": "field_6a2a4ff674a00",
        "spec_value_key": "field_6a2a4ff674a01",
        "choices": {
            "infotainment system":                      "infotainment_system",
            "touchscreen size":                         "touchscreen_display_size",
            "touchscreen display":                      "touchscreen_display_size",
            "audio system":                             "audio_system",
            "speakers":                                 "speakers",
            "tweeters":                                 "tweeters",
            "subwoofer":                                "subwoofer",
            "gps navigation":                           "gps_navigation",
            "navigation":                               "gps_navigation",
            "voice commands":                           "voice_commands",
            "smartphone connectivity":                  "smartphone_connectivity",
            "android auto":                             "smartphone_connectivity",
            "apple carplay":                            "smartphone_connectivity",
            "bluetooth":                                "bluetooth_support",
            "usb":                                      "usb_support",
            "aux":                                      "aux_compatibility",
            "aux input":                                "aux_compatibility",
            "wireless display":                         "wireless_display_screen_mirroring",
            "screen mirroring":                         "wireless_display_screen_mirroring",
            "heads-up display":                         "heads_up_display_hud",
            "hud":                                      "heads_up_display_hud",
            "radio":                                    "radio",
        },
    },
    # Advanced Features
    "advanced_features": {
        "repeater_key": "field_6a2aa821472a1",
        "spec_name_key": "field_6a2aa821472a2",
        "spec_value_key": "field_6a2aa821472a3",
        "choices": {
            "connected car technology":                 "connected_car_technology",
            "ota updates":                              "over_the_air_updates_ota",
            "over-the-air updates":                     "over_the_air_updates_ota",
            "geofencing":                               "geofencing",
            "digital key":                              "digital_key",
            "ai voice assistance":                      "ai_voice_assistance",
            "auto park assist":                         "auto_park_assist",
            "autonomous driving level":                 "autonomous_driving_level",
            "sos button":                               "sos_button",
        },
    },
    # Manufacturer Warranty
    "manufacturer_warranty": {
        "repeater_key": "field_6a2aa826472a4",
        "spec_name_key": "field_6a2aa826472a5",
        "spec_value_key": "field_6a2aa826472a6",
        "choices": {
            "battery warranty":                         "type_battery_warranty",
            "type battery warranty":                    "type_battery_warranty",
            "manufacturer warranty":                    "type_manufacturer_warranty",
            "type manufacturer warranty":               "type_manufacturer_warranty",
            "warranty":                                 "type_manufacturer_warranty",
        },
    },
}

# Specifications repeater field keys per category
SPEC_REPEATERS = {
    "engine_and_performance": {
        "repeater_key":   "field_6a2a3bcd9b778",
        "spec_name_key":  "field_6a2a4d9410d13",
        "spec_value_key": "field_6a2a4d9f10d14",
        "choices":        SPEC_TO_ENGINE_CHOICE,
    },
    "dimensions_and_weight": {
        "repeater_key":   "field_6a2a4e60ae7e7",
        "spec_name_key":  "field_6a2a4e60ae7e8",
        "spec_value_key": "field_6a2a4e60ae7e9",
        "choices":        SPEC_TO_DIMENSIONS_CHOICE,
    },
    "capacity": {
        "repeater_key":   "field_6a2a4e63ae7ea",
        "spec_name_key":  "field_6a2a4e63ae7eb",
        "spec_value_key": "field_6a2a4e63ae7ec",
        "choices":        SPEC_TO_CAPACITY_CHOICE,
    },
    "fuel_efficiency": {
        "repeater_key":   "field_6a2a4e64ae7ed",
        "spec_name_key":  "field_6a2a4e64ae7ee",
        "spec_value_key": "field_6a2a4e64ae7ef",
        "choices":        SPEC_TO_FUEL_CHOICE,
    },
    "suspension_brakes_and_steering": {
        "repeater_key":   "field_6a2a4e6dae7f0",
        "spec_name_key":  "field_6a2a4e6dae7f1",
        "spec_value_key": "field_6a2a4e6dae7f2",
        "choices":        SPEC_TO_SUSPENSION_CHOICE,
    },
    "tyres_and_wheels": {
        "repeater_key":   "field_6a2a4e71ae7f3",
        "spec_name_key":  "field_6a2a4e71ae7f4",
        "spec_value_key": "field_6a2a4e71ae7f5",
        "choices":        SPEC_TO_TYRES_CHOICE,
    },
    "battery_range_and_charging": {
        "repeater_key":   "field_6a2bf0f2292c4",
        "spec_name_key":  "field_6a2bf0f2292c5",
        "spec_value_key": "field_6a2bf0f2292c6",
        "choices":        SPEC_TO_BATTERY_CHOICE,
    },
}

# carandbike.com category labels -> ACF repeater field name
CATEGORY_TO_GROUP: dict[str, str] = {
    "engine & performance":              "engine_and_performance",
    "engine and performance":            "engine_and_performance",
    "dimensions & weight":               "dimensions_and_weight",
    "dimensions and weight":             "dimensions_and_weight",
    "capacity":                          "capacity",
    "fuel efficiency":                   "fuel_efficiency",
    "suspension brakes & steering":      "suspension_brakes_and_steering",
    "suspension, brakes & steering":     "suspension_brakes_and_steering",
    "suspension, brakes and steering":   "suspension_brakes_and_steering",
    "tyres & wheels":                    "tyres_and_wheels",
    "tyres and wheels":                  "tyres_and_wheels",
    "battery, range & charging":         "battery_range_and_charging",
    "battery range and charging":        "battery_range_and_charging",
    "comfort & convenience":             "comfort_and_convenience",
    "comfort and convenience":           "comfort_and_convenience",
    "locks & security":                  "locks_and_security",
    "locks and security":                "locks_and_security",
    "seats & upholstery":                "seats_and_upholstery",
    "seats and upholstery":              "seats_and_upholstery",
    "safety & security":                 "safety_and_security",
    "safety and security":               "safety_and_security",
    "lights":                            "lights",
    "instrumentation":                   "instrumentation",
    "entertainment":                     "entertainment",
    "advanced features":                 "advanced_features",
    "manufacturer warranty":             "manufacturer_warranty",
}

ALL_REPEATERS: dict = {**SPEC_REPEATERS, **FEATURE_REPEATERS}

# Cars Overview (cars-data model post) field keys
MODEL_OVERVIEW_FIELDS = {
    "cars_gallery":   "field_6a2f7d87e8034",
    "launched_date":  "field_6a295294cc3d4",
    "price_range":    "field_6a295419cc3d5",
    "body_type":      "field_6a29543ccc3d6",
    "fuel_type":      "field_6a29544bcc3d7",
    "transmission":   "field_6a29545bcc3d8",
    "mileage":        "field_6a295465cc3d9",
}

# Key Specifications (cars-data model post) field keys
KEY_SPEC_FIELDS = {
    "engine_cc":          "field_6a295532bb528",
    "max_torque":         "field_6a295552bb529",
    "no_of_airbags":      "field_6a295561bb52a",
    "bootspace":          "field_6a295566bb52b",
    "ground_clearance":   "field_6a295571bb52c",
    "fuel_tank_capacity": "field_6a29557bbb52d",
    "seating_capacity":   "field_6a29558bbb52e",
    "drive_type":         "field_6a295594bb52f",
}

# Latest Updates repeater (cars-data) field keys
LATEST_UPDATES_REPEATER = "field_6a2955fa1e584"
LATEST_UPDATES_HEADLINE  = "field_6a2956691e585"
LATEST_UPDATES_DATE      = "field_6a2956911e586"

# What's New Content (cars-data) field keys
WHATS_NEW_FIELDS = {
    "introduction_list":  "field_6a2957676bf68",
    "introduction_item":  "field_6a2957a06bf69",
    "introduction_image": "field_6a2958436bf6a",
    "exterior_list":      "field_6a2958996bf6b",
    "exterior_item":      "field_6a2958a96bf6c",
    "exterior_image":     "field_6a2a3903b645f",
    "interior_list":      "field_6a2a392ab6460",
    "interior_item":      "field_6a2a393eb6461",
    "interior_image":     "field_6a2a394fb6462",
    "powertrain_list":    "field_6a2a3978b6463",
    "powertrain_item":    "field_6a2a3985b6464",
    "powertrain_image":   "field_6a2a3993b6465",
}

# Parent Base Model (car-variant post) — links variant to its cars-data parent
PARENT_BASE_MODEL_FIELD = "parent_base_model"

WHATS_NEW_REPEATER_MAP = {
    "introduction": ("introduction_list", "introduction_item"),
    "exterior":     ("exterior_list",     "exterior_item"),
    "interior":     ("interior_list",     "interior_item"),
    "powertrain":   ("powertrain_list",   "powertrain_item"),
}


def _normalise_key(name: str) -> str:
    """Lowercase + collapse whitespace for map lookup."""
    return re.sub(r"\s+", " ", name.strip().lower())


def _normalise_category(category: str) -> str:
    """Normalise scraper category labels for CATEGORY_TO_GROUP lookup."""
    cat = str(category or "").strip().lower()
    cat = cat.replace("&", " and ")
    return re.sub(r"\s+", " ", cat)


def _lookup_choice(norm: str, choices: dict[str, str]) -> str | None:
    """Match a scraped spec name to an ACF select choice key."""
    if norm in choices:
        return choices[norm]

    variants = [
        re.sub(r"\s*\([^)]*\)", "", norm).strip(),
        norm.replace("-", " "),
        re.sub(r"\s*\([^)]*\)", "", norm.replace("-", " ")).strip(),
    ]
    for candidate in variants:
        if candidate and candidate in choices:
            return choices[candidate]

    return None


def _to_wp_slug(text: str) -> str:
    """Convert 'Maruti Suzuki Ertiga ZXi+' → 'maruti-suzuki-ertiga-zxi'."""
    s = text.lower().strip()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("-")


def _yes_no(value: str) -> str:
    """Normalise boolean-ish spec values to 'Yes' / 'No' / original."""
    low = str(value).strip().lower()
    if low in ("yes", "available", "standard", "true", "1"):
        return "Yes"
    if low in ("no", "not available", "na", "n/a", "false", "0", "-"):
        return "No"
    return value.strip()


def _parse_mileage_number(text: str) -> float | None:
    """Extract leading mileage number from strings like '19.01 KM/L' or '26.2'."""
    m = re.search(r"(\d+(?:\.\d+)?)", str(text or ""))
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _extract_mileage(model: dict, specs: list, features: list, variants: list) -> str:
    """
    Resolve Cars Overview mileage for the model post.

    carandbike.com uses names like 'Mileage ARAI' (not 'ARAI Mileage').
    Fall back to variant mileage values when specs lack a match.
    """
    mileage_names = {
        "arai mileage", "mileage arai", "mileage (arai)", "mileage - arai", "mileage",
    }
    for item in list(specs or []) + list(features or []):
        norm = _normalise_key(item.get("name", ""))
        if norm in mileage_names and item.get("value"):
            return str(item["value"]).strip()

    variant_values = [
        str(v.get("mileage", "")).strip()
        for v in (variants or [])
        if v.get("mileage")
    ]
    if variant_values:
        nums = [_parse_mileage_number(v) for v in variant_values]
        nums = [n for n in nums if n is not None]
        if nums:
            lo, hi = min(nums), max(nums)
            if lo == hi:
                return variant_values[0]
            return f"{lo:g} - {hi:g} KM/L"

    summary = model.get("summary", "") or ""
    m = re.search(
        r"mileage[^.]*?(?:varies from|is|of)\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*km/l",
        summary,
        re.I,
    )
    if m:
        return f"{m.group(1)} - {m.group(2)} KM/L"
    m = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*km/l", summary, re.I)
    if m:
        return f"{m.group(1)} - {m.group(2)} KM/L"

    return ""


def _airbag_row_weight(norm_name: str) -> int:
    """How many physical airbags one Yes row represents."""
    if "curtain" in norm_name:
        return 2
    if "side" in norm_name and "airbag" in norm_name:
        return 2
    return 1


def _extract_airbag_count(specs: list, features: list) -> str:
    """
    Resolve Key Specifications no_of_airbags.

    carandbike.com lists individual airbag types (Driver, Side, Curtain, etc.)
    rather than a total count — sum them when no explicit total is present.
    """
    all_items = list(specs or []) + list(features or [])

    for item in all_items:
        norm = _normalise_key(item.get("name", ""))
        if norm in {"no. of airbags", "number of airbags", "airbags", "total airbags"}:
            val = str(item.get("value", "")).strip()
            if val:
                num = re.search(r"\d+", val)
                return num.group(0) if num else val

    total = 0
    for item in all_items:
        norm = _normalise_key(item.get("name", ""))
        if "airbag" not in norm:
            continue
        val = str(item.get("value", "")).strip()
        low = val.lower()
        if low in ("no", "not available", "na", "n/a", "false", "0", "-"):
            continue
        num = re.search(r"^\d+$", val)
        if num:
            total += int(val.group(0))
        elif low in ("yes", "available", "standard", "true", "1"):
            total += _airbag_row_weight(norm)

    return str(total) if total > 0 else ""


def _build_spec_repeaters(specs: list, features: list) -> dict:
    """
    Walk specs + features and map each recognised item into the correct
    ACF repeater group.  Returns {repeater_field_key: [row, ...], …}

    Each row uses ACF sub-field names ("spec_name", "spec_value") so the REST API
    can map values into repeater rows.

    The scraper produces items like:
      {"category": "Engine & Performance", "name": "Max Power", "value": "172 bhp"}
    We try to match by spec name; category is used only as a tiebreak.
    """
    repeater_data: dict[str, list] = {}

    def _add_row(group_name: str, rconf: dict, choice_key: str, value: str):
        """
        Append one row to the repeater using ACF sub-field names.
        Deduplicates on spec_name choice key to avoid double entries.
        """
        if group_name not in repeater_data:
            repeater_data[group_name] = []
        existing_choices = {r.get("spec_name") for r in repeater_data[group_name]}
        if choice_key not in existing_choices:
            preserve = group_name in {
                "fuel_efficiency", "manufacturer_warranty", "lights",
                "engine_and_performance", "dimensions_and_weight", "capacity",
                "tyres_and_wheels", "instrumentation", "entertainment",
            }
            spec_value = str(value).strip() if preserve else _yes_no(value)
            repeater_data[group_name].append({
                "spec_name":  choice_key,
                "spec_value": spec_value,
            })

    def _try_match(group_name: str, norm: str, raw_val: str) -> bool:
        rconf = ALL_REPEATERS.get(group_name)
        if not rconf:
            return False
        choice_key = _lookup_choice(norm, rconf["choices"])
        if choice_key:
            _add_row(group_name, rconf, choice_key, raw_val)
            return True
        return False

    all_items = list(specs or []) + list(features or [])

    for item in all_items:
        if not isinstance(item, dict):
            continue
        raw_name = item.get("name", "")
        raw_val  = item.get("value", "")
        if not raw_name or not raw_val:
            continue
        norm = _normalise_key(raw_name)
        cat  = _normalise_category(item.get("category", ""))

        # Prefer the repeater that matches the scraper category (avoids mis-routing
        # e.g. Automatic Headlamps into Comfort instead of Lights).
        if cat and cat in CATEGORY_TO_GROUP:
            if _try_match(CATEGORY_TO_GROUP[cat], norm, raw_val):
                continue

        matched = False
        for group_name, rconf in SPEC_REPEATERS.items():
            if _try_match(group_name, norm, raw_val):
                matched = True
                break

        if matched:
            continue

        for group_name, rconf in FEATURE_REPEATERS.items():
            if _try_match(group_name, norm, raw_val):
                break

    return repeater_data


def _extract_key_specs(specs: list, features: list) -> dict:
    """
    Extract Key Specifications fields for the cars-data model post.
    Returns {field_name: value} — see acf_key_specs_keyed for field-key version.
    """
    all_items = list(specs or []) + list(features or [])
    key_map = {
        "engine_cc":          {"engine displacement", "displacement", "displacement (cc)"},
        "max_torque":         {"max torque", "torque"},
        "no_of_airbags":      {"no. of airbags", "number of airbags", "airbags"},
        "bootspace":          {"boot space"},
        "ground_clearance":   {"ground clearance"},
        "fuel_tank_capacity": {"fuel tank capacity", "fuel tank"},
        "seating_capacity":   {"seating capacity"},
        "drive_type":         {"drive type", "drive train", "drivetrain"},
    }
    result = {}
    for spec_field, names in key_map.items():
        for item in all_items:
            norm = _normalise_key(item.get("name", ""))
            if norm in names and item.get("value"):
                result[spec_field] = item["value"]
                break

    if not result.get("no_of_airbags"):
        airbag_count = _extract_airbag_count(specs, features)
        if airbag_count:
            result["no_of_airbags"] = airbag_count

    return result


def _extract_overview(model: dict) -> dict:
    """Build Cars Overview ACF fields for the cars-data post (by field name)."""
    price_min = model.get("price_min_inr", "")
    price_max = model.get("price_max_inr", "")
    price_range = ""
    if price_min and price_max:
        try:
            lo = f"₹{int(price_min)/100_000:.2f}L"
            hi = f"₹{int(price_max)/100_000:.2f}L"
            price_range = f"{lo} - {hi}"
        except (ValueError, TypeError):
            price_range = f"{price_min} - {price_max}"
    elif price_min:
        try:
            price_range = f"₹{int(price_min)/100_000:.2f}L"
        except (ValueError, TypeError):
            price_range = str(price_min)

    return {
        "launched_date":  model.get("launch_date", ""),
        "price_range":    price_range,
        "body_type":      model.get("body_type", ""),
        "fuel_type":      model.get("fuel_types", ""),
        "transmission":   model.get("transmission_types", ""),
        "mileage":        "",   # filled from variant specs below
    }


def _build_whats_new_acf(model: dict) -> dict:
    """
    Map scraped What's New bullet lists → ACF repeater fields on cars-data.
    Only Introduction / Exterior / Interior / Powertrain are mapped.
    """
    whats = model.get("whats_new") or {}
    acf: dict = {}

    for section_key, (repeater_name, item_name) in WHATS_NEW_REPEATER_MAP.items():
        items = whats.get(section_key) or []
        rows = [{item_name: text} for text in items if text]
        if rows:
            acf[repeater_name] = rows

    return acf


def _keyed_acf_fields(field_name_map: dict, values: dict) -> dict:
    """
    BUG 3 FIX: Convert {field_name: value} → {field_key: value} so that
    wp_uploader can POST directly into the ACF REST API without any further
    translation.  Unknown field names are silently skipped.
    """
    return {
        field_name_map[name]: val
        for name, val in values.items()
        if name in field_name_map and val not in (None, "", [], {})
    }


def map_model_to_acf(model: dict, first_variant_only: bool = False) -> dict:
    """
    Convert one scraped model dict → WordPress ACF payload.

    Model post (cars-data):
      - Cars Overview fields (overview text fields)
      - Key Specifications fields (quick-access plain fields)
      - gallery_images list (wp_uploader resolves to IDs)

    Variant posts (car-variant):
      - Specifications repeaters (engine, dimensions, capacity, etc.)
      - Features repeaters (comfort, safety, lights, etc.)
      - Basic variant info (name, price, fuel_type, transmission, mileage, engine_cc)

    Returns REST-ready payloads keyed by ACF field *names*:
      acf_model_payload    — {field_name: value} for cars-data posts
      acf_variant_payload  — {repeater_name: [{spec_name, spec_value}, …]} per variant
    """
    brand    = model.get("brand", "")
    mdl_name = model.get("model", "")
    post_title = f"{brand} {mdl_name}".strip()

    variants = model.get("variants", [])
    if first_variant_only and variants:
        variants = variants[:1]

    # ---- Model-level specs (used for Key Specs + overview mileage) --------
    model_specs = model.get("specifications", [])
    model_feats = model.get("features", [])
    if variants:
        first_v = variants[0]
        model_specs = first_v.get("specifications") or model_specs
        model_feats = first_v.get("features")       or model_feats

    model_overview = _extract_overview(model)
    key_specs_acf  = _extract_key_specs(model_specs, model_feats)
    model_overview["mileage"] = _extract_mileage(model, model_specs, model_feats, variants)
    whats_new_acf = _build_whats_new_acf(model)

    # REST payloads use ACF field names (not field_6a… keys)
    acf_overview_rest = {
        name: val for name, val in model_overview.items()
        if val not in (None, "", [], {})
    }
    acf_key_specs_rest = {
        name: val for name, val in key_specs_acf.items()
        if val not in (None, "", [], {})
    }

    intro_para = (
        model.get("intro_para")
        or model.get("summary")
        or ""
    ).strip()

    # ---- Per-variant payloads --------------------------------------------
    mapped_variants = []
    for v in variants:
        v_specs = v.get("specifications") or model.get("specifications", [])
        v_feats = v.get("features")       or model.get("features", [])

        # BUG 1 FIX: _build_spec_repeaters now returns rows keyed by field keys
        v_repeaters = _build_spec_repeaters(v_specs, v_feats)

        # Variant price display
        price_inr     = v.get("price_inr", v.get("price", ""))
        price_display = v.get("price_display", "")
        if not price_display and price_inr:
            try:
                price_display = f"₹ {int(price_inr) / 100_000:.2f} Lakh"
            except (ValueError, TypeError):
                price_display = str(price_inr)

        # REST payload: {repeater_field_name: [{spec_name, spec_value}, …]}
        acf_variant_payload: dict = {
            group_name: rows
            for group_name, rows in v_repeaters.items()
            if rows
        }

        mapped_variants.append({
            "variant_name":        v.get("variant_name", ""),
            "variant_id":          v.get("variant_id", ""),
            "price_inr":           price_inr,
            "price_display":       price_display,
            "fuel_type":           v.get("fuel_type", ""),
            "transmission":        v.get("transmission", ""),
            "mileage":             v.get("mileage", ""),
            "engine_cc":           v.get("engine_cc", ""),
            "spec_summary":        v.get("spec_summary", ""),
            "parent_model_name":   mdl_name,
            # acf_repeaters: human-readable {group_name: [rows]} for debugging
            "acf_repeaters":       v_repeaters,
            # acf_variant_payload: {repeater_field_key: [rows]} → POST directly
            "acf_variant_payload": acf_variant_payload,
        })

    return {
        "post_title":           post_title,
        "post_type":            "cars-data",
        "slug":                 _to_wp_slug(post_title),
        "post_content":         intro_para,
        "source_url":           model.get("url", ""),
        "brand":                brand,
        "model":                mdl_name,
        "model_id":             model.get("model_id", ""),
        "body_type":            model.get("body_type", ""),
        "price_min_inr":        model.get("price_min_inr", ""),
        "price_max_inr":        model.get("price_max_inr", ""),
        "price_display":        model.get("price_min_display", ""),
        "rating":               model.get("rating", ""),
        "review_count":         model.get("review_count", ""),
        "fuel_types":           model.get("fuel_types", ""),
        "colours":              model.get("colours", []),
        "featured_image_url":   model.get("featured_image_url", ""),
        "whats_new":            model.get("whats_new", {}),
        # gallery_images: list of LOCAL webp paths or original URLs
        "gallery_images":       model.get("gallery_local_paths", model.get("gallery_images", [])),
        # ACF for model post — by field name (REST-ready)
        "acf_overview":         model_overview,
        "acf_key_specs":        key_specs_acf,
        "acf_whats_new":        whats_new_acf,
        "acf_overview_rest":    acf_overview_rest,
        "acf_key_specs_rest":   acf_key_specs_rest,
        # Combined model-level ACF payload (overview + key specs + what's new)
        "acf_model_payload":    {**acf_overview_rest, **acf_key_specs_rest, **whats_new_acf},
        # Variants
        "variants":             mapped_variants,
    }


def map_all_models(scraped_data: list, first_variant_only: bool = False) -> list:
    """Map a full scraped JSON list (all models) to ACF payloads."""
    return [map_model_to_acf(m, first_variant_only) for m in scraped_data]


def main():
    parser = argparse.ArgumentParser(
        description="Map carandbike scraped JSON → WordPress ACF payload",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input_json",  help="Scraped JSON file from scrapper.py")
    parser.add_argument("--output",    default="", help="Output ACF JSON file (default: <stem>_acf.json)")
    parser.add_argument("--first-variant-only", action="store_true",
                        help="Only map the first variant per car (faster, smaller file)")
    args = parser.parse_args()

    in_path = Path(args.input_json)
    if not in_path.exists():
        print(f"ERROR: Input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading scraped data from: {in_path}")
    with open(in_path, encoding="utf-8") as f:
        scraped = json.load(f)

    if isinstance(scraped, dict):
        scraped = [scraped]

    print(f"Mapping {len(scraped)} car model(s) to ACF fields…")
    acf_data = map_all_models(scraped, first_variant_only=args.first_variant_only)

    out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_acf.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(acf_data, f, ensure_ascii=False, indent=2)

    print(f"ACF payload saved → {out_path.resolve()}")
    print(f"  Models mapped   : {len(acf_data)}")
    total_variants = sum(len(m["variants"]) for m in acf_data)
    print(f"  Total variants  : {total_variants}")

    # Diagnostic: report how many repeater rows were produced
    total_rows = 0
    for m in acf_data:
        for v in m["variants"]:
            for rows in v["acf_repeaters"].values():
                total_rows += len(rows)
    print(f"  Total repeater rows mapped: {total_rows}")
    if total_rows == 0:
        print("  WARNING: 0 repeater rows — check that scraped specs/features are not empty.")


if __name__ == "__main__":
    main()