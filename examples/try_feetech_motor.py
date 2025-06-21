#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import time

from lerobot.common.motors import Motor, MotorNormMode
from lerobot.common.motors.feetech import FeetechMotorsBus
from lerobot.common.motors.feetech.tables import MODEL_RESOLUTION, SCAN_BAUDRATES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def scan_motors(port: str, baudrate: int | None = None) -> tuple[dict[int, int] | None, int | None]:
    """Scans for motors on a given port and returns found motors and baudrate."""
    logger.info(f"Scanning for motors on port {port}...")
    # We need to provide a dummy motor to instantiate the bus
    bus = FeetechMotorsBus(port, motors={"dummy": Motor(253, "sts3215", MotorNormMode.RANGE_0_100)})

    if not bus.port_handler.openPort():
        logger.error(f"Failed to open port {port}")
        return None, None

    scan_baudrates = [baudrate] if baudrate is not None else SCAN_BAUDRATES
    for baudrate_to_scan in scan_baudrates:
        logger.info(f"Scanning at baudrate {baudrate_to_scan}...")
        bus.port_handler.setBaudRate(baudrate_to_scan)
        found_motors = bus.broadcast_ping()
        if found_motors:
            logger.info(f"Found motors at baudrate {baudrate_to_scan}: {found_motors}")
            bus.port_handler.closePort()
            return found_motors, baudrate_to_scan

    logger.info("No motors found.")
    bus.port_handler.closePort()
    return None, None


def test_motor_movement(port: str, motor_id: int, motor_model_number: int, baudrate: int):
    """Moves a specific motor to test positions."""
    # Find motor model string from model number
    motor_model = next(
        (model for model, number in FeetechMotorsBus.model_number_table.items() if number == motor_model_number),
        None,
    )
    if not motor_model:
        logger.error(f"Unknown motor model number: {motor_model_number}")
        return

    logger.info(f"Testing movement for motor {motor_id} ({motor_model}).")

    # We need to provide the actual motor to instantiate the bus
    bus = FeetechMotorsBus(
        port, motors={"my_motor": Motor(motor_id, motor_model, MotorNormMode.RANGE_0_100)}
    )
    bus.port_handler.setBaudRate(baudrate)

    try:
        if not bus.port_handler.openPort():
            logger.error(f"Failed to open port {port}")
            return

        # Enable torque
        bus.enable_torque("my_motor")
        time.sleep(0.1)

        # --- Test movement to 1800 ---
        start_pos = bus.read("Present_Position", "my_motor", normalize=False)
        logger.info(f"Starting position: {start_pos}")

        logger.info("Sending goal: 1800")
        bus.write("Goal_Position", "my_motor", 1800, normalize=False)
        time.sleep(1.5)

        end_pos = bus.read("Present_Position", "my_motor", normalize=False)
        logger.info(f"Position after moving to 1800: {end_pos}")

        # --- Test movement to 2047 ---
        logger.info("Sending goal: 2047")
        bus.write("Goal_Position", "my_motor", 2047, normalize=False)
        time.sleep(1.5)  # Wait for the motor to move

        end_pos_2 = bus.read("Present_Position", "my_motor", normalize=False)
        logger.info(f"Position after moving to 2047: {end_pos_2}")

        # Disable torque
        try:
            bus.disable_torque("my_motor")
            logger.info("Motor movement test finished and torque disabled.")
        except ConnectionError as e:
            logger.warning(f"Could not disable torque, continuing anyway: {e}")

    finally:
        if bus.port_handler.is_open:
            bus.port_handler.closePort()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan for Feetech motors and move one to its midpoint.")
    parser.add_argument("port", type=str, help="The serial port to scan (e.g., /dev/ttyUSB0)")
    parser.add_argument(
        "--baudrate",
        type=int,
        default=1_000_000,
        help="The baudrate to use. If not provided, will scan all.",
    )
    args = parser.parse_args()

    found_motors, baudrate = scan_motors(args.port, args.baudrate)

    if found_motors:
        if len(found_motors) == 1:
            motor_id = list(found_motors.keys())[0]
            logger.info(f"Only one motor found (ID: {motor_id}). Testing it.")
        else:
            try:
                motor_id_str = input(f"Enter the ID of the motor to move from {list(found_motors.keys())}: ")
                motor_id = int(motor_id_str)
                if motor_id not in found_motors:
                    logger.error(f"Motor ID {motor_id} not in the list of found motors.")
                    exit()
            except ValueError:
                logger.error("Invalid motor ID.")
                exit()

        motor_model_number = found_motors[motor_id]
        test_motor_movement(args.port, motor_id, motor_model_number, baudrate) 