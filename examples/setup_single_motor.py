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
from lerobot.common.motors.feetech.tables import SCAN_BAUDRATES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_single_motor(port: str, motor_model: str, new_id: int, new_baudrate: int, protocol_version: int, baudrate: int = None):
    """Finds a single motor, unlocks it, and sets its ID and baudrate."""
    # Step 1: Find the motor by scanning all baudrates
    logger.info(
        f"Attempting to find a single connected motor using protocol {protocol_version} by scanning all baudrates..."
    )
    bus = FeetechMotorsBus(
        port,
        motors={"dummy": Motor(253, motor_model, MotorNormMode.RANGE_0_100)},
        protocol_version=protocol_version,
    )

    found_motor_info = None
    if not bus.port_handler.openPort():
        logger.error(f"Failed to open port {port}")
        return

    scan_baudrates = [baudrate] if baudrate is not None else SCAN_BAUDRATES
    for baudrate_to_scan in scan_baudrates:
        bus.port_handler.setBaudRate(baudrate_to_scan)
        if protocol_version == 0:
            found_motors = bus.broadcast_ping()  # Returns {id: model_number}
        else:
            # Protocol 1 doesn't support broadcast, so we ping IDs individually
            found_motors = {}
            for i in range(254):  # Max ID for Feetech
                model_nb = bus.ping(i)
                if model_nb:
                    # bus.ping returns model number on success, None on fail
                    found_motors[i] = model_nb
                    break  # Found one, stop searching
                time.sleep(0.001)  # Small delay to not overwhelm the bus
        if found_motors:
            if len(found_motors) > 1:
                logger.error(f"Found more than one motor: {found_motors}. Please only connect one motor.")
                bus.port_handler.closePort()
                return
            found_id = list(found_motors.keys())[0]
            found_motor_info = {"id": found_id, "baudrate": baudrate_to_scan}
            logger.info(f"Found motor with ID {found_id} at baudrate {baudrate_to_scan}.")
            break

    if not found_motor_info:
        logger.error("Could not find any motor. Please check connection and power.")
        bus.port_handler.closePort()
        return

    # Step 2: Re-init bus with the found motor and configure it
    motor_name = "motor_to_configure"
    motors = {
        motor_name: Motor(
            id=found_motor_info["id"], model=motor_model, norm_mode=MotorNormMode.RANGE_0_100
        )
    }
    # We keep the same port handler, which is already open and at the correct (found) baudrate
    bus = FeetechMotorsBus(port, motors=motors, _port_handler=bus.port_handler)

    try:
        # Step 3: Unlock EEPROM for writing
        logger.info(f"Unlocking motor {found_motor_info['id']} for writing...")
        bus.write("Lock", motor_name, 0, normalize=False)
        time.sleep(0.1)

        # Step 4: Set the new ID
        if found_motor_info["id"] != new_id:
            logger.info(f"Setting ID from {found_motor_info['id']} to {new_id}...")
            bus.write("ID", motor_name, new_id, normalize=False)
            time.sleep(0.1)  # Wait for write to take effect
            logger.info("ID set successfully.")
        else:
            logger.info(f"Motor already has the desired ID: {new_id}.")

        # Step 5: Set the new baudrate
        if found_motor_info["baudrate"] != new_baudrate:
            logger.info(f"Setting baudrate from {found_motor_info['baudrate']} to {new_baudrate}...")
            # We need to write the index of the baudrate in the model's table
            baudrate_idx = list(bus.model_baudrate_table[motor_model].values()).index(new_baudrate)
            bus.write("Baud_Rate", motor_name, baudrate_idx, normalize=False)
            time.sleep(0.1)
            logger.info("Baudrate set successfully.")
        else:
            logger.info(f"Motor already has the desired baudrate: {new_baudrate}.")

        logger.info("Motor configuration complete.")

    except Exception as e:
        logger.error(f"An error occurred during motor configuration: {e}", exc_info=True)
    finally:
        if bus.port_handler.is_open:
            bus.port_handler.closePort()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find, unlock, and configure a single Feetech motor.")
    parser.add_argument("port", type=str, help="The serial port (e.g., /dev/ttyUSB0).")
    parser.add_argument("new_id", type=int, help="The new ID to assign to the motor.")
    parser.add_argument(
        "--motor_model",
        type=str,
        default="sts3215",
        help="The model of the motor (default: sts3215).",
    )
    parser.add_argument(
        "--new_baudrate",
        type=int,
        default=1_000_000,
        help="The new baudrate to set for the motor (default: 1,000,000).",
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=None,
        help="The baudrate to scan. If not provided, will scan all.",
    )
    parser.add_argument(
        "--protocol",
        type=int,
        default=0,
        choices=[0, 1],
        help="The protocol version to use for communication (default: 0).",
    )
    args = parser.parse_args()

    input(
        f"This script will find and reconfigure the single motor connected to '{args.port}'.\n"
        "Please ensure only ONE motor is connected. Press Enter to continue..."
    )

    setup_single_motor(
        args.port, args.motor_model, args.new_id, args.new_baudrate, args.protocol, args.baudrate
    )